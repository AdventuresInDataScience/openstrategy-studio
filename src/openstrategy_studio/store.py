"""SQLite index under <data_dir>/catalogue.db: stored datasets, saved runs, jobs.

The files on disk (parquet prices, run folders, job folders) are the data; these
tables only index them so pages can list things without walking folders.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

# A dataset is one NAME at one timeframe. `symbol` holds the name: what strategies pick and the
# folder the parquet sits in (SPY, SPY_csv, EURUSD_dukascopy). `ticker` is the source's own
# symbol, which a Yahoo refresh downloads (SPY, EURUSD=X); for an import it is what was typed.
SCHEMA = """
CREATE TABLE IF NOT EXISTS catalogue (
    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, source TEXT, first_bar TEXT, last_bar TEXT,
    n_bars INTEGER, adjusted INTEGER, tz TEXT, imported_at TEXT, notes TEXT, path TEXT, ticker TEXT,
    PRIMARY KEY (symbol, timeframe)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, spec_hash TEXT, created_at TEXT, summary TEXT, path TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, name TEXT, kind TEXT, backend TEXT, created_at TEXT, path TEXT
);
"""

CATALOGUE_COLUMNS = ["symbol", "timeframe", "source", "first_bar", "last_bar", "n_bars", "adjusted",
                     "tz", "imported_at", "notes", "path", "ticker"]


def _connect(data_dir):
    path = Path(data_dir) / "catalogue.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    # A catalogue made before datasets had a ticker gains the column; its rows read the
    # name as their ticker (list_datasets), which is what they were downloaded under.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(catalogue)")}
    if "ticker" not in columns:
        conn.execute("ALTER TABLE catalogue ADD COLUMN ticker TEXT")
        conn.commit()
    return conn


def _write(data_dir, sql, values=()):
    # closing() closes the connection; `with conn` commits (or rolls back on error).
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute(sql, values)


def _read(data_dir, sql, values=()):
    with closing(_connect(data_dir)) as conn:
        return pd.read_sql_query(sql, conn, params=values)


# --- datasets ----------------------------------------------------------------

def upsert_dataset(data_dir, row):
    values = [row.get(column) for column in CATALOGUE_COLUMNS]
    placeholders = ", ".join("?" for _ in CATALOGUE_COLUMNS)
    _write(data_dir, f"INSERT OR REPLACE INTO catalogue ({', '.join(CATALOGUE_COLUMNS)}) VALUES ({placeholders})",
           values)


def list_datasets(data_dir):
    found = _read(data_dir, f"SELECT {', '.join(CATALOGUE_COLUMNS)} FROM catalogue ORDER BY symbol, timeframe")
    found["ticker"] = found["ticker"].fillna(found["symbol"])
    return found


def get_dataset(data_dir, symbol, timeframe):
    found = _read(data_dir, "SELECT * FROM catalogue WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
    return None if found.empty else found.iloc[0].to_dict()


def delete_dataset(data_dir, symbol, timeframe):
    _write(data_dir, "DELETE FROM catalogue WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))


def rename_dataset(data_dir, symbol, timeframe, new, path):
    _write(data_dir, "UPDATE catalogue SET symbol = ?, path = ? WHERE symbol = ? AND timeframe = ?",
           (new, str(path), symbol, timeframe))


# --- runs --------------------------------------------------------------------

def add_run(data_dir, run_id, name, spec_hash, created_at, summary, path):
    _write(data_dir, "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
           (run_id, name, spec_hash, created_at, summary, str(path)))


def list_runs(data_dir):
    return _read(data_dir, "SELECT * FROM runs ORDER BY created_at DESC")


def delete_run(data_dir, run_id):
    _write(data_dir, "DELETE FROM runs WHERE run_id = ?", (run_id,))


# --- jobs --------------------------------------------------------------------

def add_job(data_dir, job_id, name, kind, backend, created_at, path):
    _write(data_dir, "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
           (job_id, name, kind, backend, created_at, str(path)))


def list_jobs(data_dir):
    return _read(data_dir, "SELECT * FROM jobs ORDER BY created_at DESC")


def delete_job(data_dir, job_id):
    _write(data_dir, "DELETE FROM jobs WHERE job_id = ?", (job_id,))
