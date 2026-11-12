"""The vectorised grid: many trials in one call must equal the same trials run one by one."""

import copy

import numpy as np
import pytest

from openstrategy_studio import engine, optimise as O
from openstrategy_studio.spec import load_template


def test_eligibility(ma_spec, momentum_spec):
    space = O.build_space(ma_spec)
    assert engine.vectorisable(ma_spec, space) == (True, "")
    ok, reason = engine.vectorisable(momentum_spec, O.build_space(momentum_spec))
    assert not ok and "entry and exit rule" in reason
    risky = copy.deepcopy(ma_spec)
    risky["risk"]["stop"]["points"] = {"value": 60, "opt": {"min": 20, "max": 100, "int": True}}
    ok, reason = engine.vectorisable(risky, O.build_space(risky))
    assert not ok and "risk.stop.points" in reason
    search = load_template("structure_search")
    ok, reason = engine.vectorisable(search, O.build_space(search, conditional=True))
    assert not ok and "Alternatives" in reason


def test_vectorised_equals_one_by_one(ma_spec, ohlcv):
    pytest.importorskip("vectorbtx")
    space = O.build_space(ma_spec)
    params_list = [{"features.fast.window": fast, "features.slow.window": slow}
                   for fast in (10, 30, 50) for slow in (100, 200)]
    pf, columns = engine.run_vectorised(ma_spec, ohlcv, params_list)
    job = {"metric": "sharpe", "constraints": {}}
    for params, column in zip(params_list, columns):
        alone = engine.run_backtest(ma_spec, ohlcv, params=params)["pf"]
        together = O.score(pf[column], job, params)
        single = O.score(alone, job, params)
        for key in ("objective", "total_return", "n_trades", "max_drawdown"):
            assert together[key] == pytest.approx(single[key], nan_ok=True), (params, key)
    assert np.isfinite(O.grid_size(space))
