import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append("..")

from src.preprocess import MissingImputer, SkewnessCorrector, FeatureEncoder, Binner


def test_missing_imputer_mean_median_mode_and_fit_guard():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [10, 20, 30, 40]})
    # fit guard
    with pytest.raises(ValueError):
        MissingImputer("mean", "a").transform(df)

    mi = MissingImputer("mean", "a").fit(df)
    out = mi.transform(df)
    assert out["a"].isna().sum() == 0
    # only column a modified
    pd.testing.assert_series_equal(out["b"], df["b"])

    mi = MissingImputer("median", "a").fit(df)
    out2 = mi.transform(df)
    assert out2["a"].isna().sum() == 0

    mi = MissingImputer("mode", "a").fit(df)
    out3 = mi.transform(df)
    assert out3["a"].isna().sum() == 0

    with pytest.raises(ValueError):
        MissingImputer("unknown", "a").fit(df)


def test_skewness_corrector_domain_handling():
    df = pd.DataFrame({"x": [-5.0, -1.0, 0.0, 1.0, 5.0]})

    out_log = SkewnessCorrector("log", "x").transform(df)
    assert np.isfinite(out_log["x"]).all()

    out_sqrt = SkewnessCorrector("sqrt", "x").transform(df)
    assert (out_sqrt["x"] >= 0).all()

    out_boxcox = SkewnessCorrector("box_cox", "x").transform(df)
    assert np.isfinite(out_boxcox["x"]).all()

    out_exp = SkewnessCorrector("exp", "x").transform(df)
    assert (out_exp["x"] > 0).all()


def test_feature_encoder_label_and_onehot():
    df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
    # fit guard
    enc = FeatureEncoder("label", "cat")
    with pytest.raises(ValueError):
        enc.transform(df)
    enc.fit(df, None)
    out = enc.transform(df)
    assert set(out.columns) == {"cat"}

    enc2 = FeatureEncoder("onehot", "cat")
    with pytest.raises(ValueError):
        enc2.transform(df)
    enc2.fit(df, None)
    out2 = enc2.transform(df)
    # onehot expands and drops original
    assert "cat" not in out2.columns
    assert any(col.startswith("cat_") for col in out2.columns)


def test_binner_quantile_and_custom():
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 100)})
    # fit guard
    with pytest.raises(ValueError):
        Binner("quantile", "x", n_bins=5).transform(df)

    bq = Binner("quantile", "x", n_bins=5).fit(df)
    out = bq.transform(df)
    assert out["x"].isna().sum() == 0

    with pytest.raises(ValueError):
        Binner("custom", "x", custom_bins=[0.0]).fit(df)

    bc = Binner("custom", "x", custom_bins=[0.0, 0.5, 1.0], labels=["L", "H"]).fit(df)
    outc = bc.transform(df)
    assert set(outc["x"].unique()) <= {"L", "H"}


