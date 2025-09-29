import numpy as np
import pandas as pd

from src.dataset import RepresentationBias, MeasurementBias, SamplingBias, LabelBias


def _make_base(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "protected_attribute": rng.integers(0, 2, n),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    return df


def test_representation_bias_boundaries_and_invariants():
    df = _make_base()
    # reduction_factor=1.0 -> unchanged
    rb = RepresentationBias("protected_attribute", 1, 1.0)
    out = rb.transform(df)
    pd.testing.assert_frame_equal(out.sort_index(axis=1), df.sort_index(axis=1))

    # reduction_factor=0.0 -> remove all group 1 rows
    rb0 = RepresentationBias("protected_attribute", 1, 0.0)
    out0 = rb0.transform(df)
    assert (out0["protected_attribute"] == 1).sum() == 0
    # group 0 rows unchanged in count
    assert (out0["protected_attribute"] == 0).sum() == (df["protected_attribute"] == 0).sum()


def test_measurement_bias_affects_only_feature_and_group():
    df = _make_base()
    mb = MeasurementBias(feature="x", protected_attribute="protected_attribute", affected_group=1, noise_std=5.0)
    out = mb.transform(df)
    # Row count unchanged
    assert len(out) == len(df)
    # Target unchanged
    pd.testing.assert_series_equal(df["target"].reset_index(drop=True), out["target"].reset_index(drop=True))
    # Unaffected columns except feature remain equal
    pd.testing.assert_series_equal(df["y"].reset_index(drop=True), out["y"].reset_index(drop=True))
    pd.testing.assert_series_equal(df["protected_attribute"].reset_index(drop=True), out["protected_attribute"].reset_index(drop=True))
    # Variance increases for affected group on feature
    var_orig = df.loc[df["protected_attribute"] == 1, "x"].var()
    var_out = out.loc[out["protected_attribute"] == 1, "x"].var()
    assert var_out > var_orig


def test_sampling_bias_boundaries_and_exact_rate():
    df = _make_base()
    # rate=1.0 -> unchanged
    sb1 = SamplingBias("protected_attribute", "target", affected_group=1, label_value=1, sampling_rate=1.0)
    out1 = sb1.transform(df)
    pd.testing.assert_frame_equal(out1.sort_index(axis=1), df.sort_index(axis=1))

    # rate=0.0 -> remove all rows of (group=1,label=1)
    sb0 = SamplingBias("protected_attribute", "target", affected_group=1, label_value=1, sampling_rate=0.0)
    out0 = sb0.transform(df)
    assert ((out0["protected_attribute"] == 1) & (out0["target"] == 1)).sum() == 0

    # exact rate 0.5 -> count halves
    rng = np.random.default_rng(123)
    df2 = _make_base(seed=123, n=2000)
    sb = SamplingBias("protected_attribute", "target", affected_group=1, label_value=1, sampling_rate=0.5)
    out = sb.transform(df2)
    orig_count = ((df2["protected_attribute"] == 1) & (df2["target"] == 1)).sum()
    out_count = ((out["protected_attribute"] == 1) & (out["target"] == 1)).sum()
    assert out_count == int(orig_count * 0.5)


def test_label_bias_binary_and_multiclass_behavior():
    # Binary
    df = _make_base()
    lb = LabelBias("protected_attribute", "target", affected_group=1, flip_probability=0.3)
    out = lb.transform(df)
    # Row count unchanged
    assert len(out) == len(df)
    # Affected group's positive rate should change
    rate_orig = df.loc[df["protected_attribute"] == 1, "target"].mean()
    rate_out = out.loc[out["protected_attribute"] == 1, "target"].mean()
    assert abs(rate_out - rate_orig) > 0.005

    # Multiclass (non-binary): ensure flips go to a different class
    dfm = _make_base()
    # Make target 3-class
    dfm["target"] = np.random.default_rng(0).integers(0, 3, len(dfm))
    lbm = LabelBias("protected_attribute", "target", affected_group=1, flip_probability=0.5)
    outm = lbm.transform(dfm)
    changed = outm.loc[outm["protected_attribute"] == 1, "target"] != dfm.loc[dfm["protected_attribute"] == 1, "target"]
    assert changed.any()

