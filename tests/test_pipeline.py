import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.dataset import RepresentationBias
from src.preprocess import FeatureEncoder
from src.pipeline import Pipeline


def _make_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "protected_attribute": rng.integers(0, 2, n),
        "feat": rng.normal(size=n),
        "cat": rng.choice(["a", "b"], n),
        "target": rng.integers(0, 2, n),
    })


def test_bias_skipped_on_eval_transform_predict():
    df = _make_df()
    bias = RepresentationBias("protected_attribute", underrepresented_group=1, reduction_factor=0.5)
    enc = FeatureEncoder("onehot", "cat")
    model = LogisticRegression(max_iter=1000)

    pipe = Pipeline([
        ("bias", bias),
        ("enc", enc),
        ("model", model),
    ])
    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    pipe.fit(train, target_column="target")

    # Bias applied on train: rows should reduce
    after_bias_train_rows = bias.transform(train).shape[0]
    assert after_bias_train_rows < len(train)

    # Transform: bias should be skipped
    transformed_test = pipe.transform(test)
    assert len(transformed_test) == len(test)

    # Predict runs without errors and returns correct length
    preds = pipe.predict(test)
    assert len(preds) == len(test)


def test_pipeline_api_guards_and_predict_proba():
    df = _make_df()
    enc = FeatureEncoder("onehot", "cat")
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([
        ("enc", enc),
        ("model", model),
    ])

    # Not fitted predict raises
    try:
        pipe.predict(df)
        assert False, "predict should raise before fit"
    except ValueError:
        pass

    # Fit and use predict_proba
    pipe.fit(df, target_column="target")
    proba = pipe.predict_proba(df)
    assert proba.shape[0] == len(df)
    assert proba.shape[1] == 2

    # score requires target column present
    score = pipe.score(df, target_column="target")
    assert isinstance(score, float)


