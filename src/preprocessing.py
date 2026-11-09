"""Raw-session contracts and unfitted, serializable sklearn pipeline factories.

Fit the complete model pipeline on frozen training rows or a CV fit fold.
Factories never read data or learn categories/statistics. See reports/preprocessing.md.
"""

from numbers import Real
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.feature_sets import (
    ALL_17,
    BOOLEAN,
    CODE_CATEGORICAL,
    FEATURE_SETS,
    NOMINAL_CATEGORICAL,
    NUMERIC,
    PAGE_COUNTS,
)

TARGET = "Revenue"


def _columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; use {list(FEATURE_SETS)}."
        )
    return list(FEATURE_SETS[feature_set])


def _check_frame(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Supply raw sessions as a pandas DataFrame with named columns."
        )
    if data.empty:
        raise ValueError("Supply at least one session.")
    if data.columns.has_duplicates:
        raise ValueError("Duplicate column names are not allowed.")


def _real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _numeric(series: pd.Series) -> np.ndarray:
    if series.isna().any() or not series.map(_real).all():
        raise ValueError(
            f"{series.name} must contain finite real numbers, not blanks/Booleans/text."
        )
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{series.name} must contain finite real numbers.")
    return values


def _booleans(series: pd.Series) -> np.ndarray:
    """Normalize explicit binary representations without Python truthiness."""
    values = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            values.append(int(value))
        elif _real(value) and value in (0, 1):
            values.append(int(value))
        elif isinstance(value, str) and value.strip().lower() in ("true", "false"):
            values.append(int(value.strip().lower() == "true"))
        else:
            raise ValueError(
                f"{series.name} requires true/false, numeric 0/1, or 'true'/'false' text; "
                "missing or ambiguous values are not allowed."
            )
    return np.asarray(values, dtype=np.int8)


def _normalize_inputs(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    _check_frame(data)
    columns = _columns(feature_set)
    if TARGET in data.columns:
        raise ValueError(
            "Revenue is the target, not an input; use split_features_target first."
        )
    missing = [col for col in columns if col not in data.columns]
    extra = [col for col in data.columns if col not in ALL_17]
    if missing or extra:
        raise ValueError(
            f"Input schema mismatch; missing={missing}, unexpected={extra}."
        )
    # Known excluded predictors may be present for matched feature comparisons.
    # They are dropped here and never validated, encoded or passed to a model.
    result = data.loc[:, columns].copy()
    for col in columns:
        series = result[col]
        if col in NUMERIC or col in CODE_CATEGORICAL:
            values = _numeric(series)
            if (values < 0).any():
                raise ValueError(f"{col} must be nonnegative.")
            if col in PAGE_COUNTS or col in CODE_CATEGORICAL:
                if (values != np.floor(values)).any():
                    raise ValueError(f"{col} must contain whole numbers.")
            if col in CODE_CATEGORICAL and (values == 0).any():
                raise ValueError(f"{col} category codes must be positive integers.")
            if col in ("BounceRates", "ExitRates", "SpecialDay") and (values > 1).any():
                raise ValueError(f"{col} must be a fraction in [0, 1].")
            result[col] = values
        elif col in BOOLEAN:
            result[col] = _booleans(series)
        else:
            if (
                series.isna().any()
                or not series.map(
                    lambda value: isinstance(value, str) and bool(value.strip())
                ).all()
            ):
                raise ValueError(f"{col} must contain nonblank category strings.")
    return result


class SessionInputValidator(TransformerMixin, BaseEstimator):
    """Select a raw feature variant, validate values and normalize Booleans.

    No imputation, clipping or data-dependent validation is performed. Column
    order is normalized; unknown category levels remain distinct from blanks.
    """

    # sklearn fitted-state checks require these attributes to be absent until fit.
    feature_names_in_: np.ndarray  # pyright: ignore[reportUninitializedInstanceVariable]
    n_features_in_: int  # pyright: ignore[reportUninitializedInstanceVariable]
    output_columns_: np.ndarray  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(self, feature_set: str = "all_17"):
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y=None):
        normalized = _normalize_inputs(X, self.feature_set)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.output_columns_ = np.asarray(normalized.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_columns_")
        return _normalize_inputs(X, self.feature_set)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "output_columns_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
        return self.output_columns_.copy()


def split_features_target(
    data: pd.DataFrame, feature_set: str = "all_17"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate Revenue (1=purchase) and return validated raw inputs in source order.

    This operation is stateless and learns no vocabulary, imputation or scaling.
    The pipeline repeats validation so reloaded inference has the same contract.
    """
    _check_frame(data)
    if TARGET not in data.columns:
        raise ValueError("Revenue is required to separate the target from inputs.")
    X = _normalize_inputs(data.drop(columns=TARGET), feature_set)
    # _check_frame rejects duplicate columns, so this lookup returns one Series.
    target = cast(pd.Series, data[TARGET])
    y = pd.Series(_booleans(target), index=data.index, name=TARGET)
    return X, y


def make_preprocessor(
    feature_set: str = "all_17", *, scale_numeric: bool = True
) -> Pipeline:
    """Return an UNFITTED validator and column transformer.

    Use scale_numeric=True for logistic regression; False for RF/XGBoost.
    One-hot vocabulary and (optional) mean/variance are learned only at fit.
    """
    columns = _columns(feature_set)
    if not isinstance(scale_numeric, bool):
        raise ValueError("scale_numeric must be True or False.")
    return Pipeline(
        [
            ("schema", SessionInputValidator(feature_set)),
            (
                "columns",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            StandardScaler() if scale_numeric else "passthrough",
                            [col for col in NUMERIC if col in columns],
                        ),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            list(NOMINAL_CATEGORICAL),
                        ),
                        ("boolean", "passthrough", list(BOOLEAN)),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )


def make_model_pipeline(
    estimator: BaseEstimator,
    feature_set: str = "all_17",
    *,
    scale_numeric: bool = True,
) -> Pipeline:
    """Combine a fresh clone of an estimator with the unfitted raw-input pipeline.

    Pass this whole object to training CV, then fit on training and save it.
    Tune estimator parameters with model__..., e.g. model__C for logistic regression.
    """
    return Pipeline(
        [
            ("preprocess", make_preprocessor(feature_set, scale_numeric=scale_numeric)),
            ("model", clone(estimator)),
        ]
    )
