"""Reusable machine learning pipeline for Signate-style competitions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TAG = "#KGNINJA"


@dataclass
class SignatePipelineResult:
    """Summary of a pipeline execution."""

    cv_mean: float
    cv_std: float
    model_name: str
    submission_path: str
    data_source: str
    notes: Optional[str]
    score_metric: str


class SignatePipeline:
    """Configurable pipeline that adapts to Signate competition datasets."""

    def __init__(
        self,
        profile: str,
        random_seed: int,
        problem_type: str,
        target_column: str,
        id_column: str,
        submission_target: str,
        drop_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.profile = profile
        self.random_seed = random_seed
        self.problem_type = problem_type
        drop_columns = [drop_columns] if isinstance(drop_columns, str) else drop_columns
        self.target_column = target_column
        self.id_column = id_column
        self.submission_target = submission_target
        self.drop_columns = set(drop_columns or [])

        self.model = self._build_model()
        self._model_name = type(self.model).__name__

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        submission_name: str,
        output_dir,
        notes: Optional[str],
        data_meta,
    ) -> SignatePipelineResult:
        """Train, evaluate, and export predictions."""

        logging.info(
            "Starting Signate pipeline | profile=%s problem_type=%s model=%s",
            self.profile,
            self.problem_type,
            self._model_name,
        )
        np.random.seed(self.random_seed)

        train_df = train_df.copy()
        test_df = test_df.copy()

        self._validate_dataframe(train_df, is_train=True)
        self._validate_dataframe(test_df, is_train=False)

        X_train = train_df.drop(columns=self._columns_to_exclude(train_df, is_train=True))
        y_train = train_df[self.target_column]

        X_test = test_df.drop(columns=self._columns_to_exclude(test_df, is_train=False))

        numeric_features, categorical_features = self._split_feature_types(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(numeric_features, categorical_features)),
                ("model", self.model),
            ]
        )

        cv = self._build_cv_splitter()
        scoring = self._build_scorer()
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        metric_name = scoring
        if self.problem_type == "regression":
            scores = -scores  # convert neg RMSE into positive RMSE for readability
            metric_name = "rmse"
        logging.info("CV scores (%s): %s", scoring, scores)

        pipeline.fit(X_train, y_train)
        logging.info("Model '%s' fitted on %d samples", self._model_name, len(train_df))

        predictions = pipeline.predict(X_test)
        if self.problem_type == "classification":
            allowed_labels = set(pd.unique(y_train))
            observed_labels = set(pd.unique(predictions))
            if not observed_labels.issubset(allowed_labels):
                raise ValueError(
                    f"Predicted labels {observed_labels} contain values outside training labels {allowed_labels}"
                )
        submission_df = pd.DataFrame({self.id_column: test_df[self.id_column], self.submission_target: predictions})

        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / submission_name
        submission_df.to_csv(submission_path, index=False)
        logging.info("Submission saved to %s", submission_path)

        note_lines = [notes] if notes else []
        if data_meta.source == "sample":
            note_lines.append("Using bundled sample dataset")
        compiled_notes = " | ".join([n for n in note_lines if n]) if note_lines else None

        return SignatePipelineResult(
            cv_mean=float(scores.mean()),
            cv_std=float(scores.std()),
            model_name=self._model_name,
            submission_path=str(submission_path),
            data_source=data_meta.source,
            notes=compiled_notes,
            score_metric=metric_name if isinstance(metric_name, str) else "custom",
        )

    def _columns_to_exclude(self, df: pd.DataFrame, *, is_train: bool) -> Iterable[str]:
        columns = {self.id_column, *self.drop_columns}
        if is_train:
            columns.add(self.target_column)
        return columns

    def _split_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_features = [
            col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)
        ]
        categorical_features = [col for col in df.columns if col not in numeric_features]
        return numeric_features, categorical_features

    def _build_preprocessor(
        self, numeric_features: Sequence[str], categorical_features: Sequence[str]
    ) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, list(numeric_features)),
                ("cat", categorical_transformer, list(categorical_features)),
            ]
        )

    def _build_model(self):
        if self.problem_type == "classification":
            return self._classification_model()
        return self._regression_model()

    def _classification_model(self):
        if self.profile == "power":
            return RandomForestClassifier(
                n_estimators=200, random_state=self.random_seed, n_jobs=-1, class_weight="balanced"
            )
        if self.profile == "boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(random_state=self.random_seed)
        return LogisticRegression(max_iter=200, random_state=self.random_seed)

    def _regression_model(self):
        if self.profile == "power":
            return RandomForestRegressor(
                n_estimators=300,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        if self.profile == "boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=self.random_seed)
        return Ridge(random_state=self.random_seed)

    def _build_cv_splitter(self):
        if self.problem_type == "classification":
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        return KFold(n_splits=5, shuffle=True, random_state=self.random_seed)

    def _build_scorer(self):
        if self.problem_type == "classification":
            return "accuracy"
        return "neg_root_mean_squared_error"

    def _validate_dataframe(self, df: pd.DataFrame, *, is_train: bool) -> None:
        required = {self.id_column, *self.drop_columns}
        if is_train:
            required.add(self.target_column)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")


__all__ = ["SignatePipeline", "SignatePipelineResult", "TAG"]
