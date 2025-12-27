"""Tests for the signate_submitter pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from signate_submitter.agent import build_success_result, resolve_problem_type
from signate_submitter.data_manager import DataManager, SAMPLE_DATA_DIR
from signate_submitter.pipeline import SignatePipeline, SignatePipelineResult


def test_pipeline_runs_on_sample_data(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(data_source="sample")
    pipeline = SignatePipeline(
        profile="fast",
        random_seed=7,
        problem_type="classification",
        target_column="Survived",
        id_column="PassengerId",
        submission_target="Survived",
        drop_columns=["Name", "Ticket"],
    )
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name="test_submission.csv",
        output_dir=manager.submission_dir,
        notes="pytest",
        data_meta=meta,
    )

    assert 0.0 <= result.cv_mean <= 1.0
    assert Path(result.submission_path).exists()
    assert result.data_source == "sample"
    submission_df = pd.read_csv(result.submission_path)
    assert list(submission_df.columns) == ["PassengerId", "Survived"]


def test_success_result_contains_required_metadata(tmp_path):
    dummy_result = SignatePipelineResult(
        cv_mean=0.5,
        cv_std=0.1,
        model_name="LogisticRegression",
        submission_path=str(tmp_path / "submission.csv"),
        data_source="sample",
        notes=None,
        score_metric="accuracy",
    )
    agent_result = build_success_result(
        run_id="test-run",
        log_path=tmp_path / "log.txt",
        result=dummy_result,
        profile="fast",
        problem_type="classification",
    )
    assert agent_result.ok is True
    assert "#KGNINJA" in agent_result.meta["tags"]
    assert agent_result.result["score_metric"] == "accuracy"


def test_problem_type_auto_detection_prefers_classification():
    train_df = pd.read_csv(SAMPLE_DATA_DIR / "train.csv")
    problem_type = resolve_problem_type(
        type("dummy", (), {"problem_type": "auto", "target_column": "Survived"})(),
        train_df,
    )
    assert problem_type == "classification"
