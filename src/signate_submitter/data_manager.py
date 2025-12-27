"""Dataset management utilities for the Signate submitter agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"


@dataclass
class DataMeta:
    """Metadata describing the dataset that was loaded."""

    source: str
    location: str
    additional: Dict[str, str]


class DataManager:
    """Handle retrieval and caching of Signate datasets."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / "signate"
        self.submission_dir = self.cache_dir / "submissions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)

    def prepare_datasets(
        self,
        *,
        data_source: str = "auto",
        dataset_dir: str | Path | None = None,
        train_filename: str = "train.csv",
        test_filename: str = "test.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        """Return train/test dataframes plus metadata.

        Parameters
        ----------
        data_source:
            ``"auto"`` will attempt to load from ``dataset_dir`` and fall back to the
            bundled sample dataset. ``"local"`` requires ``dataset_dir`` and raises on
            failure. ``"sample"`` forces the bundled dataset.
        dataset_dir:
            Directory containing train/test files.
        """

        data_source = data_source or "auto"
        if data_source not in {"auto", "local", "sample"}:
            logging.warning("Unknown data source '%s'; defaulting to 'auto'", data_source)
            data_source = "auto"

        if data_source == "sample":
            return self._load_sample()

        if dataset_dir:
            try:
                return self._load_local(dataset_dir, train_filename, test_filename)
            except Exception as exc:
                if data_source == "local":
                    logging.error("Local dataset requested but failed: %s", exc)
                    raise
                logging.warning("Falling back to sample dataset due to local error: %s", exc)

        return self._load_sample()

    def _load_local(
        self, dataset_dir: str | Path, train_filename: str, test_filename: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        dataset_dir = Path(dataset_dir)
        train_path = dataset_dir / train_filename
        test_path = dataset_dir / test_filename
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Could not find train/test at {train_path} / {test_path}. Provide valid paths."
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        meta = DataMeta(
            source="local",
            location=str(dataset_dir),
            additional={},
        )
        logging.info("Loaded local dataset from %s", dataset_dir)
        return train_df, test_df, meta

    def _load_sample(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        """Load the bundled sample dataset for offline usage."""

        train_path = SAMPLE_DATA_DIR / "train.csv"
        test_path = SAMPLE_DATA_DIR / "test.csv"
        schema_path = SAMPLE_DATA_DIR / "schema.json"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Sample dataset is missing from the repository")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        additional: Dict[str, str] = {}
        if schema_path.exists():
            additional["schema"] = json.dumps(json.loads(schema_path.read_text(encoding="utf-8")))
        meta = DataMeta(
            source="sample",
            location=str(SAMPLE_DATA_DIR),
            additional=additional,
        )
        logging.info("Loaded bundled sample dataset from %s", SAMPLE_DATA_DIR)
        return train_df, test_df, meta


__all__ = ["DataManager", "DataMeta"]
