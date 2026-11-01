"""Strategy (prediction engine) interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Turns feature rows into (prob_up, expected_return) per symbol.

    `fit` only ever receives data from BEFORE the trading day being simulated
    (the backtester enforces walk-forward training). Strategies must not
    peek at anything else.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.fitted = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> bool:
        """Train on rows with FEATURES and `fwd_ret`. Return True if usable."""

    @abstractmethod
    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like `rows` with columns prob_up, expected_return."""
