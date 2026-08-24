import numpy as np
import pandas as pd
import pytest

from signals import detect_cross


def _make_df(sma_20, sma_50):
    return pd.DataFrame(
        {"SMA_20": sma_20, "SMA_50": sma_50},
        index=pd.date_range("2024-01-01", periods=len(sma_20)),
    )


class TestDetectCross:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"SMA_20": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required column"):
            detect_cross(df)

    def test_missing_both_columns_lists_both(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match=r"\['SMA_20', 'SMA_50'\]"):
            detect_cross(df)

    def test_golden_cross_detected(self):
        # short crosses above long between day 1 and day 2.
        df = _make_df(sma_20=[10, 25], sma_50=[20, 20])

        result = detect_cross(df)

        assert result["signal"].isna().iloc[0]
        assert result["signal"].iloc[1] == 1

    def test_death_cross_detected(self):
        # short crosses below long between day 1 and day 2.
        df = _make_df(sma_20=[25, 15], sma_50=[20, 20])

        result = detect_cross(df)

        assert result["signal"].isna().iloc[0]
        assert result["signal"].iloc[1] == -1

    def test_no_crossover_stays_nan(self):
        # short stays above long throughout, no crossover.
        df = _make_df(sma_20=[25, 26, 27], sma_50=[20, 20, 20])

        result = detect_cross(df)

        assert result["signal"].isna().all()

    def test_touching_without_crossing_is_not_a_signal(self):
        # short equals long then drops back to equal, never actually crosses.
        df = _make_df(sma_20=[20, 20, 20], sma_50=[20, 20, 20])

        result = detect_cross(df)

        assert result["signal"].isna().all()

    def test_multiple_crossovers(self):
        df = _make_df(
            sma_20=[10, 25, 15, 30],
            sma_50=[20, 20, 20, 20],
        )

        result = detect_cross(df)

        assert np.isnan(result["signal"].iloc[0])
        assert result["signal"].iloc[1] == 1
        assert result["signal"].iloc[2] == -1
        assert result["signal"].iloc[3] == 1

    def test_does_not_mutate_original_df(self):
        df = _make_df(sma_20=[10, 25], sma_50=[20, 20])

        detect_cross(df)

        assert "signal" not in df.columns

    def test_preserves_other_columns(self):
        df = _make_df(sma_20=[10, 25], sma_50=[20, 20])
        df["Close"] = [100, 105]

        result = detect_cross(df)

        assert list(result["Close"]) == [100, 105]
