from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from fetch_data import (
    fetch_stock_data,
    validate_date_format,
    validate_date_range,
    validate_symbol,
)


class TestValidateSymbol:
    def test_normalizes_lowercase_symbol(self):
        assert validate_symbol("aapl") == "AAPL"

    def test_strips_whitespace(self):
        assert validate_symbol("  aapl  ") == "AAPL"

    def test_empty_symbol_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_symbol("")

    def test_invalid_characters_raise(self):
        with pytest.raises(ValueError, match="Invalid stock symbol"):
            validate_symbol("AAPL1")

    def test_too_long_symbol_raises(self):
        with pytest.raises(ValueError, match="Invalid stock symbol"):
            validate_symbol("TOOLONG")


class TestValidateDateFormat:
    def test_valid_date_string(self):
        assert validate_date_format("2024-01-15") == date(2024, 1, 15)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_format("01/15/2024")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_format(None)


class TestValidateDateRange:
    def test_valid_range(self):
        start, end = validate_date_range("2024-01-01", "2024-02-01")
        assert start == date(2024, 1, 1)
        assert end == date(2024, 2, 1)

    def test_start_equal_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_date_range("2024-01-01", "2024-01-01")

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_date_range("2024-02-01", "2024-01-01")


class TestFetchStockData:
    def _make_download_df(self):
        columns = pd.MultiIndex.from_product(
            [["Close", "Open"], ["AAPL"]], names=[None, "Ticker"]
        )
        return pd.DataFrame(
            [[100.0, 99.0], [101.0, 100.0]],
            columns=columns,
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    @patch("fetch_data.yfinance.download")
    def test_drops_ticker_column_level(self, mock_download):
        mock_download.return_value = self._make_download_df()

        df = fetch_stock_data("aapl", "2024-01-01", "2024-02-01")

        assert list(df.columns) == ["Close", "Open"]
        mock_download.assert_called_once_with(
            "AAPL", start=date(2024, 1, 1), end=date(2024, 2, 1)
        )

    @patch("fetch_data.yfinance.download")
    def test_empty_result_raises(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data found"):
            fetch_stock_data("AAPL", "2024-01-01", "2024-02-01")

    def test_invalid_symbol_raises_before_network_call(self):
        with pytest.raises(ValueError, match="Invalid stock symbol"):
            fetch_stock_data("BAD1", "2024-01-01", "2024-02-01")
