"""Module for fetching stock data from Yahoo Finance."""

import re
from datetime import date

import curl_cffi.requests.exceptions as curl_exceptions
import pandas as pd
import yfinance
import yfinance.exceptions as yf_exceptions


def validate_symbol(symbol: str) -> str:
    """
    Validate a stock ticker symbol and return it normalized (uppercase).
    """
    if not symbol:
        raise ValueError("Symbol must not be empty.")

    normalized = symbol.strip().upper()

    if not re.fullmatch(r"[A-Z]{1,5}", normalized):
        raise ValueError(f"Invalid stock symbol: {symbol!r}")

    return normalized


def validate_date_format(date_str: str) -> date:
    """
    Validate that a date string is in the 'YYYY-MM-DD' format.
    """
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid date format: {date_str!r}. Expected 'YYYY-MM-DD'.")


def validate_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    """
    Validate a start/end date pair: correct format and start before end.

    """
    start = validate_date_format(start_date)
    end = validate_date_format(end_date)

    if start >= end:
        raise ValueError(f"start_date {start_date!r} must be before end_date {end_date!r}.")

    return start, end


def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data for a given symbol between start_date and end_date.

    args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Daily OHLCV data indexed by date.
    """
    symbol = validate_symbol(symbol)
    start, end = validate_date_range(start_date, end_date)

    try:
        df = yfinance.download(symbol, start=start, end=end)
    except yf_exceptions.YFException as e:
        raise RuntimeError(f"Yahoo Finance error for {symbol!r}: {e}") from e
    except curl_exceptions.RequestException as e:
        raise RuntimeError(f"Network error while fetching {symbol!r}: {e}") from e

    if df.empty:
        raise ValueError(f"No data found for symbol {symbol!r} in the given date range.")

    # yfinance always adds a Ticker column level (for multi-symbol downloads);
    # drop it since we only ever fetch one symbol, so df["Close"] etc. stay Series.
    df.columns = df.columns.droplevel("Ticker")

    return df
