"""Module for computing basic statistics on stock price data."""

import pandas as pd


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic return and volatility stats from stock price data.

    Args:
        df (pd.DataFrame): Daily OHLCV data, as returned by fetch_stock_data.

    Returns:
        dict: overall_return, volatility, best_day, worst_day (each a
            (date, return) pair for best_day/worst_day).
    """
    close = df["Close"]

    # (today's close - yesterday's close) / yesterday's close.
    daily_returns = close.pct_change().dropna()

    overall_return = (close.iloc[-1] / close.iloc[0]) - 1  # percentage change first to last close.
    volatility = daily_returns.std()  # risk of the stock.

    best_day_date = daily_returns.idxmax()
    worst_day_date = daily_returns.idxmin()

    return {
        "overall_return": overall_return,
        "volatility": volatility,
        "best_day": (best_day_date, daily_returns.loc[best_day_date]),  # (date, value in that date)
        "worst_day": (worst_day_date, daily_returns.loc[worst_day_date]),
    }


def print_stats_summary(symbol: str, start_date: str, end_date: str, stats: dict) -> None:
    """
    Print a short, readable summary of the computed stats.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        stats (dict): Output of compute_stats.
    """
    best_date, best_return = stats["best_day"]
    worst_date, worst_return = stats["worst_day"]

    print(f"{symbol} ({start_date} to {end_date})")
    print(f"Overall return: {stats['overall_return']:+.1%}")
    print(f"Volatility (daily std dev): {stats['volatility']:.1%}")
    print(f"Best day: {best_return:+.1%} on {best_date.date()}")
    print(f"Worst day: {worst_return:+.1%} on {worst_date.date()}")


def add_moving_avgs(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Add a moving average column (e.g. "SMA_20") for the given period.
    """
    col_name = f"SMA_{period}"
    df[col_name] = df["Close"].rolling(period).mean()
    return df
