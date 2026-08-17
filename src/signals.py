import pandas as pd


def detect_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect crossovers between the SMA_20 and SMA_50 columns.

    Returns:
        pd.DataFrame: A copy of df with a "signal" column: 1 on golden-cross
            days, -1 on death-cross days, NaN elsewhere.
    """
    required_cols = {"SMA_20", "SMA_50"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"df is missing required column(s): {sorted(missing)}. "
            "Run add_moving_avgs(df, 20) and add_moving_avgs(df, 50) first."
        )

    modified_df = df.copy()

    # Bullish crossover: short-term MA crosses above long-term MA.
    bullish_cross = (modified_df["SMA_20"] > modified_df["SMA_50"]) & (
        modified_df["SMA_20"].shift(1) <= modified_df["SMA_50"].shift(1)
    )
    modified_df.loc[bullish_cross, "signal"] = 1  # add a signal of 1 for bullish crossover days

    # Bearish crossover: short-term MA crosses below long-term MA.
    bearish_cross = (modified_df["SMA_20"] < modified_df["SMA_50"]) & (
        modified_df["SMA_20"].shift(1) >= modified_df["SMA_50"].shift(1)
    )
    modified_df.loc[bearish_cross, "signal"] = -1  # add a signal of -1 for bearish crossover days
    # rest of the days will have NaN in the "signal" column.

    return modified_df
