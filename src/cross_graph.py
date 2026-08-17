import matplotlib.pyplot as plt
import pandas as pd


def cross_graph(df: pd.DataFrame) -> None:
    """
    Generate crossover chart for a specific stock.
    """
    width, height = 12, 6
    plt.figure(figsize=(width, height))
    plt.title("Crossover chart")

    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, df["SMA_20"], label="SMA 20")
    plt.plot(df.index, df["SMA_50"], label="SMA 50")

    buy_points = df[df["signal"] == 1]
    sell_points = df[df["signal"] == -1]
    plt.scatter(buy_points.index, buy_points["Close"], marker="^", color="green", label="Buy")
    plt.scatter(sell_points.index, sell_points["Close"], marker="v", color="red", label="Sell")

    plt.legend()
    plt.show()
