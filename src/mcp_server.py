"""MCP server exposing the stock analysis pipeline as async tools."""

import asyncio
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcp.server.mcpserver import Image, MCPServer

import cross_graph
import fetch_data
import signals
import stats

mcp = MCPServer(name="stock-analyzer")


def _prepare_df(symbol: str, start_date: str, end_date: str):
    df = fetch_data.fetch_stock_data(symbol, start_date, end_date)
    df = stats.add_moving_avgs(df, 20)
    df = stats.add_moving_avgs(df, 50)
    return df


@mcp.tool()
async def get_stock_stats(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Get overall return, volatility, and best/worst day for a stock over a date range.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
    """
    df = await asyncio.to_thread(fetch_data.fetch_stock_data, symbol, start_date, end_date)
    return stats.compute_stats(symbol, df)


@mcp.tool()
async def get_crossover_signals(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """
    Find golden-cross (buy) and death-cross (sell) signal dates for a stock.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
    """
    df = await asyncio.to_thread(_prepare_df, symbol, start_date, end_date)
    df = signals.detect_cross(df)
    events = df[df["signal"].notna()]

    return [
        {
            "date": str(date.date()),
            "price": round(float(row["Close"]), 2),
            "signal": "buy" if row["signal"] == 1 else "sell",
        }
        for date, row in events.iterrows()
    ]


@mcp.tool()
async def get_crossover_chart(symbol: str, start_date: str, end_date: str) -> Image:
    """
    Render a price / moving-average / crossover chart for a stock as an image.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
    """
    df = await asyncio.to_thread(_prepare_df, symbol, start_date, end_date)
    df = signals.detect_cross(df)

    def render() -> bytes:
        cross_graph.cross_graph(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf.getvalue()

    png_bytes = await asyncio.to_thread(render)
    return Image(data=png_bytes, format="png")


if __name__ == "__main__":
    mcp.run()
