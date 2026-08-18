import argparse

import cross_graph
import fetch_data
import signals
import stats


def main():
    parser = argparse.ArgumentParser(description="Show crossover graph for certain stock")
    parser.add_argument("stock", type=str, help="The ticker of the stock to show the graph of.")
    parser.add_argument("start_date", type=str, help="start date of period")
    parser.add_argument("end_date", type=str, help="end date of the period")
    args = parser.parse_args()

    df = fetch_data.fetch_stock_data(args.stock, args.start_date, args.end_date)
    stock_stats = stats.compute_stats(args.stock, df)

    df = stats.add_moving_avgs(df, 20)
    df = stats.add_moving_avgs(df, 50)
    mod_df = signals.detect_cross(df)

    stats.print_stats_summary(args.start_date, args.end_date, stock_stats)
    cross_graph.cross_graph(mod_df)


if __name__ == "__main__":
    main()
