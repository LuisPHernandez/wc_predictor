import pandas as pd
import sys

def merge_csv(file1: str, file2: str, output: str) -> None:
    df = pd.concat(
        [pd.read_csv(file1), pd.read_csv(file2)],
        ignore_index=True
    )

    df["match_date"] = pd.to_datetime(df["match_date"])
    df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])

    df.sort_values(
        by=["match_date", "home_team", "away_team", "prediction_timestamp"],
        inplace=True,
    )

    df["match_date"] = df["match_date"].dt.strftime("%Y-%m-%d")
    df["prediction_timestamp"] = df["prediction_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(output, index=False)
    print(f"Merged {len(df)} rows → {output}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_predictions.py <file1.csv> <file2.csv> <output.csv>")
        sys.exit(1)

    merge_csv(sys.argv[1], sys.argv[2], sys.argv[3])