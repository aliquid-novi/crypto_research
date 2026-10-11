import pandas as pd
from pathlib import Path

# Desktop loading
DATA_ROOT = Path(
    r"../data/"
)

def get_md(
    exchange,
    symbol,
    data_type="book",
    date=None,
):
    """Takes exchange, symbol, and data_type, in that order.
    

    """
    
    path = (
        DATA_ROOT
        / exchange.upper()
        / symbol.upper()
        / data_type
    )

    if date is not None:
        path = path / date

    if not path.exists():
        raise FileNotFoundError(
            f"No data found at:\n{path}"
        )

    df = pd.read_parquet(path)

    df["datetime"] = pd.to_datetime(
        df["receipt_timestamp"],
        unit="s",
        utc=True
    )
    
    df = (
        df
        .set_index("datetime")
        .sort_index()
    )

    return df

import pandas as pd

def add_features(
    df_1, df_1_name,
    df_2, df_2_name,
    resample_ms,
    max_quote_age_ms=None
):
    """
    df_1 - quoter leg
    df_2 - hedging leg
    resample_ms - resampling interval in milliseconds
    max_quote_age_ms - optional maximum quote age in milliseconds
                       None means no filtering
    """

    columns = [
        "mid", "bid", "ask", "bid_size",
        "ask_size", "imbalance", "microprice"
    ]

    def process_leg(df, name):

        # Resample market data
        resampled = (
            df[columns]
            .resample(f"{resample_ms}ms")
            .last()
        )

        # Get timestamp of last actual quote update
        timestamps = pd.Series(df.index, index=df.index)

        last_update = (
            timestamps
            .resample(f"{resample_ms}ms")
            .last()
            .ffill()
        )

        # Calculate quote age using resample-bin end times.
        # This avoids negative ages when an update occurs
        # inside a resampling interval.
        bin_end = (
            resampled.index
            + pd.Timedelta(milliseconds=resample_ms)
        )

        resampled["quote_age_ms"] = (
            (bin_end - last_update)
            .dt.total_seconds()
            * 1000
        )

        # Forward fill price and order book features
        resampled[columns] = resampled[columns].ffill()

        # Filter stale quotes if threshold provided
        if max_quote_age_ms is not None:
            stale = resampled["quote_age_ms"] > max_quote_age_ms
            resampled.loc[stale, columns] = float("nan")

        # Rename columns according to exchange name
        resampled = resampled.rename(
            columns={
                col: f"{name}_{col}"
                for col in resampled.columns
            }
        )

        return resampled

    df1_resample = process_leg(df_1, df_1_name)
    df2_resample = process_leg(df_2, df_2_name)

    # Merge on timestamp indexes
    df = df1_resample.join(df2_resample, how="inner")

    return df