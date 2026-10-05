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