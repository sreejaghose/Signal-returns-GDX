import pandas as pd

PRICE_COLUMNS = ["SPY", "IAU", "GLD", "TLT", "GDX", "GDXJ"]


def load_prices(path: str, sheet_name: str = "GDX", header_row: int = 6) -> pd.DataFrame:
    """Load daily close prices for the 6 ETFs, indexed by Date, sorted ascending."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw = raw[raw["Date"].notna()].copy()
    raw["Date"] = pd.to_datetime(raw["Date"])

    for col in PRICE_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values("Date").set_index("Date")
    return raw[PRICE_COLUMNS]
