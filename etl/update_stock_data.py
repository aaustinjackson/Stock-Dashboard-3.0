import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DATA_PATH = r"C:\Users\austi\PycharmProjects\StockBI\data\top10_stock_data_cleaned.csv"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOG", "NFLX", "AVGO", "JPM", "VRT"]
START_DATE = "2024-01-01"

# ---------------------------------------------
# Load existing data
# ---------------------------------------------
if os.path.exists(DATA_PATH):
    print("📂 Loading existing dataset...")
    df_existing = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df_existing["Ticker"] = df_existing["Ticker"].str.upper()
else:
    print("⚠️ No existing dataset found — starting fresh.")
    df_existing = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

# ---------------------------------------------
# Determine global start date (earliest missing)
# ---------------------------------------------
if not df_existing.empty:
    last_date = df_existing["Date"].max()
    start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
else:
    start = START_DATE

end = datetime.now().strftime("%Y-%m-%d")

if start >= end:
    print("✅ All tickers already up to date — no changes made.")
    exit()

print(f"\n📅 Downloading data from {start} to {end}...")


# ---------------------------------------------
# Download with retry (batched)
# ---------------------------------------------
def download_with_retry(tickers, start, end, retries=10):
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")

        wait = 16 * (attempt + 1)
        print(f"⏳ Retrying in {wait} seconds...")
        time.sleep(wait)

    return pd.DataFrame()


raw_data = download_with_retry(TICKERS, start, end)

if raw_data.empty:
    print("⚠️ No new data retrieved (likely rate limited). Try again later.")
    exit()

# ---------------------------------------------
# Transform multi-index dataframe → flat format
# ---------------------------------------------
print("🧹 Transforming data...")

all_data = []

# Case 1: Multiple tickers (MultiIndex columns)
if isinstance(raw_data.columns, pd.MultiIndex):
    for ticker in TICKERS:
        if ticker not in raw_data.columns.levels[0]:
            continue

        df_ticker = raw_data[ticker].copy()
        df_ticker.reset_index(inplace=True)
        df_ticker["Ticker"] = ticker
        all_data.append(df_ticker)

# Case 2: Single ticker fallback
else:
    raw_data.reset_index(inplace=True)
    raw_data["Ticker"] = TICKERS[0]
    all_data.append(raw_data)

# Combine all
new_df = pd.concat(all_data, ignore_index=True)

# Standardize column names
new_df = new_df.rename(columns=str.capitalize)

# Keep only needed columns
new_df = new_df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

# ---------------------------------------------
# Merge with existing data
# ---------------------------------------------
print("🔗 Merging with existing dataset...")

full_df = pd.concat([df_existing, new_df], ignore_index=True)

# Sort and deduplicate
full_df = (
    full_df
    .sort_values(["Ticker", "Date"])
    .drop_duplicates(subset=["Ticker", "Date"], keep="last")
)

# ---------------------------------------------
# Save
# ---------------------------------------------
full_df.to_csv(DATA_PATH, index=False)

print(f"\n💾 Updated data saved to:\n{DATA_PATH}")
print(f"✅ Final rows: {len(full_df):,}")
print(f"📊 Tickers: {full_df['Ticker'].nunique()}")