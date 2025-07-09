import requests
import pandas as pd
import time

BIRDEYE_API_KEY = 'eef5ec9d85d246069d85b9f0b9c55972'
TOKEN_ADDRESS = '8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm'
BASE_URL = 'https://public-api.birdeye.so/public/ohlcv'
INTERVALS = ['15m', '4h', '1d']

headers = {
    'X-API-KEY': BIRDEYE_API_KEY
}

def fetch_ohlcv(token_address, interval):
    params = {
        'address': token_address,
        'interval': interval
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and data['data']:
            return pd.DataFrame(data['data'])
        else:
            print(f"No data found for interval {interval}")
            return pd.DataFrame()
    else:
        print(f"Error fetching {interval} data: {response.status_code}")
        return pd.DataFrame()

def main():
    dfs = {}
    for interval in INTERVALS:
        print(f"Fetching {interval} data...")
        df = fetch_ohlcv(TOKEN_ADDRESS, interval)
        if not df.empty:
            # Convert timestamp to readable date
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df.to_csv(f'ohlcv_{interval}.csv', index=False)
            dfs[interval] = df
            print(f"Saved ohlcv_{interval}.csv")
        else:
            print(f"No data for {interval}")
        time.sleep(1)  # To avoid rate limits
    return dfs

if __name__ == "__main__":
    all_dfs = main()
    # Example: print the first few rows of 15m data
    if '15m' in all_dfs:
        print(all_dfs['15m'].head())