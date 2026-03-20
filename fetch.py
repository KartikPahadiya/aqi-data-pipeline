import requests
import pandas as pd
import time
from datetime import datetime
import os

API_KEY = "579b464db66ec23bdd000001defb204ced3c478d5e7628fbd2525939"

url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
headers = {"User-Agent": "Mozilla/5.0"}

def fetch_all_data():
    all_data = []

    for offset in range(0, 50000, 1000):
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": 1000,
            "offset": offset
        }

        r = requests.get(url, params=params, headers=headers)

        if r.status_code != 200:
            print("Server error:", r.status_code)
            time.sleep(5)
            continue

        records = r.json().get("records", [])

        if not records:
            break

        all_data.extend(records)
        time.sleep(1)

    return pd.DataFrame(all_data)

# -------------------------------
# Fetch new data
# -------------------------------
new_df = fetch_all_data()
new_df["collection_time"] = datetime.now()

file_path = "aqi_history.csv"

# -------------------------------
# Merge instead of blind append
# -------------------------------
if os.path.exists(file_path):
    old_df = pd.read_csv(file_path)
    final_df = pd.concat([old_df, new_df])
    final_df.drop_duplicates(inplace=True)
else:
    final_df = new_df

# -------------------------------
# Save updated dataset
# -------------------------------
final_df.to_csv(file_path, index=False)

print("Updated dataset:", final_df.shape)
