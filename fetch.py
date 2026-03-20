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
            time.sleep(5)
            continue

        records = r.json().get("records", [])

        if not records:
            break

        all_data.extend(records)
        time.sleep(1)

    return pd.DataFrame(all_data)

# -------------------------------
new_df = fetch_all_data()
new_df["collection_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

file_path = "aqi_history.csv"

if os.path.exists(file_path):
    old_df = pd.read_csv(file_path)

    today = datetime.now().strftime("%Y-%m-%d")
    if today in old_df["collection_time"].values:
        print("Already updated today")
        exit()

    final_df = pd.concat([old_df, new_df])
else:
    final_df = new_df

final_df.to_csv(file_path, index=False)
print("Data updated:", final_df.shape)
