import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import haversine_distances

st.title("🌍 AQI Dashboard + Prediction")

df = pd.read_csv("aqi_history.csv")
df["collection_time"] = pd.to_datetime(
    df["collection_time"],
    errors="coerce",
    format="mixed",
    dayfirst=True
)

df = df.dropna(subset=["collection_time"])

st.dataframe(df.tail(20))

pivot = df.pivot_table(
    index=["collection_time","station"],
    columns="pollutant_id",
    values="avg_value"
).reset_index().fillna(0)

stations = pivot["station"].unique()
times = pivot["collection_time"].unique()
pollutants = pivot.columns[2:]

T, N, F = len(times), len(stations), len(pollutants)

data = np.zeros((T,N,F))

for i,t in enumerate(times):
    temp = pivot[pivot["collection_time"]==t]
    for j,s in enumerate(stations):
        row = temp[temp["station"]==s]
        if not row.empty:
            data[i,j,:] = row[pollutants].values

scaler = joblib.load("scaler.pkl")
data_scaled = scaler.transform(data.reshape(-1,F)).reshape(T,N,F)

coords = df.drop_duplicates("station").set_index("station").loc[stations][["latitude","longitude"]].values

coords_rad = np.radians(coords)
dist = haversine_distances(coords_rad)*6371
A = np.exp(-(dist**2)/(dist.std()**2))

A_tensor = torch.tensor(A, dtype=torch.float32)

class SimpleSTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv2d(F, 32, (2,1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, F)

    def forward(self, x, A):
        x = x.permute(0,3,1,2)
        x = self.relu(self.temporal(x))
        x = x.squeeze(2).permute(0,2,1)
        x = torch.matmul(A, x)
        return self.fc(x)

model = SimpleSTGCN()
model.load_state_dict(torch.load("pollution_stgcn_model.pth", map_location="cpu"))
model.eval()

window = 2

if T > window:
    X_input = data_scaled[-window:]
    X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(X_input, A_tensor)

    pred = pred.numpy()[0]

    st.subheader("Prediction")

    plt.plot(pred.mean(axis=1))
    st.pyplot(plt)

plt.plot(df["avg_value"].values[-500:])
st.pyplot(plt)
