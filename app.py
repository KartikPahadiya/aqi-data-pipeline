# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from sklearn.metrics.pairwise import haversine_distances

# st.set_page_config(layout="wide")
# st.title("🌍 AQI Live Monitoring Dashboard")

# # ================================
# # Load data
# # ================================
# @st.cache_data
# def load_data():
#     df = pd.read_csv("aqi_history.csv")
#     df["collection_time"] = pd.to_datetime(df["collection_time"])
#     return df

# history = load_data()

# st.write("Dataset shape:", history.shape)

# # ================================
# # Pivot
# # ================================
# pivot = history.pivot_table(
#     index=["collection_time","station"],
#     columns="pollutant_id",
#     values="avg_value"
# ).reset_index().fillna(0)

# # ================================
# # Coordinates
# # ================================
# stations = pivot["station"].unique()

# coords = (
#     history.drop_duplicates("station")
#     .set_index("station")
#     .loc[stations][["latitude","longitude"]]
#     .values
# )

# # ================================
# # Adjacency Matrix
# # ================================
# coords_rad = np.radians(coords)
# dist_matrix = haversine_distances(coords_rad)*6371
# sigma = dist_matrix.std()
# A = np.exp(-(dist_matrix**2)/(sigma**2))

# # ================================
# # Sidebar
# # ================================
# option = st.sidebar.selectbox(
#     "Choose Visualization",
#     ["Stations Map", "Pollution Map", "Distribution", "Graph Network"]
# )

# # ================================
# # 1️⃣ Stations Map
# # ================================
# if option == "Stations Map":
#     st.subheader("📍 Monitoring Stations")

#     plt.figure(figsize=(8,6))
#     plt.scatter(coords[:,1], coords[:,0], s=10)
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.title("Stations in India")

#     st.pyplot(plt)

# # ================================
# # 2️⃣ Pollution Map
# # ================================
# elif option == "Pollution Map":
#     st.subheader("🌫 Pollution Map")

#     latest = history.sort_values("collection_time").groupby("station").tail(1)

#     plt.figure(figsize=(8,6))
#     plt.scatter(
#         latest["longitude"],
#         latest["latitude"],
#         c=latest["avg_value"],
#         cmap="Reds",
#         s=30
#     )
#     plt.colorbar(label="Pollution")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")

#     st.pyplot(plt)

# # ================================
# # 3️⃣ Distribution
# # ================================
# elif option == "Distribution":
#     st.subheader("📊 PM2.5 Distribution")

#     pm25 = pivot.get("PM2.5")

#     if pm25 is not None:
#         plt.figure(figsize=(8,5))
#         pm25.plot(kind="hist", bins=30)
#         st.pyplot(plt)
#     else:
#         st.warning("PM2.5 not found")

# # ================================
# # 4️⃣ Graph Network
# # ================================
# elif option == "Graph Network":
#     st.subheader("🔗 Spatial Graph")

#     G = nx.from_numpy_array(A)

#     pos = {i:(coords[i][1], coords[i][0]) for i in range(len(coords))}

#     plt.figure(figsize=(8,6))
#     nx.draw(G, pos, node_size=10)

#     st.pyplot(plt)
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import haversine_distances

st.title("🌍 AQI Dashboard + Prediction")

# ================================
# Load data
# ================================
df = pd.read_csv("aqi_history.csv")
df["collection_time"] = pd.to_datetime(df["collection_time"])

st.subheader("📊 Latest Data")
st.dataframe(df.tail(20))

# ================================
# Pivot
# ================================
pivot = df.pivot_table(
    index=["collection_time","station"],
    columns="pollutant_id",
    values="avg_value"
).reset_index().fillna(0)

stations = pivot["station"].unique()
times = pivot["collection_time"].unique()
pollutants = pivot.columns[2:]

T, N, F = len(times), len(stations), len(pollutants)

# ================================
# Build tensor
# ================================
data = np.zeros((T,N,F))

for i,t in enumerate(times):
    temp = pivot[pivot["collection_time"]==t]
    for j,s in enumerate(stations):
        row = temp[temp["station"]==s]
        if not row.empty:
            data[i,j,:] = row[pollutants].values

# ================================
# Load scaler
# ================================
scaler = joblib.load("scaler.pkl")

data_scaled = scaler.transform(data.reshape(-1,F)).reshape(T,N,F)

# ================================
# Adjacency matrix
# ================================
coords = df.drop_duplicates("station").set_index("station").loc[stations][["latitude","longitude"]].values

coords_rad = np.radians(coords)
dist = haversine_distances(coords_rad)*6371
A = np.exp(-(dist**2)/(dist.std()**2))

A_tensor = torch.tensor(A, dtype=torch.float32)

# ================================
# Model
# ================================
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

# ================================
# Prediction
# ================================
window = 2

if T > window:
    X_input = data_scaled[-window:]
    X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(X_input, A_tensor)

    pred = pred.numpy()[0]

    st.subheader("🤖 Prediction (Next timestep)")

    plt.figure(figsize=(10,5))
    plt.plot(pred.mean(axis=1))
    plt.title("Predicted Pollution Trend")
    st.pyplot(plt)

# ================================
# Visualization
# ================================
st.subheader("📈 Pollution Trend")

plt.figure(figsize=(10,5))
plt.plot(df["avg_value"].values[-500:])
plt.title("Recent Pollution Values")
st.pyplot(plt)
