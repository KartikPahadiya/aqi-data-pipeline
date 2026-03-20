# # train_model.py
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import haversine_distances

# MODEL_PATH = "pollution_stgcn_model.pth"

# history = pd.read_csv("aqi_history.csv")
# history["collection_time"] = pd.to_datetime(history["collection_time"])

# pivot = history.pivot_table(
#     index=["collection_time","station"],
#     columns="pollutant_id",
#     values="avg_value"
# ).reset_index().fillna(0)

# stations = pivot["station"].unique()
# times = pivot["collection_time"].unique()
# pollutants = pivot.columns[2:]

# T, N, F = len(times), len(stations), len(pollutants)

# if T < 4:
#     print("Not enough data")
#     exit()

# data = np.zeros((T,N,F))

# for i,t in enumerate(times):
#     temp = pivot[pivot["collection_time"]==t]
#     for j,s in enumerate(stations):
#         row = temp[temp["station"]==s]
#         if not row.empty:
#             data[i,j,:] = row[pollutants].values

# scaler = StandardScaler()
# data = scaler.fit_transform(data.reshape(-1,F)).reshape(T,N,F)

# window = 2
# X, y = [], []

# for i in range(T-window):
#     X.append(data[i:i+window])
#     y.append(data[i+window])

# X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
# y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

# coords = history.drop_duplicates("station").set_index("station").loc[stations][["latitude","longitude"]].values
# coords_rad = np.radians(coords)
# dist = haversine_distances(coords_rad)*6371
# A = np.exp(-(dist**2)/(dist.std()**2))
# A_tensor = torch.tensor(A, dtype=torch.float32)

# class SimpleSTGCN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.temporal = nn.Conv2d(F, 32, (window,1))
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(32, F)

#     def forward(self, x, A):
#         x = x.permute(0,3,1,2)
#         x = self.relu(self.temporal(x))
#         x = x.squeeze(2).permute(0,2,1)
#         x = torch.matmul(A, x)
#         return self.fc(x)

# model = SimpleSTGCN()

# if os.path.exists(MODEL_PATH):
#     model.load_state_dict(torch.load(MODEL_PATH))
#     print("Continuing training")

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()

# for epoch in range(20):
#     pred = model(X_tensor, A_tensor)
#     loss = loss_fn(pred, y_tensor)

#     optimizer.zero_grad()
#     loss.backward()


import joblib

joblib.dump(scaler, "scaler.pkl")
#     optimizer.step()

# torch.save(model.state_dict(), MODEL_PATH)
# print("Model updated")
