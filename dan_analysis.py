import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import db_conn

# %%
conn = db_conn.fn()

df = pd.read_sql(
    r"SELECT * FROM osu_dataset " r"WHERE mapname LIKE %s " r"OR mapname LIKE %s",
    conn,
    params=(
        "%Regular Dan Phase%",
        "%LN Dan Phase%",
    ),
)
conn.dispose()
# %%
df["dan_level"] = (
    df["mapname"]
    .str.extract(r"\[(.*?)\sDan")
    .replace(
        {
            "0th": 0,
            "1st": 1,
            "2nd": 2,
            "3rd": 3,
            "4th": 4,
            "5th": 5,
            "6th": 6,
            "7th": 7,
            "8th": 8,
            "9th": 9,
            "10th": 10,
            "Gamma": 11,
            "Azimuth": 12,
            "Zenith": 13,
        }
    )
)
df["dan_type"] = df["mapname"].str.extract(r"-\s(.*?)\s").replace({"Regular": "RC"})
df["dan"] = df["dan_type"] + df["dan_level"].astype(str)
# %%
uid = df["username"] + "/" + df["year"].astype(str)
mid = df["dan"] + "/" + df["speed"].astype(str)
acc = df["accuracy"]
# %%
from sklearn.preprocessing import LabelEncoder

uid_le = LabelEncoder()
mid_le = LabelEncoder()

uid_enc = uid_le.fit_transform(uid)
mid_enc = mid_le.fit_transform(mid)

n_uid = len(uid_le.classes_)
n_mid = len(mid_le.classes_)
# %%
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.uid_emb = nn.Embedding(n_uid, 2)
        self.mid_emb = nn.Embedding(n_mid, 2)

    def forward(self, x_uid, x_mid):
        x_uid_emb = self.uid_emb(x_uid)
        x_mid_emb = self.mid_emb(x_mid)
        return nn.functional.sigmoid(x_uid_emb - x_mid_emb).mean(dim=-1)


ds = TensorDataset(
    torch.tensor(uid_enc), torch.tensor(mid_enc), torch.tensor(acc).to(torch.float)
)
n_train = int(len(ds) * 0.8)
train_ds, test_ds = random_split(ds, [n_train, len(ds) - n_train])
batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# %%
m = Model()
epochs = 100
lr = 0.002
optim = torch.optim.Adam(m.parameters(), lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    print(f"Epoch: {epoch}", end=" ")
    for x_uid, x_mid, y in train_dl:
        y_pred = m(x_uid, x_mid)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
    print(f"RMSE: {loss.item() ** 0.5:.2%}", end=" ")
    for x_uid, x_mid, y in test_dl:
        y_pred = m(x_uid, x_mid)
        loss = loss_fn(y_pred, y)
    print(f"Test RMSE: {loss.item() ** 0.5:.2%}")
# %%

w_uid_emb = m.uid_emb.weight.detach().numpy()
w_mid_emb = m.mid_emb.weight.detach().numpy()

# %%


df_mid = (
    pd.DataFrame([mid_le.classes_, *w_mid_emb.T])
    .T.rename(columns={0: "mid", 1: "emb1", 2: "emb2"})
    .sort_values("emb1", ascending=False)
).assign(mid=lambda x: x.mid.str[:-2])

df_uid = (
    pd.DataFrame([uid_le.classes_, *w_uid_emb.T])
    .T.rename(columns={0: "uid", 1: "emb1", 2: "emb2"})
    .sort_values("emb1", ascending=False)
)

# %%
import plotly.express as px

fig = px.scatter(
    df_mid,
    x="emb1",
    y="emb2",
    text="mid",
)
fig.show()
# %%

fig = px.scatter(
    df_uid.sample(1000),
    x="emb1",
    y="emb2",
    text="uid",
)
fig.show()
