import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn, sigmoid
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
# df = pd.read_sql(
#     r"SELECT * FROM osu_dataset WHERE `keys` = 7",
#     conn,
# )
# %%
# Only keep users and maps who have more than 100 records
df = df.groupby("username").filter(lambda x: len(x) > 3)
df = df.groupby("mapname").filter(lambda x: len(x) > 3)
df = df.reset_index(drop=True)
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
# mid = df["mapname"] + "/" + df["speed"].astype(str)
acc = df["accuracy"]
ln_ratio = df["ln_ratio"]
# %%

uid_le = LabelEncoder()
mid_le = LabelEncoder()
acc_qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform")

uid_enc = uid_le.fit_transform(uid)
mid_enc = mid_le.fit_transform(mid)
acc_tf = acc_qt.fit_transform(acc.to_numpy().reshape(-1, 1)).reshape(-1)

n_uid = len(uid_le.classes_)
n_mid = len(mid_le.classes_)

# %%

import torch.nn.functional as F


class ExpLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        return F.linear(x, torch.exp(self.weight), self.bias)


class Model(nn.Module):
    def __init__(
        self,
        rc_emb: int = 1,
        ln_emb: int = 1,
        rc_delta_emb: int = 3,
        ln_delta_emb: int = 3,
    ):
        super().__init__()
        self.uid_emb_rc = nn.Embedding(n_uid, rc_emb)
        self.mid_emb_rc = nn.Embedding(n_mid, rc_emb)
        self.uid_emb_ln = nn.Embedding(n_uid, ln_emb)
        self.mid_emb_ln = nn.Embedding(n_mid, ln_emb)
        self.delta_rc_to_acc = nn.Sequential(
            ExpLinear(rc_emb, rc_delta_emb),
            nn.ReLU(),
            ExpLinear(rc_delta_emb, 1),
        )
        self.delta_ln_to_acc = nn.Sequential(
            ExpLinear(ln_emb, ln_delta_emb),
            nn.ReLU(),
            ExpLinear(ln_delta_emb, 1),
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_uid, x_mid, ln_ratio):
        x_uid_emb_rc = self.uid_emb_rc(x_uid)
        x_mid_emb_rc = self.mid_emb_rc(x_mid)
        x_uid_emb_ln = self.uid_emb_ln(x_uid)
        x_mid_emb_ln = self.mid_emb_ln(x_mid)
        x_rc_delta = (x_uid_emb_rc - x_mid_emb_rc) / 2
        x_ln_delta = (x_uid_emb_ln - x_mid_emb_ln) / 2
        x_rc_acc = sigmoid(self.delta_rc_to_acc(x_rc_delta))
        x_ln_acc = sigmoid(self.delta_ln_to_acc(x_ln_delta))
        y = x_rc_acc.squeeze() * (1 - ln_ratio) + x_ln_acc.squeeze() * ln_ratio
        return y


# %%
ds = TensorDataset(
    torch.tensor(uid_enc),
    torch.tensor(mid_enc),
    torch.tensor(ln_ratio).to(torch.float),
    torch.tensor(acc_tf).to(torch.float),
)
# %%
n_train = int(len(ds) * 0.8)
train_ds, test_ds = random_split(ds, [n_train, len(ds) - n_train])
batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# %%
m = Model()
# %%
epochs = 30
lr = 0.003
optim = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.000001)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    print(f"Epoch: {epoch}", end=" ")
    train_losses = []
    test_losses = []
    for x_uid, x_mid, x_ln_ratio, y in train_dl:
        y_pred = m(x_uid, x_mid, x_ln_ratio)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_losses.append(loss.item())
    print(f"RMSE: {torch.tensor(train_losses).mean().item() ** 0.5:.2%}", end=" ")
    with torch.no_grad():
        for x_uid, x_mid, x_ln_ratio, y in test_dl:
            y_pred = m(x_uid, x_mid, x_ln_ratio)
            loss = loss_fn(
                torch.tensor(acc_qt.inverse_transform(y_pred.cpu().reshape(-1, 1))),
                torch.tensor(acc_qt.inverse_transform(y.cpu().reshape(-1, 1))),
            )
            test_losses.append(loss.item())
    print(f"Test RMSE: {torch.tensor(test_losses).mean().item() ** 0.5:.2%}")
# %%
w_uid_emb_rc = m.uid_emb_rc.weight.detach().numpy().squeeze()
w_uid_emb_ln = m.uid_emb_ln.weight.detach().numpy().squeeze()
w_mid_emb_rc = m.mid_emb_rc.weight.detach().numpy().squeeze()
w_mid_emb_ln = m.mid_emb_ln.weight.detach().numpy().squeeze()
# %%
df_mid = (
    pd.DataFrame([mid_le.classes_, w_mid_emb_rc, w_mid_emb_ln])
    .T.rename(columns={0: "mid", 1: "emb1", 2: "emb2"})
    .sort_values("emb1", ascending=False)
).assign(mid=lambda x: x.mid.str[:-2])

df_uid = (
    pd.DataFrame([uid_le.classes_, w_uid_emb_rc, w_uid_emb_ln])
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
    df_uid,
    x="emb1",
    y="emb2",
    text="uid",
)
fig.show()
