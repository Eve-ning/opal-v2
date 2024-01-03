import torch
from torch import nn

x_uid = torch.tensor([[0, 1, 1], [1, 2, 2]])
x_mid = torch.tensor([[1, 1, 0], [1, 1, 2]])
x_ratio = torch.tensor([[0.75, 0.75, 0.25], [0.75, 0.75, 0.5]])
# %%
n_rc_emb = 4
n_ln_emb = 4
uid_rc_emb = nn.Embedding(3, n_rc_emb)
mid_rc_emb = nn.Embedding(
    3,
    n_rc_emb,
)
uid_ln_emb = nn.Embedding(3, n_ln_emb)
mid_ln_emb = nn.Embedding(3, n_ln_emb)
# %%
x_uid_rc_emb = uid_rc_emb(x_uid)
x_mid_rc_emb = mid_rc_emb(x_mid)
x_uid_ln_emb = uid_ln_emb(x_uid)
x_mid_ln_emb = mid_ln_emb(x_mid)
x_delta_rc_emb = nn.functional.sigmoid(x_uid_rc_emb - x_mid_rc_emb)
x_delta_ln_emb = nn.functional.sigmoid(x_uid_ln_emb - x_mid_ln_emb)
# %%
w_delta_rc_emb = nn.Linear(n_rc_emb, 1, bias=False)
w_delta_rc_emb_prob = nn.functional.softmax(w_delta_rc_emb.weight, dim=1)
w_delta_ln_emb = nn.Linear(n_ln_emb, 1, bias=False)
w_delta_ln_emb_prob = nn.functional.softmax(w_delta_ln_emb.weight, dim=1)
# %%
x_rc = (x_delta_rc_emb @ w_delta_rc_emb_prob.T).mean(dim=-1)
x_rc_scaled = x_rc * x_ratio
x_ln = (x_delta_ln_emb @ w_delta_ln_emb_prob.T).mean(dim=-1)
x_ln_scaled = x_ln * (1 - x_ratio)
# %%
x_rc_scaled + x_ln_scaled
