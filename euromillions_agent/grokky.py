import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def euler_phi_upto(m: int) -> np.ndarray:
    phi = np.arange(m + 1)
    for i in range(2, m + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi[1:]  # 1..m

def fit_g1_linear(phi_train, poi_train, phi_all):
    # R glm(poi~g) default gaussian => linear baseline is fine
    Xtr = np.c_[np.ones(len(phi_train)), phi_train]
    beta, *_ = np.linalg.lstsq(Xtr, poi_train, rcond=None)
    Xall = np.c_[np.ones(len(phi_all)), phi_all]
    return Xall @ beta

class SeqDS(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.y) - self.seq_len)
    def __getitem__(self, i):
        return self.X[i:i+self.seq_len], self.y[i+self.seq_len]

class TinyGrok(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d), nn.LeakyReLU(),
            nn.Linear(d, d), nn.LeakyReLU(),
            nn.Linear(d, 1)
        )
    def forward(self, x):  # x: (B, seq, 2)
        h = self.net(x)     # (B, seq, 1)
        return h[:, -1, 0]  # use last step

def walk_forward_grok(poi, train_len=200, seq_len=10, epochs=30, lr=1e-3):
    T = len(poi)
    phi = euler_phi_upto(T).astype(np.float64)

    preds = []
    y_true = []
    p_scores = []  # for AUC on "high poi"
    y_bin = []

    for t in range(train_len, T):
        # window indices
        w0, w1 = t - train_len, t
        poi_tr = poi[w0:w1]
        phi_tr = phi[w0:w1]
        phi_all = phi[w0:w1+1]  # include current t for baseline

        # baseline g1 fitted ONLY on window
        g1_all = fit_g1_linear(phi_tr, poi_tr, phi_all)
        g1_tr = g1_all[:-1]
        g1_t = g1_all[-1]

        resid_tr = poi_tr - g1_tr
        resid_prev = np.r_[0.0, resid_tr[:-1]]

        # train-only normalization
        phi_mu, phi_sd = phi_tr.mean(), phi_tr.std() + 1e-12
        rmu, rsd = resid_prev.mean(), resid_prev.std() + 1e-12

        X_tr = np.stack([(phi_tr - phi_mu)/phi_sd, (resid_prev - rmu)/rsd], axis=1)
        y_tr = resid_tr

        # tiny model
        ds = SeqDS(X_tr, y_tr, seq_len=seq_len)
        if len(ds) < 5:
            resid_hat_t = float(resid_tr[-1])  # fallback persistence
        else:
            dl = DataLoader(ds, batch_size=32, shuffle=True)
            model = TinyGrok(d=16)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(epochs):
                model.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    opt.step()

            # predict for time t using last seq_len points ending at t-1
            # build X for t from last known phi and resid_prev
            phi_t = phi[t]
            resid_prev_t = resid_tr[-1]  # last residual becomes resid_prev at t
            x_t = np.array([(phi_t - phi_mu)/phi_sd, (resid_prev_t - rmu)/rsd], dtype=np.float32)

            # we feed a seq by taking last seq_len rows from X_tr and appending x_t
            seq = np.vstack([X_tr[-(seq_len-1):], x_t[None, :]])
            with torch.no_grad():
                resid_hat_t = float(model(torch.tensor(seq[None, :, :]))[0].item())

        poi_hat_t = float(g1_t + resid_hat_t)

        # store
        preds.append(poi_hat_t)
        y_true.append(float(poi[t]))

        # AUC task: classify whether poi[t] is "high" relative to TRAIN WINDOW
        thr = np.quantile(poi_tr, 0.80)
        yb = 1 if poi[t] >= thr else 0
        # use poi_hat_t as score (monotonic); you can also use resid_hat_t
        p_scores.append(poi_hat_t)
        y_bin.append(yb)

    # metrics
    auc = roc_auc_score(y_bin, p_scores) if len(set(y_bin)) == 2 else float("nan")
    mae = float(np.mean(np.abs(np.array(preds) - np.array(y_true))))
    return {"auc": auc, "mae": mae, "preds": preds, "y_true": y_true, "scores": p_scores, "y_bin": y_bin}
# Example usage:
# poi = np.random.rand(500)  # replace with actual data         