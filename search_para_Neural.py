import numpy as np, pandas as pd, torch, warnings, time, optuna
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = "mlp" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0);
np.random.seed(0)

X_train = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv")['label'].values
X_test = pd.read_csv("X_test_1.csv").values

mu, std = X_train.mean(0), X_train.std(0, ddof=0)
X_train = np.clip(X_train, mu - 3 * std, mu + 3 * std)
X_test = np.clip(X_test, mu - 3 * std, mu + 3 * std)

scaler = RobustScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


class TabDS(Dataset):
    def __init__(self, X, y): self.X = torch.tensor(X); self.y = torch.tensor(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loader(X_sub, y_sub, batch=512):
    class_cnt = np.bincount(y_sub, minlength=28)
    weights = 1.0 / class_cnt[y_sub]  # len == len(y_sub)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    ds = TabDS(X_sub, y_sub)
    return DataLoader(ds, batch_size=batch, sampler=sampler,
                      num_workers=0, pin_memory=True)


class MLP(nn.Module):
    def __init__(self, in_dim=300, hidden=512, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 28)
        )

    def forward(self, x): return self.net(x)


class_cnt = np.bincount(y_train, minlength=28)
beta = 0.9999
alpha = (1 - beta) / (1 - beta ** class_cnt)
alpha = torch.tensor(alpha / alpha.mean(), dtype=torch.float32)


def cv_macro_f1(params, folds=3, epochs=5, batch=512):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    f1s = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        model = MLP(hidden=params['hidden'], drop=params['drop']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        #crit  = FocalLoss(alpha, gamma=params['gamma'])
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        crit = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loader_tr = make_loader(X_train[tr_idx], y_train[tr_idx], batch)
        loader_va = DataLoader(TabDS(X_train[va_idx], y_train[va_idx]),
                               batch_size=batch, shuffle=False)

        # ---- mini train ----
        model.train()
        for _ in range(epochs):
            for xb, yb in loader_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()

        # ---- val ----
        model.eval();
        preds = []
        with torch.no_grad():
            for xb, _ in loader_va:
                preds.append(model(xb.to(device)).argmax(1).cpu())
        f1s.append(f1_score(y_train[va_idx], torch.cat(preds).numpy(), average='macro'))
    return np.mean(f1s)


# =============================================================

# =============================================================
def objective_random(trial):
    params = {
        'lr': trial.suggest_loguniform('lr', 5e-4, 2e-3),
        'hidden': trial.suggest_categorical('hidden', [256]),
        'drop': trial.suggest_uniform('drop', 0.2, 0.5),
    }
    return cv_macro_f1(params)


study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective_random, n_trials=20, show_progress_bar=True)
print("Random best:", study.best_params, study.best_value)


# =============================================================

# =============================================================
def objective_tpe(trial):
    bp = study.best_params
    params = {
        'lr': trial.suggest_loguniform('lr', bp['lr'] / 3, bp['lr'] * 3),
        'hidden': trial.suggest_categorical('hidden',
                                            [max(128, bp['hidden'] // 2), bp['hidden']]),
        'drop': trial.suggest_uniform('drop',
                                      max(0.1, bp['drop'] - 0.1), min(0.6, bp['drop'] + 0.1))
    }
    return cv_macro_f1(params)


study_tpe = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
study_tpe.optimize(objective_tpe, n_trials=50, show_progress_bar=True)
best = study_tpe.best_params
best['lr'] = float(best['lr'])  #
print("TPE best:", best, study_tpe.best_value)

# =============================================================

# =============================================================
EPOCHS, BATCH = 30, 512
loader_full = make_loader(X_train, y_train, BATCH)

model = MLP(hidden=best['hidden'], drop=best['drop']).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=best['lr'], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
crit = nn.CrossEntropyLoss(weight=class_weights_tensor)

for ep in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in loader_full:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad();
        crit(model(xb), yb).backward();
        opt.step()
    sched.step()
print("Full training done.")
