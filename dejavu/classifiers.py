"""The six classifiers trained on top of the frozen features (Section 3.2)."""

import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

CLASSIFIERS = ["1-MLP", "2-MLP", "3-MLP", "LR", "SVM", "kNN"]
MLP_CLASSIFIERS = ["1-MLP", "2-MLP", "3-MLP"]
C_GRID = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]


def build_mlp(name, input_size, num_classes):
    if name == "1-MLP":
        return nn.Sequential(nn.Linear(input_size, num_classes))
    if name == "2-MLP":
        return nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.5),
                             nn.Linear(256, num_classes))
    if name == "3-MLP":
        return nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.5),
                             nn.Linear(512, 256), nn.ReLU(),
                             nn.Linear(256, num_classes))
    raise ValueError(f"Unknown MLP '{name}'")


def class_weights(y, num_classes):
    """Inverse-frequency class weights for the cross-entropy loss, normalized to sum to 1."""
    counts = np.bincount(y, minlength=num_classes).astype(float)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    return torch.tensor(w / w.sum(), dtype=torch.float32)


class MLPClassifier:
    def __init__(self, name, input_size, num_classes, device="cuda"):
        self.name, self.input_size, self.num_classes, self.device = name, input_size, num_classes, device
        self.model = build_mlp(name, input_size, num_classes).to(device)

    def fit(self, X, y, weights, epochs=50, batch_size=64, lr=3e-4):
        # The features fit in GPU memory: batches are drawn from a random permutation of the
        # training set at every epoch (torch.randperm on the global generator, seeded by the
        # caller), as a shuffled DataLoader would do, without its per-batch CPU overhead.
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(y, dtype=torch.long).to(self.device)
        n = len(X)
        steps = (n + batch_size - 1) // batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        self.model.train()
        bar = tqdm(total=epochs * steps, unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5)
        for epoch in range(epochs):
            perm = torch.randperm(n).to(self.device)
            total = torch.zeros((), device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X[idx]), Y[idx])
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(idx)
                bar.update()
            bar.set_description(f"    epoch {epoch + 1}/{epochs}  loss {total.item() / n:.4f}")
        bar.close()
        return self

    @torch.no_grad()
    def predict_scores(self, X, batch_size=4096):
        """Softmax scores [N, C]; predictions are their argmax."""
        self.model.eval()
        X = torch.as_tensor(X)
        out = [torch.softmax(self.model(X[i:i + batch_size].to(self.device)), dim=1).cpu()
               for i in range(0, len(X), batch_size)]
        return torch.cat(out).numpy()

    def save(self, path):
        torch.save({"name": self.name, "input_size": self.input_size, "num_classes": self.num_classes,
                    "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device)
        clf = cls(ckpt["name"], ckpt["input_size"], ckpt["num_classes"], device)
        clf.model.load_state_dict(ckpt["state_dict"])
        return clf


def _svm_scores(decision):
    """LinearSVC margins -> scores: sigmoid (binary) / softmax (multi-class). Used for AP/AUC only."""
    decision = np.asarray(decision)
    if decision.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-decision))
        return np.stack([1.0 - p1, p1], axis=1)
    e = np.exp(decision - decision.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class SklearnClassifier:
    """LR / linear SVM / kNN with hyper-parameters tuned by 5-fold CV on the training set."""

    def __init__(self, name, n_jobs=-1):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import LinearSVC
        self.name = name
        if name == "LR":
            self.estimator, self.grid = LogisticRegression(max_iter=10000, n_jobs=n_jobs), {"C": C_GRID}
        elif name == "SVM":
            self.estimator, self.grid = LinearSVC(max_iter=10000, dual=False), {"C": C_GRID}
        elif name == "kNN":
            self.estimator = KNeighborsClassifier(metric="cosine", n_jobs=n_jobs)
            self.grid = {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]}
        else:
            raise ValueError(f"Unknown classifier '{name}'")
        self.n_jobs = n_jobs
        self.best_params = None

    def fit(self, X, y, **_):
        from sklearn.model_selection import GridSearchCV
        # verbose=1 prints how many fits the 5-fold grid search runs
        gs = GridSearchCV(self.estimator, param_grid=self.grid, cv=5, n_jobs=self.n_jobs, verbose=1)
        gs.fit(X, y)
        self.estimator, self.best_params = gs.best_estimator_, gs.best_params_
        return self

    def predict_scores(self, X):
        if self.name == "SVM":
            return _svm_scores(self.estimator.decision_function(X))
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)

    def save(self, path):
        joblib.dump({"name": self.name, "estimator": self.estimator, "best_params": self.best_params}, path)

    @classmethod
    def load(cls, path, **_):
        d = joblib.load(path)
        obj = cls.__new__(cls)
        obj.name, obj.estimator, obj.best_params = d["name"], d["estimator"], d["best_params"]
        return obj


def build_classifier(name, input_size, num_classes, device="cuda", n_jobs=-1):
    if name in MLP_CLASSIFIERS:
        return MLPClassifier(name, input_size, num_classes, device)
    return SklearnClassifier(name, n_jobs=n_jobs)


def classifier_file(run_dir, name, run):
    ext = "pth" if name in MLP_CLASSIFIERS else "pkl"
    return os.path.join(run_dir, "classifiers", f"run{run}.{ext}")


def load_classifier(path, device="cuda"):
    return (MLPClassifier.load(path, device) if path.endswith(".pth") else SklearnClassifier.load(path))


def predict(clf, X):
    """Return (predicted labels, scores). Labels are always the argmax of the class scores
    (for kNN/LR/SVM, the estimator's own prediction, which is the same)."""
    scores = clf.predict_scores(X)
    preds = clf.predict(X) if isinstance(clf, SklearnClassifier) else scores.argmax(1)
    return np.asarray(preds), scores


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
