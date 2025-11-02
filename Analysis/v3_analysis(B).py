#--------------------------------------------------------------------------------------------------
# IMPORTS

import numpy as np

#--------------------------------------------------------------------------------------------------
# FUNCTION DEFENITIONS

def relu(z): 
    return np.maximum(0, z)

def d_relu(z): 
    return (z > 0).astype(float)

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

#--------------------------------------------------------------------------------------------------
# DATA 

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

input_dim, hidden_dim, output_dim = 2, 2, 1
lr, epochs = 0.3, 500

#--------------------------------------------------------------------------------------------------
# B-INIT

def _make_B(method, output, hidden, seed=21):
    rng_local = np.random.default_rng(seed)
    if method == "normalize":
        B = rng_local.normal(0, 1.0, size=(output, hidden))
        B /= (np.linalg.norm(B) + 1e-12)
        note = "L2-normalized"
    elif method == "varmatch":
        B = rng_local.normal(0, np.sqrt(1/hidden), size=(output, hidden))
        note = "Var-match: N(0, 1/√hidden)"
    elif method == "orthogonal":
        B = rng_local.normal(0, 1.0, size=(output, hidden))
        Q, _ = np.linalg.qr(B.T)
        B = Q.T
        note = "QR-orthogonal row(s)"
    elif method == "posdef":
        if output == hidden:
            B = rng_local.normal(0, 1.0, size=(output, hidden))
            B = (B + B.T) / 2.0
            B = B + 0.5 * np.eye(B.shape[0])
            note = "Symmetrized + 0.5I (PD)"
        else:
            B = np.abs(rng_local.normal(0, 1.0, size=(output, hidden))) + 0.1
            note = "Positivity proxy (non-square -> PD not defined)"
    else:
        raise ValueError(f"Unknown B init: {method}")
    return B, note

#--------------------------------------------------------------------------------------------------
# TRAIN

def _train_once_with_B(B_init_name, seed=21):
    rng_local = np.random.default_rng(seed)

    W1 = rng_local.normal(0, np.sqrt(1/(input_dim+1)), size=(input_dim+1, hidden_dim))
    W2 = rng_local.normal(0, np.sqrt(1/(hidden_dim+1)), size=(hidden_dim+1, output_dim))
    W1[-1, :] = np.array([+0.5, -0.5], dtype=W1.dtype)

    B, note = _make_B(B_init_name, output_dim, hidden_dim, seed=seed)

    idx = np.arange(len(X))
    convergence_epoch = None

    for ep in range(epochs):
        rng_local.shuffle(idx)
        Xb = add_bias(X[idx])
        Yb = Y[idx]

        h  = relu(Xb @ W1)
        hb = add_bias(h)
        y  = sigmoid(hb @ W2)

        e = (y - Yb)
        d_hid = (e @ B) * d_relu(h)

        W2 -= lr * (hb.T @ e) / len(X)
        W1 -= lr * (Xb.T @ d_hid) / len(X)

        if ep % 10 == 0:
            y_val = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
            if np.array_equal((y_val >= 0.5).astype(int).ravel(), Y.ravel()) and convergence_epoch is None:
                convergence_epoch = ep

    y_fin = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
    loss  = binary_cross_entropy(y_fin, Y)
    acc   = float(np.mean((y_fin >= 0.5).astype(int) == Y))

    return {
        "method": B_init_name,
        "note": note,
        "loss": loss,
        "acc": acc,
        "conv_epoch": convergence_epoch,
        "outputs": y_fin.ravel().round(3),
        "pred": (y_fin >= 0.5).astype(int).ravel()
    }

#--------------------------------------------------------------------------------------------------
# SWEEP

n_runs = 100
methods = ["normalize", "varmatch", "orthogonal", "posdef"]

rng_master = np.random.default_rng()

summary = {}

for m in methods:
    successes, fails = 0, 0
    conv_epochs, losses = [], []
    for i in range(n_runs):
        seed_i = int(rng_master.integers(0, 2**63-1))
        res = _train_once_with_B(m, seed=21)
        losses.append(res["loss"])
        if res["acc"] == 1.0:
            successes += 1
            conv_epochs.append(res["conv_epoch"])
        else:
            fails += 1
    summary[m] = {
        "successes": successes,
        "fails": fails,
        "avg_loss": float(np.mean(losses)),
        "avg_conv_epoch": (np.mean(conv_epochs) if conv_epochs else None)
    }

for m in methods:
    s = summary[m]
    print(f"\n=== B init: {m} ===")
    print(f"Successes: {s['successes']} / {n_runs}")
    print(f"Avg Loss: {s['avg_loss']:.4f}")
    print(f"Avg Convergence Epoch: {s['avg_conv_epoch']}")
