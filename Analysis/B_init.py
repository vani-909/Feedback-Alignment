#--------------------------------------------------------------------------------------------------
# IMPORTS
import random
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------
# FUNCTIONS 

def tanh(z, beta=0.5):
    return np.tanh(beta * z)

def d_tanh(a, beta=0.5):
    return beta * (1.0 - a**2)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0], 1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(
        y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)
    )

#--------------------------------------------------------------------------------------------------
# DATA

X_raw = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
X = 2*X_raw - 1  # map to {-1,+1}
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

input_dim, hidden_dim, output_dim = 2, 2, 1

#--------------------------------------------------------------------------------------------------
# B INITIALIZATION 

def make_B(method, output_dim, hidden_dim, rng):
    if method == "random":
        B = rng.normal(0, 1.0, size=(output_dim, hidden_dim))
   
    elif method == "varmatch":
        B = rng.normal(0, np.sqrt(1/hidden_dim), size=(output_dim, hidden_dim))
      
    elif method == "orthonormal":
        B = rng.normal(0, np.sqrt(1/(hidden_dim+1)), size=(output_dim, hidden_dim))
        B, _ = np.linalg.qr(B.T)
        B = B.T

    elif method == "positive":
        B = np.abs(rng.normal(0, 1.0, size=(output_dim, hidden_dim))) + 0.1

    else:
        raise ValueError(f"Unknown B init: {method}")
    
    return B

#--------------------------------------------------------------------------------------------------
# ONE RUN 

def train_once_with_B(method, lr=1.0, epochs=1000, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))

    # Init weights 
    W1 = rng.normal(0, np.sqrt(1/(input_dim+1)), size=(input_dim+1, hidden_dim))     # (3,2)
    W2 = rng.normal(0, np.sqrt(1/(hidden_dim+1)), size=(hidden_dim+1, output_dim))  # (3,1)
    W1[-1, :] = np.array([+0.1, -0.1])

    # B init 
    B = make_B(method, output_dim, hidden_dim, rng)

    # Conductance mapping offset 
    S = max(-W1.min(), -W2.min(), -B.min()) + 1e-3
    def to_conductance(W): return W + S
    def to_weight(G):      return G - S

    G1 = to_conductance(W1)
    G2 = to_conductance(W2)
    GB = to_conductance(B)   # fixed feedback path

    convergence_epoch = None

    for ep in range(epochs):
        rng.shuffle(idx)
        Xb = add_bias(X[idx])
        Yb = Y[idx]

        # Read effective weights
        W1_eff = to_weight(G1)
        W2_eff = to_weight(G2)
        B_eff  = to_weight(GB)

        # Forward
        h  = tanh(Xb @ W1_eff)
        hb = add_bias(h)
        y  = sigmoid(hb @ W2_eff)

        # Error
        e = (y - Yb)

        # FA hidden update
        d_hid = (e @ B_eff) * d_tanh(h)

        # Updates 
        lr_W2 = lr
        lr_W1 = 0.25 * lr
        W2_eff -= lr_W2 * (hb.T @ e)     / len(X)
        W1_eff -= lr_W1 * (Xb.T @ d_hid) / len(X)

        # Write back
        G1 = to_conductance(W1_eff)
        G2 = to_conductance(W2_eff)

        # Convergence check every 10 epochs
        if ep % 10 == 0:
            h_val = tanh(add_bias(X) @ W1_eff)
            y_val = sigmoid(add_bias(h_val) @ W2_eff)
            pred = (y_val >= 0.5).astype(int)
            if np.array_equal(pred.ravel(), Y.ravel()):
                convergence_epoch = ep
                break

    # Final success
    W1_final = to_weight(G1)
    W2_final = to_weight(G2)
    y_final = sigmoid(add_bias(tanh(add_bias(X) @ W1_final)) @ W2_final)
    pred_final = (y_final >= 0.5).astype(int)
    success = np.array_equal(pred_final.ravel(), Y.ravel())

    return success, convergence_epoch

#--------------------------------------------------------------------------------------------------
# SWEEP

methods = ["random", "varmatch", "orthonormal", "positive"]

lr = 1.0
epochs = 1000
n_runs = 1500
base_seed = 12345

summary = {}
for m in methods:
    succ = 0
    conv_epochs = []
    for r in range(n_runs):
        seed = base_seed + random.randint(0, 1000000)
        ok, conv_ep = train_once_with_B(m, lr=lr, epochs=epochs, seed=seed)
        succ += int(ok)
        if conv_ep is not None:
            conv_epochs.append(conv_ep)
    avg_conv = float(np.mean(conv_epochs)) if len(conv_epochs) else np.inf
    summary[m] = {"successes": succ, "avg_conv": avg_conv}
    print(f"{m:>10}: success {succ}/{n_runs} | avg_conv: {'∞' if np.isinf(avg_conv) else f'{avg_conv:.1f}'}")

#--------------------------------------------------------------------------------------------------
# PLOT 

labels = methods
succ_rate = [100*summary[m]["successes"]/n_runs for m in methods]
avg_conv  = [summary[m]["avg_conv"] for m in methods]
x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(9, 4))
bars = ax1.bar(x, succ_rate, alpha=0.6)
ax1.set_ylabel("Success rate (%)")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_title("Effect of B initialization")
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

for b, v in zip(bars, succ_rate):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, avg_conv, "o-", color="tab:red")
ax2.set_ylabel("Avg convergence epoch", color="tab:red")

plt.tight_layout()
plt.show()
