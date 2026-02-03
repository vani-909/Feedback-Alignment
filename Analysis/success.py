#--------------------------------------------------------------------------------------------------
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------
# FUNCTION DEFENITIONS

def tanh(z, beta=0.5):
    return np.tanh(beta * z)

def d_tanh(a, beta=0.5):
    # a = tanh(beta z)
    return beta * (1.0 - a**2)

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

#--------------------------------------------------------------------------------------------------
# DATA 

X_raw = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
X     = 2*X_raw - 1   # now in {-1, +1}
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

input, hidden, output = 2, 2, 1

#--------------------------------------------------------------------------------------------------
# TRAIN

def train_once_return_success(lr, epochs, rng_seed=None):
    rng_local = np.random.default_rng(rng_seed)
    idx = np.arange(len(X))

    # Init
    W1 = rng_local.normal(0, np.sqrt(1/(input+1)), size=(input+1, hidden))
    W2 = rng_local.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))
    W1[-1, :] = np.array([0.1, -0.1]) 

    # Fixed random feedback matrix B 
    B = rng_local.normal(0, 1.0, size=(output, hidden))
    B, _ = np.linalg.qr(B.T)
    B = B.T

    # Conductance mapping offset
    S = max(-W1.min(), -W2.min(), -B.min()) + 1e-3

    def to_conductance(W): return W + S
    def to_weight(G):      return G - S

    # Map to conductances
    G1 = to_conductance(W1)
    G2 = to_conductance(W2)
    GB = to_conductance(B)

    convergence_epoch = None

    for ep in range(epochs):
        rng_local.shuffle(idx)
        Xb = add_bias(X[idx])
        Yb = Y[idx]

        # Read effective weights from "hardware"
        W1_eff = to_weight(G1)
        W2_eff = to_weight(G2)
        B_eff  = to_weight(GB)

        # Forward with tanh
        h  = tanh(Xb @ W1_eff)
        hb = add_bias(h)
        y  = sigmoid(hb @ W2_eff)
        e  = (y - Yb)

        # FA hidden update using tanh derivative
        d_hid = (e @ B_eff) * d_tanh(h)

        # Updates
        lr_W2 = lr
        lr_W1 = 0.25 * lr  

        W2_eff -= lr_W2 * (hb.T @ e)     / len(X)
        W1_eff -= lr_W1 * (Xb.T @ d_hid) / len(X)


        # Write back to conductances (hardware)
        G1 = to_conductance(W1_eff)
        G2 = to_conductance(W2_eff)

        W1 = W1_eff.copy()
        W2 = W2_eff.copy()

        # Convergence check every 10 epochs 
        if ep % 10 == 0:
            h_val = tanh(add_bias(X) @ W1)
            y_val = sigmoid(add_bias(h_val) @ W2)
            if np.array_equal((y_val >= 0.5).astype(int).ravel(), Y.ravel()):
                convergence_epoch = ep
                break

    # Final success on hardware-effective weights
    W1_final = to_weight(G1)
    W2_final = to_weight(G2)
    h_final  = tanh(add_bias(X) @ W1_final)
    y_final  = sigmoid(add_bias(h_final) @ W2_final)
    pred     = (y_final >= 0.5).astype(int)
    success  = np.array_equal(pred.ravel(), Y.ravel())

    return success, convergence_epoch


#--------------------------------------------------------------------------------------------------
# GRID SEARCH

def run_grid_search(lrs, epochs_list, n_runs=100, base_seed=12345):
    results = {}
    for lr in lrs:
        for ep in epochs_list:
            succ = 0
            conv_epochs = []
            for run in range(n_runs):
                seed = base_seed + (hash((lr, ep)) % 10_000_000) + run
                ok, conv_ep = train_once_return_success(lr, ep, rng_seed=seed)
                succ += int(ok)
                if conv_ep is not None:
                    conv_epochs.append(conv_ep)
            avg_conv = float(np.mean(conv_epochs)) if len(conv_epochs) else np.inf
            results[(lr, ep)] = {"successes": succ, "avg_convergence_epoch": avg_conv}
            print(f"[lr={lr:>6}, epochs={ep:>4}]  success: {succ:>3}/{n_runs}  "
                  f"avg_conv: {'∞' if np.isinf(avg_conv) else f'{avg_conv:.1f}'}")

    def _rank_key(item):
        (lr, ep), stats = item
        return (-stats["successes"], stats["avg_convergence_epoch"], ep, -lr)

    best_item = sorted(results.items(), key=_rank_key)[0]
    (best_lr, best_ep), best_stats = best_item

    print("\nTop 10 configurations:")
    for i, ((lr, ep), stats) in enumerate(sorted(results.items(), key=_rank_key)[:10], 1):
        avgc = stats["avg_convergence_epoch"]
        print(f"{i}. lr={lr}, epochs={ep}  ->  success={stats['successes']}/{n_runs}, "
              f"avg_conv: {'∞' if np.isinf(avgc) else f'{avgc:.1f}'}")

    avgc = best_stats["avg_convergence_epoch"]
    print(f"\nBEST CONFIG -> lr={best_lr}, epochs={best_ep} "
          f"(success={best_stats['successes']}/{n_runs}, avg_conv={'∞' if np.isinf(avgc) else f'{avgc:.1f}'})")

    return results, (best_lr, best_ep), best_stats


#--------------------------------------------------------------------------------------------------
# RUN

lrs = [0.5, 0.7, 1.0]  
epochs_list = [300, 500, 700, 1000, 1500, 2000, 2300]  
         

results, (best_lr, best_epochs), best_stats = run_grid_search(lrs, epochs_list, n_runs=100)

#--------------------------------------------------------------------------------------------------
# Plot

n_runs = 100 

top = sorted(results.items(), key=lambda kv: (-kv[1]["successes"], kv[1]["avg_convergence_epoch"]))[:10]

labels = [f"lr={lr}, ep={ep}" for (lr, ep), _ in top]
succ_rate = [100 * stats["successes"] / n_runs for _, stats in top]
avg_conv  = [stats["avg_convergence_epoch"] for _, stats in top]
x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(9,4))
bars = ax1.bar(x, succ_rate, alpha=0.6)

for b, v in zip(bars, succ_rate):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{v:.1f}%", 
             ha="center", va="bottom", fontsize=9)

ax1.set_ylabel("Success rate (%)")
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha="right")
ax1.set_title("Top 10 Configurations")
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(x, avg_conv, "o-", color="tab:red")
ax2.set_ylabel("Avg convergence epoch", color="tab:red")

plt.tight_layout()
plt.show()
