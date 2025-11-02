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

input, hidden, output = 2, 2, 1

#--------------------------------------------------------------------------------------------------
# TRAIN

def train_once_return_success(lr, epochs, rng_seed=None):
    rng_local = np.random.default_rng(rng_seed)
    idx = np.arange(len(X))

    W1 = rng_local.normal(0, np.sqrt(1/(input+1)), size=(input+1, hidden))
    W2 = rng_local.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))
    W1[-1, :] = np.array([+0.5, -0.5])

    max_attempts = 50
    min_alignment = -0.1
    B_local = None
    for _ in range(max_attempts):
        B_try = rng_local.normal(0, 1.0, size=(output, hidden))
        B_try, _ = np.linalg.qr(B_try.T)
        B_try = B_try.T
        x_s = add_bias(X[0:1, :])
        y_t = Y[0:1, :]
        h_s = relu(x_s @ W1)
        y_p = sigmoid(add_bias(h_s) @ W2)
        e_s = y_p - y_t
        W2_hidden = W2[:-1, :]
        alignment = e_s.item() * (W2_hidden.T @ B_try.T).item() * e_s.item()
        if alignment > min_alignment:
            B_local = B_try
            break
    if B_local is None:
        B_local = B_try

    convergence_epoch = None
    for ep in range(epochs):
        rng_local.shuffle(idx)
        Xb = add_bias(X[idx])
        Yb = Y[idx]
        h  = relu(Xb @ W1)
        hb = add_bias(h)
        y  = sigmoid(hb @ W2)
        e  = (y - Yb)
        d_hid = (e @ B_local) * d_relu(h)
        W2 -= lr * (hb.T @ e) / len(X)
        W1 -= lr * (Xb.T @ d_hid) / len(X)
        if ep % 10 == 0:
            y_val = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
            if np.array_equal((y_val >= 0.5).astype(int).ravel(), Y.ravel()):
                convergence_epoch = ep
                break

    y_final = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
    pred = (y_final >= 0.5).astype(int)
    success = np.array_equal(pred.ravel(), Y.ravel())
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
            print(f"[lr={lr:>6}, epochs={ep:>3}]  success: {succ:>3}/{n_runs}  "
                  f"avg_conv: {'∞' if np.isinf(avg_conv) else f'{avg_conv:.1f}'}")

    def _rank_key(item):
        (lr, ep), stats = item
        return (-stats["successes"], stats["avg_convergence_epoch"], ep, -lr)

    best_item = sorted(results.items(), key=_rank_key)[0]
    (best_lr, best_ep), best_stats = best_item

    print("\nTop 5 configurations:")
    for i, ((lr, ep), stats) in enumerate(sorted(results.items(), key=_rank_key)[:5], 1):
        avgc = stats["avg_convergence_epoch"]
        print(f"{i}. lr={lr}, epochs={ep}  ->  success={stats['successes']}/{n_runs}, "
              f"avg_conv: {'∞' if np.isinf(avgc) else f'{avgc:.1f}'}")

    avgc = best_stats["avg_convergence_epoch"]
    print(f"\nBEST CONFIG -> lr={best_lr}, epochs={best_ep} "
          f"(success={best_stats['successes']}/{n_runs}, avg_conv={'∞' if np.isinf(avgc) else f'{avgc:.1f}'})")

    return results, (best_lr, best_ep), best_stats

#--------------------------------------------------------------------------------------------------
# RUN

lrs = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
epochs_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

results, (best_lr, best_epochs), best_stats = run_grid_search(lrs, epochs_list, n_runs=100)


# -------------------------------------------------------------------------------------------------
# TESTING SEED

extra_seeds = [3, 9, 21]
print(f"\nTesting BEST config (lr={best_lr}, epochs={best_epochs}) on seeds {extra_seeds}:")
for s in extra_seeds:
    ok, conv_ep = train_once_return_success(best_lr, best_epochs, rng_seed=s)
    print(f"  Seed={s:>2} -> success={ok}, convergence_epoch={'∞' if conv_ep is None else conv_ep}")
