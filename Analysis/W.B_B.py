# ============================================================
# Fixed W1 & W2, vary only B
# Study success rate vs alignment (INIT or FINAL)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG

N_TRIALS = 1500
LEARNING_RATE = 1.0
EPOCHS = 700
NBINS = 25
ALIGN_AT = "init"        # "init" or "final"
MASTER_SEED = 123
CHECK_EVERY = 10


# ------------------------------------------------------------
# ACTIVATIONS 

def tanh(z, beta=0.5):
    return np.tanh(beta * z)

def d_tanh(a, beta=0.5):
    return beta * (1.0 - a**2)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------
# UTILS

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0], 1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true, eps=1e-8):
    return -np.mean(
        y_true * np.log(y_pred + eps) +
        (1 - y_true) * np.log(1 - y_pred + eps)
    )

def cosine_similarity_vec(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------
# XOR DATA 

X_raw = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
X = 2 * X_raw - 1          # map to {-1, +1}
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)
Xb = add_bias(X)

INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 2, 2, 1


# ------------------------------------------------------------
# FIXED W1 & W2 

master_rng = np.random.default_rng(MASTER_SEED)

W1_FIXED = master_rng.normal(
    0, np.sqrt(1/(INPUT_DIM+1)), size=(INPUT_DIM+1, HIDDEN_DIM)
)
W2_FIXED = master_rng.normal(
    0, np.sqrt(1/(HIDDEN_DIM+1)), size=(HIDDEN_DIM+1, OUTPUT_DIM)
)

# Bias init 
W1_FIXED[-1, :] = np.array([+0.1, -0.1])


# ------------------------------------------------------------
# SINGLE RUN (vary only B)

def run_one(trial_seed, lr=LEARNING_RATE, epochs=EPOCHS, align_at=ALIGN_AT):

    rng = np.random.default_rng(trial_seed)

    W1 = W1_FIXED.copy()
    W2 = W2_FIXED.copy()

    # Random orthogonal feedback matrix B
    B = rng.normal(0, np.sqrt(1/(HIDDEN_DIM+1)), size=(OUTPUT_DIM, HIDDEN_DIM))
    B, _ = np.linalg.qr(B.T)
    B = B.T

    # Conductance offset
    S = max(-W1.min(), -W2.min(), -B.min()) + 1e-3
    toG = lambda W: W + S
    toW = lambda G: G - S

    G1, G2, GB = toG(W1), toG(W2), toG(B)

    # alignment at INIT 
    if align_at == "init":
        W2_eff = toW(G2)
        W_last = W2_eff[:-1, :].T   
        B_eff  = toW(GB)
        alignment_value = cosine_similarity_vec(
            W_last.ravel(), B_eff.ravel()
        )

    converged = False

    # TRAINING LOOP 
    for ep in range(epochs):

        W1_eff = toW(G1)
        W2_eff = toW(G2)
        B_eff  = toW(GB)

        # Forward
        h = tanh(Xb @ W1_eff)
        hb = add_bias(h)
        y = sigmoid(hb @ W2_eff)

        # Error
        e = (y - Y)

        # Feedback Alignment update
        d_hid = (e @ B_eff) * d_tanh(h)

        W2_eff -= lr * (hb.T @ e) / len(X)
        W1_eff -= 0.25 * lr * (Xb.T @ d_hid) / len(X)

        G1, G2 = toG(W1_eff), toG(W2_eff)

        # Convergence check
        if ep % CHECK_EVERY == 0 or ep == epochs - 1:
            h_chk = tanh(add_bias(X) @ toW(G1))
            y_chk = sigmoid(add_bias(h_chk) @ toW(G2))
            pred = (y_chk >= 0.5).astype(int)
            if np.array_equal(pred.ravel(), Y.ravel()):
                converged = True
                break

    # alignment at FINAL 
    if align_at == "final":
        W2_final = toW(G2)
        W_last = W2_final[:-1, :].T
        B_final = toW(GB)
        alignment_value = cosine_similarity_vec(
            W_last.ravel(), B_final.ravel()
        )

    return alignment_value, converged


# ------------------------------------------------------------
# SWEEP MANY TRIALS

def main():

    aligns = np.empty(N_TRIALS, dtype=float)
    succ   = np.empty(N_TRIALS, dtype=int)

    print(f"Running {N_TRIALS} trials | lr={LEARNING_RATE}, epochs={EPOCHS}, align_at='{ALIGN_AT}'")

    for t in range(N_TRIALS):
        a, ok = run_one(MASTER_SEED + 1 + t)
        aligns[t] = a
        succ[t] = 1 if ok else 0
        if (t+1) % 200 == 0:
            print(f"  ... {t+1}/{N_TRIALS}")

    # Bin alignment → success rate
    bins = np.linspace(-1.0, 1.0, NBINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bidx = np.clip(np.digitize(aligns, bins) - 1, 0, NBINS-1)

    rate = np.full(NBINS, np.nan)
    count = np.zeros(NBINS, dtype=int)

    for i in range(NBINS):
        mask = (bidx == i)
        count[i] = mask.sum()
        if count[i] > 0:
            rate[i] = succ[mask].mean()

    # Plot
    plt.figure(figsize=(8,5.5))
    plt.scatter(aligns, succ, s=10, alpha=0.15, label="runs")
    plt.plot(centers, rate, marker='o', lw=2, label="binned success rate")

    plt.title(f"Success vs alignment (vary B only, {ALIGN_AT})")
    plt.xlabel("Alignment (W, B)")
    plt.ylabel("Success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(0.9, 0.1), loc="lower right")
    plt.tight_layout()
    plt.show()

    print(f"\nTotal trials: {N_TRIALS}")
    print(f"Overall success rate: {succ.mean()*100:.1f}%")

    good = np.where(rate >= 0.5)[0]
    if len(good) > 0:
        print(f"~50% success threshold alignment ≈ {centers[good[0]]:.3f} (count={count[good[0]]})")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
