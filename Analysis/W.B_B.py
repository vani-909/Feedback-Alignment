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
EPOCHS = 1000
NBINS = 25
ALIGN_AT = "init"        # "init" or "final"
MASTER_SEED = 123
CHECK_EVERY = 10

G_off = 1.1669e-3
G_on  = 1.3327e-3

Wmax = 10
Bmax = 1

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

    # Conductance offset (Mean shift)
    G0 = 0.5 * (G_on + G_off)
    dG = (G_on - G_off)

    alpha_W = 2.0 * Wmax / dG
    alpha_B = 2.0 * Bmax / dG

    toG_W = lambda W: np.clip(G0 + (W / alpha_W), G_off, G_on)
    toW_W = lambda G: alpha_W * (G - G0)

    toG_B = lambda Bm: np.clip(G0 + (Bm / alpha_B), G_off, G_on)
    toW_B = lambda G: alpha_B * (G - G0)

    G1, G2, GB = toG_W(W1), toG_W(W2), toG_B(B)


    w_vec = None
    b_vec = None

    # alignment at INIT 
    if align_at == "init":
        W2_eff = toW_W(G2)
        W_last = W2_eff[:-1, :].T   
        B_eff  = toW_B(GB)

        w_vec = W_last.ravel().copy()
        b_vec = B_eff.ravel().copy()

        alignment_value = cosine_similarity_vec(
            W_last.ravel(), B_eff.ravel()
        )
    
    # init u and cancellation
    W1_eff0 = toW_W(G1)
    h0 = tanh(Xb @ W1_eff0)
    hb0 = add_bias(h0)
    y0 = sigmoid(hb0 @ toW_W(G2))
    e0 = (y0 - Y)

    u0 = (e0 @ B_eff) * d_tanh(h0)         
    U0 = u0.sum(axis=0)                       
    cancel = np.linalg.norm(U0) / (np.sum(np.linalg.norm(u0, axis=1)) + 1e-12)
    B0 = B_eff.ravel().copy()                 

    converged = False

    # TRAINING LOOP 
    for ep in range(epochs):

        W1_eff = toW_W(G1)
        W2_eff = toW_W(G2)
        B_eff  = toW_B(GB)

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

        G1, G2 = toG_W(W1_eff), toG_W(W2_eff)

        # Convergence check
        if ep % CHECK_EVERY == 0 or ep == epochs - 1:
            h_chk = tanh(add_bias(X) @ toW_W(G1))
            y_chk = sigmoid(add_bias(h_chk) @ toW_W(G2))
            pred = (y_chk >= 0.5).astype(int)
            if np.array_equal(pred.ravel(), Y.ravel()):
                converged = True
                break

    # alignment at FINAL 
    if align_at == "final":
        W2_final = toW_W(G2)
        W_last = W2_final[:-1, :].T
        B_final = toW_B(GB)
        alignment_value = cosine_similarity_vec(
            W_last.ravel(), B_final.ravel()
        )

    # final hidden activations
    W1_final = toW_W(G1)
    h_final = tanh(Xb @ W1_final)  
    return alignment_value, converged, w_vec, b_vec, B0, u0, cancel, h_final


# ------------------------------------------------------------
# SWEEP MANY TRIALS

def main():

    aligns = np.empty(N_TRIALS, dtype=float)
    succ   = np.empty(N_TRIALS, dtype=int)

    wvecs = np.zeros((N_TRIALS, HIDDEN_DIM), dtype=float)
    bvecs = np.zeros((N_TRIALS, HIDDEN_DIM), dtype=float)

    B0_all = np.zeros((N_TRIALS, HIDDEN_DIM), dtype=float)    
    u0_all = np.zeros((N_TRIALS, 4, HIDDEN_DIM), dtype=float)  
    cancel0 = np.zeros(N_TRIALS, dtype=float)                

    hF_all = np.zeros((N_TRIALS, 4, HIDDEN_DIM), dtype=float)

    print(f"Running {N_TRIALS} trials | lr={LEARNING_RATE}, epochs={EPOCHS}, align_at='{ALIGN_AT}'")

    for t in range(N_TRIALS):
        a, ok, wv, bv, B0, u0, c0, hF = run_one(MASTER_SEED + 1 + t)
        aligns[t] = a
        succ[t] = 1 if ok else 0

        if wv is not None: wvecs[t, :] = wv
        if bv is not None: bvecs[t, :] = bv

        B0_all[t,:] = B0
        u0_all[t,:,:] = u0
        cancel0[t] = c0

        hF_all[t, :, :] = hF

        if (t+1) % 200 == 0:
            print(f"  ... {t+1}/{N_TRIALS}")

    # Bin alignment -> success rate
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


    # Seperating left dip and right dips
    fail_idx = np.where(succ == 0)[0]
    a_fail = aligns[fail_idx]

    # sort failures by alignment
    ordf = np.argsort(a_fail)
    af = a_fail[ordf]
    idxf = fail_idx[ordf]

    # find best split point = 2-means in 1D)
    best_j = None
    best_score = -1.0

    # prefix sums for fast mean computation
    pref = np.cumsum(af)
    n = len(af)

    for j in range(1, n):  # split after j-1
        n1, n2 = j, n - j
        m1 = pref[j-1] / n1
        m2 = (pref[-1] - pref[j-1]) / n2
        score = (m2 - m1)**2  # bigger gap between means => better split
        if score > best_score:
            best_score = score
            best_j = j

    left_idx  = idxf[:best_j]   # more negative group  -> LEFT dip
    right_idx = idxf[best_j:]   # less negative group  -> RIGHT dip

    # Left dip
    def norm2(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    W_dirs = np.array([norm2(wvecs[i]) for i in left_idx])
    B_dirs = np.array([norm2(bvecs[i]) for i in left_idx])

    plt.figure(figsize=(7,7))
    ax = plt.gca()

    # W arrows
    for v in W_dirs:
        ax.arrow(0, 0, v[0], v[1], head_width=0.03, length_includes_head=True, alpha=0.25, color='tab:blue')

    # B arrows
    for v in B_dirs:
        ax.arrow(0, 0, v[0], v[1], head_width=0.03, length_includes_head=True, alpha=0.25, color='tab:orange')

    ax.set_title("Init vectors for FAILED runs")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.show()

    k = left_idx[3]  # pick a random failure from the left dip
    print("align0:", aligns[k], "success:", succ[k])
    print("B0:", B0_all[k])
    print("u0 per sample:\n", u0_all[k])
    print("cancel0:", cancel0[k])

    h1, h2 = hF_all[k][:,0], hF_all[k][:,1]
    simF = np.dot(h1, h2) / ((np.linalg.norm(h1)+1e-12)*(np.linalg.norm(h2)+1e-12))
    print("final hidden similarity (cos):", simF)   
    print(hF_all[k][:,0])
    print(hF_all[k][:,1])
    print("h2 + h1:", hF_all[k][:,1] + hF_all[k][:,0])

    # Right dip
    k = right_idx[3]  # pick a random failure from the right dip
    print("align0:", aligns[k], "success:", succ[k])
    print("B0:", B0_all[k])
    print("u0 per sample:\n", u0_all[k])
    print("cancel0:", cancel0[k])

    h1, h2 = hF_all[k][:,0], hF_all[k][:,1]
    simF = np.dot(h1, h2) / ((np.linalg.norm(h1)+1e-12)*(np.linalg.norm(h2)+1e-12))
    print("final hidden similarity (cos):", simF)
    print(hF_all[k][:,0])
    print(hF_all[k][:,1])
    print("h2 + h1:", hF_all[k][:,1] + hF_all[k][:,0])


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
