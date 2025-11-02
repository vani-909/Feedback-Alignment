#--------------------------------------------------------------------------------------------------
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------
# FUNCTIONS
def relu(z): return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(float)
def sigmoid(z): return 1/(1+np.exp(-z))
def add_bias(A): return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)
def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true*np.log(y_pred+1e-8) + (1-y_true)*np.log(1-y_pred+1e-8))

#--------------------------------------------------------------------------------------------------
# DATA
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

input, hidden, output = 2, 2, 1
lr, epochs = 0.3, 1000
rng = np.random.default_rng(42)


#--------------------------------------------------------------------------------------------------

def _init_WB(rng_local):
    W1 = rng_local.normal(0, np.sqrt(1/(input+1)), size=(input+1, hidden))
    W2 = rng_local.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))
    W1[-1, :] = np.array([+0.5, -0.5])
    # B with QR as in your script (alignment screen)
    max_attempts, min_alignment = 50, -0.1
    B_local = None
    for _ in range(max_attempts):
        B_try = rng_local.normal(0, 1.0, size=(output, hidden))
        B_try, _ = np.linalg.qr(B_try.T)
        B_try = B_try.T
        x_s = add_bias(X[0:1, :]); y_t = Y[0:1, :]
        h_s = relu(x_s @ W1); y_p = sigmoid(add_bias(h_s) @ W2); e_s = y_p - y_t
        W2_hidden = W2[:-1, :]
        align = e_s.item() * (W2_hidden.T @ B_try.T).item() * e_s.item()
        if align > min_alignment:
            B_local = B_try; break
    if B_local is None: B_local = B_try
    return W1, W2, B_local

def train_trace(lr=0.3, epochs=1000, seed=42):
    rng_local = np.random.default_rng(seed)
    W1, W2, B_local = _init_WB(rng_local)
    idx = np.arange(len(X))
    losses, accs = [], []
    for ep in range(epochs):
        rng_local.shuffle(idx)
        Xb, Yb = add_bias(X[idx]), Y[idx]
        h  = relu(Xb @ W1); hb = add_bias(h); y  = sigmoid(hb @ W2)
        e  = (y - Yb); d_hid = (e @ B_local) * d_relu(h)
        W2 -= lr * (hb.T @ e) / len(X); W1 -= lr * (Xb.T @ d_hid) / len(X)
        y_val = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
        losses.append(binary_cross_entropy(y_val, Y))
        accs.append(float(np.mean((y_val >= 0.5).astype(int) == Y)))
        if not np.isfinite(losses[-1]): break
    return np.array(losses), np.array(accs)

def classify_run(losses):
    if len(losses) < 3 or not np.all(np.isfinite(losses)): return "diverged"
    diffs = np.diff(losses)
    inc = np.sum(diffs > 0)
    if losses[-1] > 5*losses[0] or np.max(losses) > 10*losses[0]:
        return "diverged"
    if inc == 0:
        return "gradual"
    return "oscillating"

def plot_trajectory(losses, title=""):
    import matplotlib.pyplot as plt
    n = len(losses)
    ks = np.linspace(0, n-1, min(15, n)).astype(int)
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(n), losses)
    for i in range(len(ks)-1):
        a, b = ks[i], ks[i+1]
        plt.annotate("", xy=(b, losses[b]), xytext=(a, losses[a]),
                     arrowprops=dict(arrowstyle="->", lw=1.2))
        plt.plot([a], [losses[a]], marker="o")
    plt.plot([n-1], [losses[-1]], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# Example: visualize lr=0.3, epochs=1000 (change seed to test others)
losses, accs = train_trace(lr=0.1, epochs=1000, seed=21)
state = classify_run(losses)
plot_trajectory(losses, title=f"lr=0.3, epochs=1000  -> {state}")
