import numpy as np


# Shifting weights to non-negative conductances
def mapping(W, eps=1e-3): 
    """
    Input: math weights W (R x C)
    Return: shifted weights G (R x C) and per-row offsets S (R,) 
    """
    
    row_min = W.min(axis=1)
    S = np.maximum(0.0, -row_min) + eps  # ensure strictly >0
    G = W + S[:, None]
    return G, S


# Hardware readout - cancels the shift
def readout(Xv, G, S):
    return Xv @ G - (Xv @ S[:,None])     # (sum_i V_i * (G_ij) - sum_i V_i * S_i)


def tanh(z): return np.tanh(z)
def d_tanh(y): return 1 - y**2

def sigmoid(z): return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

#--------------------------------------------------------------------------------------------

# XOR  
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4,2)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)          # (4,1)

# Network architecture
input, hidden, output = 2, 2, 1

rng = np.random.default_rng(3)

# Weights (can be negative). Init Xavier
W1 = rng.normal(0,np.sqrt(1/(input+1)), size=(input+1, hidden))   # (2x2)
W2 = rng.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))  # (2x1)

W1[-1, :] = np.array([+0.5, -0.5])  # different hidden bias 

# Random feedback matrix B
B = rng.normal(0, 1.0, size=(output, hidden))   # (1x2)
B /= (np.linalg.norm(B) + 1e-12)


lr = 2          # 1
epochs = 500    # 1000

idx = np.arange(len(X))

for ep in range(epochs):
    rng.shuffle(idx)
    Xb = add_bias(X[idx])
    Yb = Y[idx]

    # forward
    h = tanh(Xb @ W1)                 # (4,2)
    hb = add_bias(h)                  # (4,3)
    y = sigmoid(hb @ W2)              # (4,1)

    # BCE-with-sigmoid: dL/dy_lin = (y - Y)
    e = (y - Yb)                      # (4,1)
    d_hid = (e @ B) * d_tanh(h)       # (4,2)

    # eWBe >1 ??

    # batch updates
    W2 -= lr * (hb.T @ e) / len(X)
    W1 -= lr * (Xb.T @ d_hid) / len(X)

h = tanh(add_bias(X) @ W1)
y = sigmoid(add_bias(h) @ W2)
pred = (y >= 0.5).astype(int)
print("Outputs:", y.round(3).ravel())
print("Pred   :", pred.ravel().tolist(), " Targets:", Y.ravel().tolist())


#-------------------------------------------------------------------------------------------


G1, S1 = mapping(W1)   # Layer 1: (2x2), S1 shape (2,)
G2, S2 = mapping(W2)   # Layer 2: (2x1), S2 shape (2,)


print("\nLayer1 (W1) programmed conductances G1:\n", G1.round(4))
print("Layer1 row-offsets S1 (reference column devices):", S1.round(4))
print("\nLayer2 (W2) programmed conductances G2:\n", G2.round(4))
print("Layer2 row-offsets S2 (reference column devices):", S2.round(4))


# Re-run forward using the "hardware view"
h_hw = tanh(readout(add_bias(X), G1, S1))
y_hw = sigmoid(readout(add_bias(h_hw), G2, S2))

print("\nHardware outputs:", y_hw.round(3).ravel())


# Change 6 references to 1
# Error progression - trainig
# heatmap