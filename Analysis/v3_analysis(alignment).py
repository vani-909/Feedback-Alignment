#--------------------------------------------------------------------------------------------------
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS

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
# TRAIN 

def alignment(vec_a, vec_b, eps=1e-12):
    a = vec_a.reshape(-1)
    b = vec_b.reshape(-1)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def record(ep, W1, W2, B, hidden_dim, alignment_history, pdirections_history):
    # 1. Alignment between forward readout and feedback:
    # Use W2 without bias, compare to B.T 
    W2_no_bias = W2[:-1, :].reshape(-1)           
    B_col = B.T.reshape(-1)                       
    alignment_history.append((ep, alignment(W2_no_bias, B_col)))

    # 2. Preferred directions of hidden units: 
    # Use W1 without bias, normalized
    W1_no_bias = W1[:-1, :]                   
    ps = []
    for j in range(hidden_dim):
        v = W1_no_bias[:, j]
        n = np.linalg.norm(v)
        ps.append((v / (n + 1e-12)))
    pdirections_history.append((ep, np.array(ps)))  

def train(seed=21, record_every=5):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, np.sqrt(1/(input_dim+1)), size=(input_dim+1, hidden_dim))
    W2 = rng.normal(0, np.sqrt(1/(hidden_dim+1)), size=(hidden_dim+1, output_dim))

    W1[-1, :] = np.array([+0.5, -0.5], dtype=W1.dtype)

    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1.0, size=(output_dim, hidden_dim))  
    Q, _ = np.linalg.qr(B.T)     
    B = Q.T 

    print("\nInitialization...")
    print("\nW1:  ", W1)
    print("\nW2:  ", W2)
    print("\nB:  ", B)

    idx = np.arange(len(X))
    alignment_history = []
    pdirections_history = []  
    s_history = [(0, np.nan)]  # (epoch, ⟨eB, eW2^T⟩)

    record(0, W1, W2, B, hidden_dim, alignment_history, pdirections_history)

    for ep in range(1, epochs + 1):
        rng.shuffle(idx)
        Xb = add_bias(X[idx])
        Yb = Y[idx]

        # forward
        h  = relu(Xb @ W1)
        hb = add_bias(h)
        y  = sigmoid(hb @ W2)

        # FA hidden error 
        e = (y - Yb)                                  
        d_hid = (e @ B) * d_relu(h)                   

        # SGD
        W2 -= lr * (hb.T @ e) / len(X)
        W1 -= lr * (Xb.T @ d_hid) / len(X)

        if (ep % record_every) == 0 or ep == epochs:
            record(ep, W1, W2, B, hidden_dim, alignment_history, pdirections_history)
            # (eB).(eW2^T)
            W2_nb = W2[:-1, :].reshape(1, -1)                
            EB    = e @ B                                     
            EW2T  = e @ W2_nb                                 
            s_t   = float(np.sum(EB * EW2T))                  # Frobenius inner product
            s_history.append((ep, s_t))

    # final metrics
    y_fin = sigmoid(add_bias(relu(add_bias(X) @ W1)) @ W2)
    loss  = binary_cross_entropy(y_fin, Y)
    acc   = float(np.mean((y_fin >= 0.5).astype(int) == Y))

    return {
        "B": B,
        "W1": W1,
        "W2": W2,
        "alignment_history": np.array(alignment_history),        # (n_points, 2): [epoch, cosine]
        "pdirections_history": pdirections_history,              # list of (epoch, [[p1x,p1y],...])
        "s_history": np.array(s_history),                        # (n_points, 2): [epoch, (eB).(eW2^T)]
        "loss": loss,
        "acc": acc,
        "outputs": y_fin.ravel()
    }

#--------------------------------------------------------------------------------------------------
# RUN 

res = train(seed=21, record_every=5)
print("\nAfter training...")
print("\nW1:  ", res['W1'])
print("\nW2:  ", res['W2'])
print("\nB:  ", res['B'])
print(f"Final acc: {res['acc']:.3f} | Final loss: {res['loss']:.4f}")
print("Final outputs:", np.round(res["outputs"], 3))


#--------------------------------------------------------------------------------------------------
# PLOT

# 1) Alignment curve: cos(W2^T, B)
epochs_arr = res["alignment_history"][:,0]
cos_arr    = res["alignment_history"][:,1]

plt.figure()
plt.plot(epochs_arr, cos_arr)
plt.xlabel("Epoch")
plt.ylabel("Alignment (cosine)")   # 1.0 = perfectly aligned
plt.grid(True)
plt.show()

# 2) Preferred direction evolution 
checkpoints = [e for (e, _) in res["pdirections_history"]]

plt.figure()
th = np.linspace(0, 2*np.pi, 361)  # unit circle
plt.plot(np.cos(th), np.sin(th), linewidth=1)

for (ep, P) in res["pdirections_history"]:
    for j in range(P.shape[0]):
        px, py = P[j]
        plt.arrow(0, 0, px, py, head_width=0.05, head_length=0.07, length_includes_head=True, alpha=0.6)

plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(True)
plt.show()

# 3) (eB).(eW2^T)
ep_s = res["s_history"][:,0]
svals = res["s_history"][:,1]
plt.figure()
plt.plot(ep_s, svals)
plt.axhline(0, linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("(eB)·(eW2^T)")
plt.grid(True)
plt.show()
