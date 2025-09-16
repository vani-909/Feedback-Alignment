#--------------------------------------------------------------------------------------------------
#IMPORTS

import numpy as np
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------
# FUNCTION DEFENITIONS

# Hardware readout - cancels the shift
def readout(Xv, G, S):
    return Xv @ G - S * np.sum(Xv, axis=1, keepdims=True)     


def tanh(z): return np.tanh(z)
def d_tanh(y): return 1 - y**2

def sigmoid(z): return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))



#--------------------------------------------------------------------------------------------
# Initialization 

# XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4,2)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)          # (4,1)

# Network architecture
input, hidden, output = 2, 2, 1

rng = np.random.default_rng()

# Weights (can be negative). Init Xavier
W1 = rng.normal(0,np.sqrt(1/(input+1)), size=(input+1, hidden))   # (2x2)
W2 = rng.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))  # (2x1)

W1[-1, :] = np.array([+0.5, -0.5])  # different hidden bias 

# Random feedback matrix B
max_attempts = 50  
min_alignment = -0.1  # Allow slightly negative alignment

for attempt in range(max_attempts):
    B = rng.normal(0, 1.0, size=(output, hidden))
    B, _ = np.linalg.qr(B.T)
    B = B.T
    
    x_sample = add_bias(X[0:1, :])
    y_true = Y[0:1, :]
    
    h = tanh(x_sample @ W1)
    y_pred = sigmoid(add_bias(h) @ W2)
    e_sample = y_pred - y_true
    
    W2_hidden = W2[:-1, :]
    intermediate = W2_hidden.T @ B.T
    alignment = e_sample.item() * intermediate.item() * e_sample.item()
    
    if alignment > min_alignment: 
        print(f"Using B with alignment: {alignment:.4f}")
        break
else:
    print("Using random B (no suitable found)")


#--------------------------------------------------------------------------------------------
# TRAINING

lr = 0.3         
epochs = 500    

idx = np.arange(len(X))

# Store training history
train_losses = []
train_accuracies = []
convergence_epoch = None
epoch_numbers = []

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

    # batch updates
    W2 -= lr * (hb.T @ e) / len(X)
    W1 -= lr * (Xb.T @ d_hid) / len(X)

    # Monitor training progress
    if ep % 10 == 0:
        h_val = tanh(add_bias(X) @ W1)
        y_val = sigmoid(add_bias(h_val) @ W2)
        
        loss = binary_cross_entropy(y_val, Y)
        pred_val = (y_val >= 0.5).astype(int)
        accuracy = np.mean(pred_val == Y)
        
        epoch_numbers.append(ep)
        train_losses.append(loss)
        train_accuracies.append(accuracy)
        
        # Check for convergence
        if np.array_equal(pred_val.ravel(), Y.ravel()) and convergence_epoch is None:
            convergence_epoch = ep
            print(f"Converged at epoch {ep}")
            # break
            
        if ep % 100 == 0:
            print(f"Epoch {ep}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
    

h = tanh(add_bias(X) @ W1)
y = sigmoid(add_bias(h) @ W2)
pred = (y >= 0.5).astype(int)
print("Outputs:", y.round(3).ravel())
print("Pred   :", pred.ravel().tolist(), " Targets:", Y.ravel().tolist())


# ---------------------------------------------------------------------------------------------------
# PLOTS

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epoch_numbers, train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epoch_numbers, train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.ylim(0, 1.1)

if convergence_epoch is not None:
        plt.axvline(x=convergence_epoch, color='r', linestyle='--', alpha=0.7, label=f'Converged at {convergence_epoch}')
        plt.legend()

plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------------------
# HARDWARE READOUT

S = max(-W1.min(), -W2.min(), -B.min()) + 1e-3  # with eps

G1 = W1 + S
G2 = W2 + S
B = B + S

print("\n Offset = ", S)
print("\nLayer1 (W1) programmed conductances G1:\n", G1.round(4))
print("\nLayer2 (W2) programmed conductances G2:\n", G2.round(4))
print("\nB matrix:\n", B.round(4))


# Re-run forward using the "hardware view"
h_hw = tanh(readout(add_bias(X), G1, S))
y_hw = sigmoid(readout(add_bias(h_hw), G2, S))

print("\nHardware outputs:", y_hw.round(3).ravel())



#-------------------------------------------------------------------------------------------------
'''
NOTES:

1. Per-row offset changed to a single global offset
2. Seed not set -> not same outcome everytime (ALWAYS CORRECT: seed = 3  => lr = 2, epochs = 500 {too high lr and ep, may diverge})
3. lr = 0.2 and epochs = 500; convergence check added
4. Orthogonal initialization of B
5. Error monitoring during training and added plots
6. Remove output bias - didn't work. Poor performance
7. Reducing lr every 100 epochs didn't work
8. Rejection sampling for valid orthogonal B
9. Was that too restrictive? New approach allowing slight negative alignment.
10. No. of devices = 6 connections + 3 biases + 1 S + B (1, 2)  = 12 devices

'''