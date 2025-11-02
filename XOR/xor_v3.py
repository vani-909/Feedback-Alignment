#--------------------------------------------------------------------------------------------------
#IMPORTS

import numpy as np
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------
# FUNCTION DEFENITIONS

# Hardware readout - cancels the shift
def readout(Xv, G, S):
    return Xv @ G - S * np.sum(Xv, axis=1, keepdims=True)     

def relu(z): 
    return np.maximum(0, z)

def d_relu(z): 
    return (z > 0).astype(float)

def sigmoid(z): return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))


#--------------------------------------------------------------------------------------------------
# HEATMAP VISUALIZATION FUNCTIONS

def plot_decision_boundary(X, Y, W1, W2):
    """Plot decision boundary of the network"""
    # Create a grid of points
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict for each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_bias = add_bias(grid_points)
    
    h_grid = relu(grid_points_bias @ W1)
    y_grid = sigmoid(add_bias(h_grid) @ W2)
    Z = y_grid.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(label='Output probability')
    
    # Plot training data
    colors = ['red' if y == 0 else 'blue' for y in Y.flatten()]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='black', label='XOR Data')
    
    plt.title('Decision Boundary Heatmap')
    plt.xlabel('Input X1')
    plt.ylabel('Input X2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hidden_activations(X, W1):
    """Plot heatmap of hidden layer activations"""
    # Create a grid of points
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_bias = add_bias(grid_points)
    
    # Get hidden activations
    h_grid = relu(grid_points_bias @ W1)
    
    # Plot each hidden neuron's activation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(2):
        Z = h_grid[:, i].reshape(xx.shape)
        im = axes[i].contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title(f'Hidden Neuron {i+1} Activation')
        axes[i].set_xlabel('Input X1')
        axes[i].set_ylabel('Input X2')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_weight_heatmaps(W1, W2, B):
    """Plot heatmaps of weight matrices"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # W1 heatmap
    im1 = axes[0].imshow(W1, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('W1 Weights (Input → Hidden)')
    axes[0].set_xlabel('Hidden Neurons')
    axes[0].set_ylabel('Input Features + Bias')
    plt.colorbar(im1, ax=axes[0])
    
    # W2 heatmap  
    im2 = axes[1].imshow(W2, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('W2 Weights (Hidden → Output)')
    axes[1].set_xlabel('Output Neurons')
    axes[1].set_ylabel('Hidden Neurons + Bias')
    plt.colorbar(im2, ax=axes[1])
    
    # B heatmap
    im3 = axes[2].imshow(B, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('B Feedback Matrix')
    axes[2].set_xlabel('Hidden Neurons')
    axes[2].set_ylabel('Output Neurons')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()



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
    
    h = relu(x_sample @ W1)  # CHANGED: tanh → relu
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

    # forward - CHANGED: tanh → relu for hidden layer
    h = relu(Xb @ W1)                 # (4,2) 
    hb = add_bias(h)                  # (4,3)
    y = sigmoid(hb @ W2)              # (4,1) 

    # BCE-with-sigmoid: dL/dy_lin = (y - Y)
    e = (y - Yb)                      # (4,1)
    d_hid = (e @ B) * d_relu(h)       # (4,2) 

    # batch updates
    W2 -= lr * (hb.T @ e) / len(X)
    W1 -= lr * (Xb.T @ d_hid) / len(X)

    # Monitor training progress
    if ep % 10 == 0:
        h_val = relu(add_bias(X) @ W1)  
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
    

h = relu(add_bias(X) @ W1)  
y = sigmoid(add_bias(h) @ W2)
pred = (y >= 0.5).astype(int)
print("Outputs:", y.round(3).ravel())
print("Pred   :", pred.ravel().tolist(), " Targets:", Y.ravel().tolist())


#-------------------------------------------------------------------------------------------
# HEATMAP VISUALIZATIONS

# 1. Decision Boundary Heatmap
print("Plotting decision boundary...")
plot_decision_boundary(X, Y, W1, W2)

# 2. Hidden Layer Activations Heatmap  
print("Plotting hidden layer activations...")
plot_hidden_activations(X, W1)

# 3. Weight Matrices Heatmaps
print("Plotting weight matrices...")
plot_weight_heatmaps(W1, W2, B)


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
B_hardware = B + S  

print("\n Offset = ", S)
print("\nLayer1 (W1) programmed conductances G1:\n", G1.round(4))
print("\nLayer2 (W2) programmed conductances G2:\n", G2.round(4))
print("\nB matrix conductances:\n", B_hardware.round(4))


# Re-run forward using the "hardware view"
h_hw = relu(readout(add_bias(X), G1, S))  
y_hw = sigmoid(readout(add_bias(h_hw), G2, S))

print("\nHardware outputs:", y_hw.round(3).ravel())
print("Matches software:", np.allclose(y_hw, y, atol=1e-3))


#-------------------------------------------------------------------------------------------------
'''
NOTES:

1. Changed tanh to ReLu
2. Heatmaps implemented

'''