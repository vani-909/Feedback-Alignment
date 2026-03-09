#--------------------------------------------------------------------------------------------------
#IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#--------------------------------------------------------------------------------------------------
# FUNCTION DEFENITIONS

# scaled tanh with beta=0.5
def tanh(z, beta=0.5):
    return np.tanh(beta * z)

def d_tanh(a, beta=0.5):
    # a = tanh(beta z)
    return beta * (1.0 - a**2)

def sigmoid(z): return 1/(1+np.exp(-z))

def add_bias(A):
    return np.concatenate([A, np.ones((A.shape[0],1), dtype=A.dtype)], axis=1)

def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))


#--------------------------------------------------------------------------------------------------
# HEATMAP VISUALIZATION FUNCTIONS

def plot_decision_boundary(X_raw, Y, W1, W2):
    # Create a grid 
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx_raw, yy_raw = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Flatten grid and map to network input space [-1,1]
    grid_raw = np.c_[xx_raw.ravel(), yy_raw.ravel()]      # in "visual" space
    grid_net = 2 * grid_raw - 1                           # what the network sees
    grid_net_bias = add_bias(grid_net)

    # Forward pass on grid
    h_grid = tanh(grid_net_bias @ W1)
    y_grid = sigmoid(add_bias(h_grid) @ W2)
    Z = y_grid.reshape(xx_raw.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx_raw, yy_raw, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Output probability')

    # Plot training data in raw 0/1 coordinates
    colors = ['red' if y == 0 else 'blue' for y in Y.flatten()]
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=colors,
                s=100, edgecolors='black', label='XOR Data')

    plt.title('Decision Boundary Heatmap')
    plt.xlabel('Input X1')
    plt.ylabel('Input X2')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


def plot_hidden_activations(X_raw, W1):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx_raw, yy_raw = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid_raw = np.c_[xx_raw.ravel(), yy_raw.ravel()]
    grid_net = 2 * grid_raw - 1        # map to network space
    grid_net_bias = add_bias(grid_net)

    # Hidden activations
    h_grid = tanh(grid_net_bias @ W1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(2):
        Z = h_grid[:, i].reshape(xx_raw.shape)
        im = axes[i].contourf(xx_raw, yy_raw, Z, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title(f'Hidden Neuron {i+1} Activation')
        axes[i].set_xlabel('Input X1')
        axes[i].set_ylabel('Input X2')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_weight_heatmaps(W1, W2, B):
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

# XOR Data
X_raw = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4,2)
X = 2*X_raw - 1   # now in {-1, +1}

Y = np.array([[0],[1],[1],[0]], dtype=np.float32)          # (4,1)

# Network architecture
input, hidden, output = 2, 2, 1

# Random number generator
rng = np.random.default_rng()

# Weights (can be negative). Init - Xavier
W1 = rng.normal(0, np.sqrt(1/(input+1)), size=(input+1, hidden))     # (3x2)
W2 = rng.normal(0, np.sqrt(1/(hidden+1)), size=(hidden+1, output))  # (3x1)

# Bias
W1[-1, :] = np.array([+0.1, -0.1])  


# Random feedback matrix B
B = rng.normal(0, np.sqrt(1/(hidden+1)), size=(output, hidden))
B, _ = np.linalg.qr(B.T)
B = B.T

print("Before Trianing Weights:")
print("W1:\n", W1)
print("W2:\n", W2)
print("B:\n", B)


#--------------------------------------------------------------------------------------------
# Conductance Mapping

G_off = 1.1669e-3
G_on  = 1.3327e-3

Wmax = 10
Bmax = 1

G0 = 0.5 * (G_on + G_off)
dG = (G_on - G_off)

alpha_W = 2.0 * Wmax / dG
alpha_B = 2.0 * Bmax / dG

toG_W = lambda W: np.clip(G0 + (W / alpha_W), G_off, G_on)
toW_W = lambda G: alpha_W * (G - G0)

toG_B = lambda Bm: np.clip(G0 + (Bm / alpha_B), G_off, G_on)
toW_B = lambda G: alpha_B * (G - G0)

G1, G2, GB = toG_W(W1), toG_W(W2), toG_B(B)

#--------------------------------------------------------------------------------------------
# TRAINING

lr = 1.0         
epochs = 1000

idx = np.arange(len(X))

# Store training history
train_losses = []
train_accuracies = []
convergence_epoch = None
epoch_numbers = []
align_epochs, align_vals = [], []

# Record for animation
record_every = 5  
ep_hist = []
w_hist = []       # normalized W2 (2D, no bias)
b_fixed = None    # normalized B (2D)
cos_hist = []     # cosine similarity between w and b


for ep in range(epochs):
    rng.shuffle(idx)
    Xb = add_bias(X[idx])             # (4,3) = (0,0,1), (0,1,1), (1,0,1), (1,1,1)
    Yb = Y[idx]

    # Forward pass
    W1_eff = toW_W(G1)                # Read from hardware
    W2_eff = toW_W(G2)
    B_eff  = toW_B(GB)

    h = tanh(Xb @ W1_eff)                 # (4,2) => tanh(wx + b)
    hb = add_bias(h)                      # (4,3) => (h, 1)
    y = sigmoid(hb @ W2_eff)              # (4,1)


    # Backward pass
    e = (y - Yb)                          # (4,1)

    # FA update direction
    d_hid = (e @ B_eff) * d_tanh(h)       # (4,2) 

    # Updates
    lr_W2 = lr
    lr_W1 = 0.25 * lr  

    W2_eff -= lr_W2 * (hb.T @ e)     / len(X)
    W1_eff -= lr_W1 * (Xb.T @ d_hid) / len(X)

    G1 = toG_W(W1_eff)           # Write to hardware
    G2 = toG_W(W2_eff)

    # Synchronize logical weights
    W1 = W1_eff.copy()
    W2 = W2_eff.copy()

    # Record for animation
    if ep % record_every == 0:
        w = W2_eff[:-1, 0].copy()    
        b = B_eff[0, :].copy()       

        w_n = w / (np.linalg.norm(w) + 1e-12)
        b_n = b / (np.linalg.norm(b) + 1e-12)

        if b_fixed is None:
            b_fixed = b_n.copy()

        ep_hist.append(ep)
        w_hist.append(w_n)
        cos_hist.append(float(np.dot(w_n, b_fixed)))

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

        # Check alignment
        W = W2_eff[:-1, :].T  
        g_true = e @ W         
        g_fb   = e @ B_eff           
        align_epochs.append(ep)
        align_vals.append(float(np.mean(np.sum(g_true * g_fb, axis=1))))  # ⟨e^T W B e⟩
            
        if ep % 100 == 0:
            print(f"Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
        

print("After Training Weights:")
print("W1:\n", W1)
print("W2:\n", W2)
print("B:\n", B)

#--------------------------------------------------------------------------------------------
# INFERENCE     

h = tanh(add_bias(X) @ W1)  
y = sigmoid(add_bias(h) @ W2)
pred = (y >= 0.5).astype(int)
print("Outputs:", y.round(3).ravel())
print("Pred   :", pred.ravel().tolist(), " Targets:", Y.ravel().tolist())


#-------------------------------------------------------------------------------------------
# # PLOTS

# 1. Decision Boundary Heatmap
print("Plotting decision boundary...")
plot_decision_boundary(X_raw, Y, W1, W2)

# 2. Hidden Layer Activations Heatmap  
print("Plotting hidden layer activations...")
plot_hidden_activations(X_raw, W1)

# 3. Weight Matrices Heatmaps
print("Plotting weight matrices...")
plot_weight_heatmaps(W1, W2, B)

# 4. Loss and Accuracy Plots
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

# 5. Alignment
plt.figure(figsize=(5,3))
plt.plot(align_epochs, align_vals, marker='o')
plt.axhline(0.0, ls='--', alpha=0.5)
plt.title('Alignment ⟨e^T W B e⟩')
plt.xlabel('Epoch'); plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout(); 
plt.show()

#-----------------------------------------------------------------------------------------
# ANIMATION: W2 vector rotates towards fixed B vector

def save_alignment_animation(ep_hist, w_hist, b_fixed, cos_hist,
                             out_mp4="fa_alignment.mp4",
                             out_gif="fa_alignment.gif",
                             fps=30):

    if b_fixed is None or len(w_hist) == 0:
        print("Nothing recorded for animation. Increase epochs or reduce record_every.")
        return

    w_arr = np.array(w_hist)   
    ep_arr = np.array(ep_hist)
    cos_arr = np.array(cos_hist)

    fig, (axV, axC) = plt.subplots(1, 2, figsize=(11, 4))

    # Vector panel
    axV.set_title("Vector view (normalized)")
    axV.set_xlim(-1.2, 1.2)
    axV.set_ylim(-1.2, 1.2)
    axV.set_aspect('equal', 'box')
    axV.grid(True, alpha=0.3)
    axV.axhline(0, alpha=0.4)
    axV.axvline(0, alpha=0.4)

    # Plot fixed B arrow once
    b0 = np.array(b_fixed, dtype=float)
    b_line, = axV.plot([0, b0[0]], [0, b0[1]], lw=3, label="B (fixed)")
    w_line, = axV.plot([0, w_arr[0, 0]], [0, w_arr[0, 1]], lw=3, label="W2 (evolving)")
    txt = axV.text(-1.15, 1.05, "", fontsize=10)

    axV.legend(loc="lower left")

    # Cosine panel
    axC.set_title("cos(W2, B) over time")
    axC.set_xlim(ep_arr.min(), ep_arr.max())
    axC.set_ylim(-1.05, 1.05)
    axC.grid(True, alpha=0.3)
    axC.axhline(0, ls="--", alpha=0.5)

    cos_line, = axC.plot([], [], lw=2)
    cos_dot,  = axC.plot([], [], marker="o")

    def init():
        cos_line.set_data([], [])
        cos_dot.set_data([], [])
        txt.set_text("")
        return (b_line, w_line, cos_line, cos_dot, txt)

    def update(i):
        w = w_arr[i]
        w_line.set_data([0, w[0]], [0, w[1]])

        cos_line.set_data(ep_arr[:i+1], cos_arr[:i+1])
        cos_dot.set_data([ep_arr[i]], [cos_arr[i]])

        txt.set_text(f"epoch={ep_arr[i]}   cos={cos_arr[i]:+.3f}")
        return (b_line, w_line, cos_line, cos_dot, txt)

    anim = FuncAnimation(fig, update, frames=len(ep_arr), init_func=init,
                         interval=1000/fps, blit=True)

    # Try MP4 (ffmpeg) first, else GIF
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(out_mp4, writer=writer)
        print(f"Saved video: {out_mp4}")
    except Exception as e_mp4:
        print(f"MP4 save failed ({e_mp4}). Trying GIF...")
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(out_gif, writer=writer)
            print(f"Saved gif: {out_gif}")
        except Exception as e_gif:
            print(f"GIF save also failed: {e_gif}")

    plt.close(fig)

# Run animation save
save_alignment_animation(ep_hist, w_hist, b_fixed, cos_hist,
                         out_mp4="fa_alignment.mp4",
                         out_gif="fa_alignment.gif",
                         fps=30)

