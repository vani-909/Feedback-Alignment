import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit  

def sigmoid(x, y0, L, k, x0):
    return y0 + L * expit(k * (x - x0))


df = pd.read_csv("./Sample/transfer_Vg_Id.csv")

# median curve over all cycles
df = df.groupby("Vg (V)", as_index=False)["Id (A)"].median()

x = df["Vg (V)"].to_numpy(float)
y = df["Id (A)"].to_numpy(float)

# sort 
i = np.argsort(x)
x, y = x[i], y[i]

# better initial guesses from plateaus 
n = max(5, int(0.1 * len(x)))          # 10% at ends
y_low  = float(np.median(y[:n]))       # left plateau
y_high = float(np.median(y[-n:]))      # right plateau
L_guess  = y_high - y_low
y0_guess = y_low

y_half = y0_guess + 0.5 * L_guess
x0_guess = float(x[np.argmin(np.abs(y - y_half))])

# slope-based k guess 
dy_dx = np.gradient(y, x)
slope0 = float(dy_dx[np.argmin(np.abs(x - x0_guess))])

# max slope ≈ L*k/4  => k ≈ 4*slope/L
k_guess = abs(4.0 * slope0 / (L_guess + 1e-30))
k_guess = float(np.clip(k_guess, 0.1, 200.0))

p0 = [y0_guess, L_guess, k_guess, x0_guess]

# bounds to prevent near-linear bad fits 
ymin, ymax = float(y.min()), float(y.max())
bounds = (
    [ymin - abs(L_guess)*2, -np.inf,  0.0,  x.min()-0.5],   # y0, L, k, x0
    [ymax + abs(L_guess)*2,  np.inf,  1e4,  x.max()+0.5],
)

params, cov = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=200000)
y0, L, k, x0 = params
print(f"y0={y0:.6e}, L={L:.6e}, k={k:.4f}, x0={x0:.4f}")

xx = np.linspace(x.min(), x.max(), 600)
plt.figure()
plt.scatter(x, y, s=12, label="data")
plt.plot(xx, sigmoid(xx, *params), label="sigmoid fit")
plt.grid(True); plt.legend()
plt.xlabel("Vg (V)"); plt.ylabel("Id (A)")
plt.show()

# range
vg_min, vg_max = float(np.min(x)), float(np.max(x))
id_min, id_max = float(np.min(y)), float(np.max(y))
print(f"Vg range : [{vg_min:.6f}, {vg_max:.6f}] V")
print(f"Id range : [{id_min:.6e}, {id_max:.6e}] A")

# 0 point
print("Zero-point: x =", x0, " y =", y0 + 0.5*L)

#-----------------------------------------------------------------
# EXTRAPOLATION CHECK

def x_at_percent(p, k, x0):
    return x0 + (1.0 / k) * np.log(p / (1.0 - p))

p_low  = 0.01   # 1% of near low
p_high = 0.99   # 99% of near high

x_low  = x_at_percent(p_low,  k, x0)
x_high = x_at_percent(p_high, k, x0)

y_low_fit  = sigmoid(x_low,  *params)
y_high_fit = sigmoid(x_high, *params)

print(f"x@{p_low*100:.0f}%  = {x_low:.4f} V, y = {y_low_fit:.6e} A")
print(f"x@{p_high*100:.0f}% = {x_high:.4f} V, y = {y_high_fit:.6e} A")

# plot
xx2 = np.linspace(-2.0, 2.0, 2000)
yy2 = sigmoid(xx2, *params)

plt.figure()
plt.plot(xx2, yy2, label="sigmoid fit")
plt.scatter(x, y, s=12, alpha=0.7, label="data")

# mark extrapolated points
plt.axvline(x_low,  linestyle=":", linewidth=1.5, label=f"x@1% = {x_low:.3f} V")
plt.axvline(x_high, linestyle=":", linewidth=1.5, label=f"x@99% = {x_high:.3f} V")
plt.scatter([x_low, x_high], [y_low_fit, y_high_fit], s=60)

plt.grid(True)
plt.legend()
plt.xlabel("Vg (V)")
plt.ylabel("Id (A)")
plt.title("Sigmoid extrapolation check")
plt.show()





'''

    L is responsible for scaling the output range from [0,1] to [0,L]
    b adds bias to the output and changes its range from [0,L] to [b,L+b]
    k is responsible for scaling the input, which remains in (-inf,inf)
    x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].

'''