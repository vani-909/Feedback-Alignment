# The on-off ratio calculation in main IV script is incorrect. Use this script to calculate correctly from measurement data

import pandas as pd
import numpy as np

voltage_col = "Vg (V)"          
current_col = "Id (A)"         

df = pd.read_csv('EqRAM/Data/IV_sweep/030/transfer_Vg_Id.csv')

Vg = df[voltage_col].to_numpy(dtype=float)
Id = df[current_col].to_numpy(dtype=float)

Iabs = np.abs(Id)

# ON state = maximum current magnitude
# OFF state = minimum current magnitude
idx_on = int(np.argmax(Iabs))
idx_off = int(np.argmin(Iabs))

I_on = Iabs[idx_on]
I_off = Iabs[idx_off]
ratio = I_on / (I_off + 1e-12)

print("=== ON/OFF ratio from CSV ===")
print(f"ON  index = {idx_on}")
print(f"OFF index = {idx_off}")
print(f"Vg(ON)  = {Vg[idx_on]:.6f} V")
print(f"Vg(OFF) = {Vg[idx_off]:.6f} V")
print(f"I(ON)   = {I_on:.6e} A")
print(f"I(OFF)  = {I_off:.6e} A")
print(f"ON/OFF ratio = {ratio:.4f}")