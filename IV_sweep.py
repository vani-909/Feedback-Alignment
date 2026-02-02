# -----------------------------------------------------------------------------
# Vani R
# Eindhoven University of Technology
# Date: 4 Nov 2025
# -----------------------------------------------------------------------------

import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa
import serial 
import serial.tools.list_ports

# -----------------------------------------------------------------------------
# Directory setup

name = 'IV_sweep'
DataName = './Data/' + name

try:
    os.mkdir(DataName)
    print(f"Directory '{DataName}' created successfully.")
except FileExistsError:
    print(f"Directory '{DataName}' already exists.")

dirName = DataName + '/' + str(len(os.listdir(DataName)) + 1).zfill(3)
os.mkdir(dirName)


# -----------------------------------------------------------------------------
# UZI = Arduino

def connect_uzi():
    uzi = serial.Serial('COM12', baudrate=9600, timeout=5) # Check COM port in Device Manager 
    print("Connected to UZI at COM12")
    time.sleep(2)   
    return uzi

def ON(device_id):
    uzi.write(f'OutP{device_id:02d}H\n'.encode('utf-8'))
    time.sleep(0.01)

def OFF(device_id):
    uzi.write(f'OutP{device_id:02d}L\n'.encode('utf-8'))
    time.sleep(0.01)

uzi = connect_uzi()

# -----------------------------------------------------------------------------
# Keithley 2602A setup 

class _SMU2602A:
    def __init__(self, resource="TCPIP0::169.254.0.1::5025::SOCKET"):
        rm = visa.ResourceManager('@py')
        self.inst = rm.open_resource(resource, timeout=8000)  
        self.inst.read_termination = '\n'
        self.inst.write_termination = '\n'
        self.reset()
        self._drain_output()

        # set up both channels
        for ch in ("smua", "smub"):
            self.write(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")   # or DCAMPS
            self.write(f"{ch}.source.levelv = 0.0")
            self.write(f"{ch}.measure.autorangei = {ch}.AUTORANGE_ON")
            # self.write(f"{ch}.measure.rangei = 1e-4")  # Use if autorange = off
            self.write(f"{ch}.measure.nplc = 1")      

    def _drain_output(self):
        try:            
            self.write("errorqueue.clear()")      
            self.write("dataqueue.clear()")                  
            self.write('print("END_OF_BUFFER")')  
            buf = ""
            while "END_OF_BUFFER" not in buf:
                buf = self.read()
        except Exception:
            pass

    def write(self, s): 
        self.inst.write(s)

    def read(self):     
        return self.inst.read().strip()
    
    def reset(self):    
        self.write("reset()")

    def apply_voltage(self, ch, V):
        self.write(f"{ch}.source.levelv = {float(V)}")

    def measure_i(self, ch) -> float:
        self.write(f"print({ch}.measure.i())")
        return float(self.read())
    
    def measure_r(self, ch):
        self.write(f"print({ch}.measure.r())")
        return float(self.read())

    def close(self):
        for ch in ("smua", "smub"):
            self.apply_voltage(ch, 0.0)
        self._drain_output()
        self.inst.close()


# -----------------------------------------------------------------------------
# Measurement parameters

Vgstart   = -0.8     
Vgend     = 0.8   
Vgstep    = 0.02    

ScanRate  = 0.10     
Vd        = -0.10   
ncycles   = 3
channel   = 'smub'   # Gate

MeasurementDelay = abs(Vgstep / ScanRate)    # 0.02/0.1 = 0.2 s 

read_bias = 0.10     # V for ON/OFF ratio

# 1 sweep
_step = -abs(Vgstep) if Vgend < Vgstart else abs(Vgstep)
Vg_forward = np.arange(Vgstart, Vgend + _step*0.5, _step) 
Vg_back = Vg_forward[-2::-1]                           
Vg = np.concatenate([Vg_forward, Vg_back])

# Repeat for ncycles
Vsweep = np.tile(Vg, ncycles)

# Transfer curve
plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Vg (V)')
ax.set_ylabel('Id (A)')
ax.set_title('Transfer Curve')
ax.grid(True)
V_live, I_live = [], []


# -----------------------------------------------------------------------------
# Connect SMU and run sweep

smu = _SMU2602A("TCPIP0::169.254.0.1::5025::SOCKET")

drain = 'smua'   # fixed Vd, measure Id
gate  = 'smub'   # swept Vg

smu.write("smua.source.output = smua.OUTPUT_ON")   # drain channel ON
smu.write("smub.source.output = smub.OUTPUT_ON")   # gate channel ON

# Apply drain bias and zero the gate
smu.apply_voltage(drain, Vd)     
smu.apply_voltage(gate, 0.0)

data = []
print("Starting transfer sweep...")

ON(4)

time.sleep(2)


try:
    for v in Vsweep:
        smu.apply_voltage(gate, float(v))       # sweep Vg on smub
        time.sleep(MeasurementDelay)            
        Id = smu.measure_i(drain)               # measure Id on smua
        data.append((v, Id))

        # live plot (Vg vs Id)
        V_live.append(v)
        I_live.append(Id)
        line.set_data(V_live, I_live)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

        smu._drain_output()

finally:
    # safe state    
    smu.close()
    OFF(4)
    plt.ioff()
    print("Sweep complete.")


# -----------------------------------------------------------------------------
# Save and analyse data 

data = np.array(data, dtype=float)
Vg = data[:, 0]
Id = data[:, 1]

# Per-point conductance 
G = np.zeros_like(Id)
nz = np.abs(Vg) > 1e-6
G[nz] = Id[nz] / Vg[nz]

# Infer points per cycle from Vsweep and ncycles
points_per_cycle = len(Vsweep) // ncycles if ncycles > 0 else len(Vsweep)
cycle_idx = np.repeat(np.arange(ncycles), points_per_cycle)[:len(Vg)]
idx_in_cycle = np.arange(len(Vg)) % points_per_cycle

# Save CSV 
csv_file = os.path.join(dirName, 'transfer_Vg_Id.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Vg (V)', 'Id (A)', 'G (S)', 'cycle', 'idx_in_cycle'])
    for vg, id_, g_, c_, k_ in zip(Vg, Id, G, cycle_idx, idx_in_cycle):
        writer.writerow([vg, id_, g_, int(c_), int(k_)])
print(f"Saved data to {csv_file}")

# Compute ON/OFF ratio at ±read_bias
def nearest_idx(arr, val):
    return int(np.abs(arr - val).argmin())

idx_on  = nearest_idx(Vg,  +read_bias)
idx_off = nearest_idx(Vg,  -read_bias)
I_on  = abs(Id[idx_on])
I_off = abs(Id[idx_off])
ratio = I_on / (I_off + 1e-12)

# -----------------------------------------------------------------------------
# Plot and save figures

# 1) Full transfer curve 
plt.figure(figsize=(6, 4))
plt.plot(Vg, Id, lw=1, alpha=0.35)
cmap = plt.cm.viridis(np.linspace(0, 1, max(ncycles, 1)))
for c in range(ncycles):
    s = c * points_per_cycle
    e = min((c + 1) * points_per_cycle, len(Vg))
    if s < e:
        plt.plot(Vg[s:e], Id[s:e], lw=1.8, label=f'{c}', color=cmap[c])
plt.xlabel('Vg (V)')
plt.ylabel('Id (A)')
plt.title('Transfer Curve: Id vs Vg (cycles)')
plt.grid(True)
if ncycles > 1:
    leg = plt.legend(title='Cycle', loc='best', frameon=True)
    leg._legend_box.align = "left"
plt.tight_layout()
plt.savefig(os.path.join(dirName, 'Transfer_Vg_Id.png'), dpi=300)
plt.close()

# 2) Conductance vs Vg 
plt.figure(figsize=(6, 4))
plt.plot(Vg[nz], G[nz], lw=1.8)
plt.xlabel('Vg (V)')
plt.ylabel('G = Id/Vg (S)')
plt.title('Conductance vs Vg')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dirName, 'Conductance_vs_Vg.png'), dpi=300)
plt.close()

# Add Ig vs Vg plot

# -----------------------------------------------------------------------------
# Save report

report_path = os.path.join(dirName, 'Report_transfer.txt')
with open(report_path, 'w') as f:
    f.write("=== Transfer Curve (Id vs Vg) ===\n")
    f.write(f"Vg range: {Vg.min():.3f} to {Vg.max():.3f} V, step ~{Vgstep:.3f} V\n")
    f.write(f"Vd bias : {Vd:.3f} V (on smua)\n")
    f.write(f"Cycles  : {ncycles}  | points/cycle: {points_per_cycle}\n")
    f.write(f"MeasurementDelay (s/step): {MeasurementDelay:.3f}\n")
    f.write(f"Read bias (±): {read_bias:.3f} V\n")
    f.write(f"I(ON)  @ +{read_bias:.2f} V = {I_on:.4e} A (idx {idx_on})\n")
    f.write(f"I(OFF) @ -{read_bias:.2f} V = {I_off:.4e} A (idx {idx_off})\n")
    f.write(f"ON/OFF ratio = {ratio:.2f}\n")
print(f"ON/OFF ratio @ {read_bias:.2f} V: {ratio:.2f}")
print(f"Results saved in: {dirName}")
