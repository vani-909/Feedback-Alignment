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

name = 'Pulse_train'
DataName = './EqRAM/Data/' + name

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
    uzi = serial.Serial('COM12', baudrate=9600, timeout=5)
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

# Ensure all relays are OFF before starting experiment
time.sleep(2)
for i in range(1, 7):
    OFF(i)
    time.sleep(0.01)


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
    
    def fast_flush(self):
        self.inst.flush(visa.constants.BufferOperation.discard_read_buffer)
        self.write("errorqueue.clear()")
        self.write("dataqueue.clear()")

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

    def measure_v(self, ch) -> float:
        self.write(f"print({ch}.measure.v())")
        return float(self.read())
    
    def measure_all(self, ch_read, ch_prog):
        self.write(f'print(string.format("%e,%e,%e",{ch_read}.measure.i(),{ch_prog}.measure.i(),{ch_read}.measure.v()))')
        a = self.read().strip().split(",", 2)
        return float(a[0]), float(a[1]), float(a[2])

    def close(self):
        for ch in ("smua", "smub"):
            self.apply_voltage(ch, 0.0)
        self._drain_output()
        self.inst.close()

# -----------------------------------------------------------------------------
# Measurement parameters 

A = 'smua'                   # readout channel (drain)
B = 'smub'                   # programming channel (gate)

Vpos       = 0.7           
Vneg       = -0.7           
pulses     = 50              
cycles     = 2              

BACKSWEEP  = True            # if True: run Vpos block then Vneg block

Vread      = -0.10            # read bias on A

OnTime     = 0.5             # pulse width
OffTime    = 0.5             # delay

measPeriod = 0.10 

# -----------------------------------------------------------------------------
# Live plot

plt.ion()
fig2, (axVg, axId, axIg, axVd) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
axVg.set_ylabel('Vg (V)')
axId.set_ylabel('Id (A)')
axIg.set_ylabel('Ig (A)')
axVd.set_ylabel('Vd (V)')
axVd.set_xlabel('Time (s)')
for ax in (axVg, axId, axIg, axVd):
    ax.grid(True)

la_Vg, = axVg.plot([], [], lw=1.8, drawstyle='steps-post', label='Vg')
la_Id, = axId.plot([], [], lw=1.8, label='Id')
la_Ig, = axIg.plot([], [], lw=1.4, label='Ig')
la_Vd, = axVd.plot([], [], lw=1.4, label='Vd')
axVg.legend(loc='best'); axId.legend(loc='best'); axIg.legend(loc='best'); axVd.legend(loc='best')

t_all, Vg_all, Id_all, Ig_all, Vd_all = [], [], [], [], []

# -----------------------------------------------------------------------------
# Connect SMU and run pulse-train

smu = _SMU2602A("TCPIP0::169.254.0.1::5025::SOCKET")

smu.write("smua.source.output = smua.OUTPUT_ON")
smu.write("smub.source.output = smub.OUTPUT_ON")

smu.apply_voltage(A, Vread)
smu.apply_voltage(B, 0.0)
smu._drain_output()

# Pulse sequence 
seq = [Vpos] * int(pulses)
if BACKSWEEP:
    seq += [Vneg] * int(pulses)

print("Starting pulse-train...")

t0 = time.time()

try:
    for cyc in range(int(cycles)):
        for pulse in seq:

            ON(4)
            smu.apply_voltage(B, pulse)

            n_on = max(1, int(round(OnTime / measPeriod)))
            for _ in range(n_on):
                time.sleep(measPeriod)
                ts_m = time.time() - t0

                iA, iB, vA = smu.measure_all(A, B)    # Id, Ig, Vd
                vG = pulse                            # Vg

                t_all.append(ts_m)
                Vg_all.append(vG)
                Id_all.append(iA)
                Ig_all.append(iB)
                Vd_all.append(vA)

                # refresh plot
                la_Vg.set_data(t_all, Vg_all)
                la_Id.set_data(t_all, Id_all)
                la_Ig.set_data(t_all, Ig_all)
                la_Vd.set_data(t_all, Vd_all)
                axVg.relim(); axVg.autoscale_view()
                axId.relim(); axId.autoscale_view()
                axIg.relim(); axIg.autoscale_view()
                axVd.relim(); axVd.autoscale_view()
                plt.pause(0.001)
            smu.fast_flush()

            OFF(4)
            smu.apply_voltage(B, 0.0)

            n_off = max(1, int(round(OffTime / measPeriod)))
            for _ in range(n_off):
                time.sleep(measPeriod)
                ts_m = time.time() - t0

                iA, iB, vA = smu.measure_all(A, B)    # Id, Ig, Vd
                vG = 0.0                              # gate is off

                t_all.append(ts_m)
                Vg_all.append(vG)
                Id_all.append(iA)
                Ig_all.append(iB)
                Vd_all.append(vA)

                la_Vg.set_data(t_all, Vg_all)
                la_Id.set_data(t_all, Id_all)
                la_Ig.set_data(t_all, Ig_all)
                la_Vd.set_data(t_all, Vd_all)
                axVg.relim(); axVg.autoscale_view()
                axId.relim(); axId.autoscale_view()
                axIg.relim(); axIg.autoscale_view()
                axVd.relim(); axVd.autoscale_view()
                plt.pause(0.001)
            smu.fast_flush()
        smu._drain_output()

finally:
    # safe state
    smu.close()
    OFF(4)
    plt.ioff()
    print("Pulse-train complete.")

# -----------------------------------------------------------------------------
# Save CSVs and plot

data = os.path.join(dirName, 'pulse_train.csv')
with open(data, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time (s)', 'Vg (V)', 'Id (A)', 'Ig (A)', 'Vd (V)'])
    for t, vg, ida, iga, vd in zip(t_all, Vg_all, Id_all, Ig_all, Vd_all):
        w.writerow([float(t), float(vg), float(ida), float(iga), float(vd)])

fig2.tight_layout()
fig2.savefig(os.path.join(dirName, 'pulse_train.png'), dpi=300)

# -----------------------------------------------------------------------------
# Save report

report_path = os.path.join(dirName, 'Report_pulse_train.txt')
with open(report_path, 'w') as f:
    f.write("=== Pulse-Train Programming (B) + Readout (A) ===\n")
    f.write(f"Pulses           : {pulses}\n")
    f.write(f"Cycles           : {cycles}\n")
    f.write(f"Backsweep        : {BACKSWEEP}\n")
    f.write(f"Vpos / Vneg      : {Vpos:+.3f} V / {Vneg:+.3f} V\n")
    f.write(f"Read bias (A)    : {Vread:.3f} V\n")
    f.write(f"ON / OFF time    : {OnTime:.3f} s / {OffTime:.3f} s\n")
    f.write(f"measPeriod       : {measPeriod:.3f} s\n")
    f.write(f"Samples (all)    : {len(t_all)}\n")

print("Saved:")
print(f"  {data}  (Vg, Id, Ig, Vd vs time)")
print(f"  {report_path}")
print(f"Folder: {dirName}")
