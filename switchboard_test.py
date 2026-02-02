import time
import serial
import serial.tools.list_ports

def connect_uzi():
    uzi = serial.Serial('COM12', baudrate=9600, timeout=5)
    print("Connected to UZI at COM12")
    time.sleep(2)   
    return uzi

def ON(device_id):
    uzi.write(f'OutP{device_id:02d}H\n'.encode('utf-8'))
    time.sleep(1)

def OFF(device_id):
    uzi.write(f'OutP{device_id:02d}L\n'.encode('utf-8'))
    time.sleep(1)

uzi = connect_uzi()

for i in range(5):
    print("on")
    ON(4)
    print("off")
    OFF(4)