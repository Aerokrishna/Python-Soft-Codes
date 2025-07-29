import serial
import time
import tkinter as tk
from tkinter import ttk
import threading

# ============ Serial Setup ============
PORT = '/dev/ttyACM0'  # Update for your system
BAUD = 115200

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Connected to Pico")
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    ser = None

# ============ GUI Setup ============
root = tk.Tk()
root.title("PID Tuner")
root.geometry("400x300")  # Increased window size

# Style
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12))

# Slider value labels
kp_val = tk.StringVar()
ki_val = tk.StringVar()
kd_val = tk.StringVar()
kp_val.set("1.00")
ki_val.set("0.10")
kd_val.set("0.05")

# Sliders and labels
def update_kp(val):
    kp_val.set(f"{float(val):.2f}")

def update_ki(val):
    ki_val.set(f"{float(val):.2f}")

def update_kd(val):
    kd_val.set(f"{float(val):.2f}")

ttk.Label(root, text="Kp").grid(row=0, column=0, padx=10, pady=10, sticky="w")
kp_slider = ttk.Scale(root, from_=0.0, to=10.0, orient='horizontal', command=update_kp)
kp_slider.set(1.0)
kp_slider.grid(row=0, column=1, padx=10)
ttk.Label(root, textvariable=kp_val).grid(row=0, column=2)

ttk.Label(root, text="Ki").grid(row=1, column=0, padx=10, pady=10, sticky="w")
ki_slider = ttk.Scale(root, from_=0.0, to=1.0, orient='horizontal', command=update_ki)
ki_slider.set(0.1)
ki_slider.grid(row=1, column=1, padx=10)
ttk.Label(root, textvariable=ki_val).grid(row=1, column=2)

ttk.Label(root, text="Kd").grid(row=2, column=0, padx=10, pady=10, sticky="w")
kd_slider = ttk.Scale(root, from_=0.0, to=5.0, orient='horizontal', command=update_kd)
kd_slider.set(0.05)
kd_slider.grid(row=2, column=1, padx=10)
ttk.Label(root, textvariable=kd_val).grid(row=2, column=2)

# Send PID values
def send_pid_values():
    if ser is None or not ser.is_open:
        print("Serial port not open.")
        return

    kp = kp_slider.get()
    ki = ki_slider.get()
    kd = kd_slider.get()

    data = f"{kp:.4f} {ki:.4f} {kd:.4f}\n"
    ser.write(data.encode())
    print(f"Sent to Pico: {data.strip()}")

ttk.Button(root, text="Send", command=send_pid_values).grid(row=3, column=0, columnspan=3, pady=20)

# ============ Background Serial Listener ============
def serial_listener():
    while ser and ser.is_open:
        try:
            if ser.in_waiting:
                msg = ser.readline().decode('utf-8').strip()
                if msg:
                    print(f"Pico says: {msg}")
        except Exception as e:
            print(f"Error in serial thread: {e}")
            break

listener_thread = threading.Thread(target=serial_listener, daemon=True)
listener_thread.start()

# ============ Run GUI ============
root.mainloop()

# Close serial on exit
if ser and ser.is_open:
    ser.close()
    print("Serial connection closed.")
