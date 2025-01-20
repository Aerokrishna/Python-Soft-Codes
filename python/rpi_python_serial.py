import serial
import time

# Configure the serial connection
ser = serial.Serial('/dev/ttyACM0', 9600)  # Adjust port as needed (e.g., /dev/ttyS0 for GPIO)
time.sleep(2)  # Wait for Arduino to reset

# Data to send
data = [123, 456, 789]  # Example data

try:
    while True:
        # Convert data to a comma-separated string
        data_str = ",".join(map(str, data)) + "\n"
        
        # Send data over serial
        ser.write(data_str.encode('utf-8'))
        print(f"Sent: {data_str.strip()}")

        # Wait for a response (optional)
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino replied: {response}")

        time.sleep(1)  # Send every second

except KeyboardInterrupt:
    print("Exiting.")
finally:
    ser.close()
