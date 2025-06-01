import serial
import time

# Configure the serial port
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)  # Use '/dev/ttyAMA0' for older Raspberry Pi models
ser.flush()  # Clear the serial buffer

try:
    while True:
        # Send a message to the Arduino
        ser.write(b"Hello from Raspberry Pi\n")
        print("Sent: Hello from Raspberry Pi")

        # Wait for a response from the Arduino
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').rstrip()
            print(f"Received: {response}")

        # Wait for 1 second before sending the next message
        time.sleep(1)

except KeyboardInterrupt:
    print("Program terminated")

finally:
    ser.close()  # Close the serial connection