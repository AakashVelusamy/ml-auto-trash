import subprocess
import numpy as np
import cv2
import threading
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from keras.models import load_model
import sys
import os

# ----------------------------
# Logging setup
# ----------------------------
LOG_PATH = "/home/pi/logs.txt"
def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log("\n========== Starting classifier.py ==========")

# ----------------------------
# LCD Setup (I2C 16x2)
# ----------------------------
lcd = CharLCD('PCF8574', 0x27)
lcd.clear()

# ----------------------------
# Custom Characters (5x8)
# ----------------------------
pacman_open = [
    0b01110, 0b11111, 0b11000, 0b10000,
    0b11000, 0b11111, 0b01110, 0b00000
]
pacman_closed = [
    0b01110, 0b11111, 0b11111, 0b11111,
    0b11111, 0b11111, 0b01110, 0b00000
]
water_bottle = [
    0b00100, 0b01110, 0b01110, 0b01110,
    0b01110, 0b01110, 0b01110, 0b00000
]
banana = [
    0b00100, 0b01100, 0b01100, 0b01100,
    0b01110, 0b00110, 0b00010, 0b00000
]
paperball = [
    0b00000, 0b01110, 0b10101, 0b01010,
    0b10101, 0b01110, 0b00000, 0b00000
]
pen = [
    0b00000, 0b01100, 0b01100, 0b00100,
    0b00100, 0b00100, 0b00100, 0b00000
]
dustbin = [
    0b00000, 0b11111, 0b11011, 0b10101,
    0b11011, 0b10101, 0b01110, 0b00000
]

lcd.create_char(0, pacman_open)
lcd.create_char(1, pacman_closed)
lcd.create_char(2, water_bottle)
lcd.create_char(3, banana)
lcd.create_char(4, paperball)
lcd.create_char(5, pen)
lcd.create_char(6, dustbin)

# ----------------------------
# Ultrasonic Sensor Setup
# ----------------------------
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    elapsed = stop_time - start_time
    distance = (elapsed * 34300) / 2
    return distance

# ----------------------------
# Animation thread
# ----------------------------
animation_running = False
animation_status_text = ""

def pacman_loop():
    """Loop Pac-Man eating across LCD until animation_running stops."""
    items = [2, 3, 4, 5, 6]
    try:
        while animation_running:
            for pos in range(0, 16):
                if not animation_running:
                    break
                lcd.clear()
                for i, it in enumerate(items):
                    food_pos = 3 + (i * 3)
                    if food_pos > pos:
                        lcd.cursor_pos = (0, food_pos)
                        lcd.write_string(chr(it))
                lcd.cursor_pos = (0, pos)
                lcd.write_string(chr(0 if pos % 2 == 0 else 1))
                lcd.cursor_pos = (1, 0)
                lcd.write_string(animation_status_text[:16].ljust(16))
                time.sleep(0.35)
            time.sleep(0.7)
    except Exception as e:
        log(f"Pacman loop error: {e}")

# ----------------------------
# Startup Display
# ----------------------------
lcd.clear()
lcd.cursor_pos = (0, 0)
lcd.write_string("kuppa thotti " + chr(6))
lcd.cursor_pos = (1, 0)
lcd.write_string("activating...")
time.sleep(2)

# ----------------------------
# Load Model
# ----------------------------
try:
    model_path = "/home/pi/model/model.keras"
    model = load_model(model_path)
    CLASSES = ["paper_ball", "banana_peel", "plastic_bottle", "pen"]
    INPUT_SIZE = (128, 128)
    log("Model loaded successfully.")
except Exception as e:
    log(f"Error loading model: {e}")
    lcd.clear()
    lcd.write_string("Model load fail")
    time.sleep(5)
    sys.exit(1)

lcd.clear()
lcd.cursor_pos = (0, 0)
lcd.write_string("kuppa thotti " + chr(6))
lcd.cursor_pos = (1, 0)
lcd.write_string("ready!")

output_path = "/home/pi/tmp/captured.jpg"

# ----------------------------
# Main Loop
# ----------------------------
try:
    while True:
        distance = get_distance()
        if distance < 15:
            log(f"Object detected at {distance:.2f} cm")

            animation_status_text = "capturing..."
            animation_running = True
            anim_thread = threading.Thread(target=pacman_loop, daemon=True)
            anim_thread.start()

            subprocess.run(["rpicam-still", "-o", output_path, "-t", "1000", "-n"], check=True)
            animation_status_text = "predicting..."

            img = cv2.imread(output_path)
            if img is None:
                log("Failed to read captured image.")
                animation_running = False
                anim_thread.join()
                lcd.clear()
                lcd.write_string("capture failed")
                time.sleep(2)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
            img_array = np.expand_dims(img, axis=0)

            preds = model.predict(img_array)
            pred_idx = int(np.argmax(preds))
            pred_class = CLASSES[pred_idx]
            confidence = float(np.max(preds)) * 100.0
            log(f"Predicted: {pred_class} ({confidence:.2f}%)")

            animation_running = False
            anim_thread.join()

            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string("kuppa identified")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(chr(6))
            time.sleep(2)

            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string(pred_class[:16])
            lcd.cursor_pos = (1, 0)
            lcd.write_string(f"{confidence:.2f}%")
            time.sleep(10)

            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string("kuppa thotti " + chr(6))
            lcd.cursor_pos = (1, 0)
            lcd.write_string("ready!")

            time.sleep(2)
        else:
            time.sleep(0.5)

except KeyboardInterrupt:
    log("Stopped by user (KeyboardInterrupt).")
finally:
    animation_running = False
    lcd.clear()
    lcd.write_string("goodbye!")
    time.sleep(7)
    lcd.clear()
    GPIO.cleanup()
    log("GPIO cleaned up, exiting gracefully.")
