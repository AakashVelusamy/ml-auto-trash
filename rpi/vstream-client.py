import cv2

# Replace with your Pi's IP address
RPI_IP = "192.168.0.175"
PORT = 8000

url = f"tcp://{RPI_IP}:{PORT}"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Failed to connect to stream")
    exit()

print("✅ Connected to Raspberry Pi stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame lost")
        break

    cv2.imshow("Raspberry Pi Live Stream", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
