import socket
import subprocess

HOST = '0.0.0.0'
PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("📡 Waiting for laptop connection on port 8000...")

conn, addr = server_socket.accept()
print(f"✅ Connected from {addr}")

try:
    # Stream H.264 video using rpicam-vid
    cmd = [
        "rpicam-vid",
        "-t", "0",              # run indefinitely
        "--width", "640",
        "--height", "480",
        "--framerate", "15",
        "--codec", "h264",
        "-o", "-",              # output to stdout
        "-n"                    # no preview window
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            data = proc.stdout.read(1024)
            if not data:
                break
            conn.sendall(data)

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")
finally:
    conn.close()
    server_socket.close()
    print("Server closed.")

