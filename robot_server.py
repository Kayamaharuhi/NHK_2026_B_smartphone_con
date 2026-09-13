import asyncio
import json
import os
import time
import urllib.parse
import serial
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="明石布 ロボットコントローラー API")

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
BAUD_RATE = int(os.environ.get("BAUD_RATE", 115200))

serial_lock = threading.Lock()
ser_conn = None

telemetry_data = {
    "yaw": 0.0,
    "con_alive": False,
    "roller_mode": False,
    "steer_currents": [0, 0, 0, 0],
    "can_devices": {},
    "timestamp": 0.0,
    "serial_connected": False,
    "cloth": None,  
}

connected_clients = set()
clients_lock = threading.Lock()

def safe_send_serial(char_to_send: str) -> bool:
    global ser_conn
    with serial_lock:
        if ser_conn is not None and getattr(ser_conn, "is_open", False):
            try:
                ser_conn.write(char_to_send.encode("utf-8"))
                ser_conn.flush()
                print(f"[Serial TX] Sent 1 char: '{char_to_send}'")
                return True
            except Exception as e:
                print(f"[Serial TX Error] Failed to send '{char_to_send}': {e}")
                return False
        else:
            print(f"[Serial TX Warn] Cannot send '{char_to_send}': Serial connection is not open")
            return False

def serial_reader_thread():
    global telemetry_data, ser_conn

    while True:
        try:
            print(f"[Serial] Connecting to {SERIAL_PORT} (Baud: {BAUD_RATE})...")
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) as ser:
                with serial_lock:
                    ser_conn = ser
                telemetry_data["serial_connected"] = True
                print("[Serial] Connected successfully!")

                while True:
                    with serial_lock:
                        if ser.in_waiting > 128:
                            ser.reset_input_buffer()

                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    try:
                        parts = line.split(',')
                        header = parts[0].strip()

                        if header == "DATA" and len(parts) >= 8:
                            telemetry_data["yaw"] = float(parts[1]) / 1000.0
                            telemetry_data["con_alive"] = (parts[2].strip() == "1")
                            telemetry_data["roller_mode"] = (parts[3].strip() == "1")
                            telemetry_data["steer_currents"] = [
                                int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
                            ]
                            telemetry_data["timestamp"] = time.time()
                        elif header == "CAN" and len(parts) >= 3:
                            can_id = parts[1].strip()
                            telemetry_data["can_devices"][f"0x{can_id}"] = [p.strip() for p in parts[2:]]
                            telemetry_data["timestamp"] = time.time()
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"[Serial] Connection error: {e}. Reconnecting in 1s...")
            with serial_lock:
                ser_conn = None
            telemetry_data["serial_connected"] = False
            time.sleep(1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with clients_lock:
        connected_clients.add(websocket)
    print(f"[WS] Client connected. Total active clients: {len(connected_clients)}")

    send_task = None
    try:
        async def send_loop():
            while True:
                try:
                    await websocket.send_text(json.dumps(telemetry_data))
                except Exception:
                    break
                await asyncio.sleep(0.1)

        send_task = asyncio.create_task(send_loop())

        while True:
            raw_msg = await websocket.receive_text()
            if not raw_msg:
                continue

            if raw_msg.startswith("{"):
                try:
                    payload = json.loads(raw_msg)
                    msg_type = payload.get("type")
                    if msg_type == "cloth_alert":
                        cloth_data = payload.get("data")
                        telemetry_data["cloth"] = cloth_data
                        telemetry_data["timestamp"] = time.time()
                        await broadcast_telemetry()
                except Exception as e:
                    print(f"[WS JSON Parse Error] {e}")
            else:
                one_char = raw_msg[0]
                safe_send_serial(one_char)

    except WebSocketDisconnect:
        print("[WS] Client disconnected cleanly")
    except Exception as e:
        print(f"[WS Error] {e}")
    finally:
        with clients_lock:
            connected_clients.discard(websocket)
        if send_task:
            send_task.cancel()

async def broadcast_telemetry():
    msg = json.dumps(telemetry_data)
    with clients_lock:
        clients_list = list(connected_clients)
    for ws in clients_list:
        try:
            await ws.send_text(msg)
        except Exception:
            pass

@app.post("/api/cloth_alert")
async def post_cloth_alert(request: Request):
    try:
        data = await request.json()
        telemetry_data["cloth"] = data
        telemetry_data["timestamp"] = time.time()
        await broadcast_telemetry()
        is_ng = data.get("is_ng", False)
        status = data.get("status", "UNKNOWN")
        print(f"[Cloth API] {' ALERT' if is_ng else 'INFO'}: {status}")
        return {"status": "ok", "broadcasted": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def get_index():
    found_candidates = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for candidate in ["robot_index.html", "index.html"]:
        index_path = os.path.join(base_dir, candidate)
        if os.path.exists(index_path) and os.path.isfile(index_path):
            mtime = os.path.getmtime(index_path)
            found_candidates.append((mtime, index_path, candidate))

    if found_candidates:
        # 更新日時(mtime)が最も新しいファイルを自動選択
        found_candidates.sort(key=lambda x: x[0], reverse=True)
        newest_mtime, newest_path, newest_name = found_candidates[0]
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest_mtime))
        print(f"[HTTP GET /] 配信中: {newest_name} (最終更新: {mtime_str})")
        
        return FileResponse(
            newest_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return Response(content="<h1>robot_index.html or index.html not found</h1>", media_type="text/html")

@app.get("/{filename:path}")
async def get_static_files(filename: str):
    decoded_filename = urllib.parse.unquote(filename)
    
    for name in [decoded_filename, filename]:
        file_path = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
    


if __name__ == "__main__":
    threading.Thread(target=serial_reader_thread, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
