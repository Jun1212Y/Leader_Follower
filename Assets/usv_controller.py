import socket
import json
import math
import time
import threading
import struct
import cv2
import numpy as np

# =========================================================
# 1) 雙船 V 字隊形控制：UDP 設定
# =========================================================
UDP_IP = "127.0.0.1"

# 左護法 (Follower_Left)
PORT_LEFT_RX = 5066
PORT_LEFT_TX = 5065

# 右護法 (Follower_Right)
PORT_RIGHT_RX = 5068
PORT_RIGHT_TX = 5067

sock_left = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_left.bind((UDP_IP, PORT_LEFT_RX))
sock_left.setblocking(False)

sock_right = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_right.bind((UDP_IP, PORT_RIGHT_RX))
sock_right.setblocking(False)

# =========================================================
# 2) OpenCV 視覺：TCP 設定
# =========================================================
HOST = "0.0.0.0"
PORT = 9999
WINDOW_NAME = "OpenCV Wake Tracking"

# =========================================================
# 3) 隊形與控制參數
# =========================================================
OFFSET = 180
FORMATION_BACK_DIST = 20.0
FORMATION_SIDE_DIST = 15.0
SAFE_DIST = 5.0

DEADZONE = 8.0
KP_STEER = 0.02

# =========================================================
# 4) 視覺輔助控制參數（沿用原本邏輯，改為偵測尾流）
# =========================================================
ENABLE_VISION_ASSIST = True
VISION_CENTER_X_TOL = 0.25   # 尾流在畫面中央的容許誤差比例
VISION_WAKE_AREA_TH = 1500   # 尾流面積超過多少代表距離太近 (需根據畫面調整)
VISION_THROTTLE_SCALE = 0.35

# =========================================================
# 5) OpenCV 傳統視覺處理參數
# =========================================================
SHOW_WINDOW = True        # 顯示處理後的畫面
SHOW_OVERLAY_TEXT = True  # 顯示 FPS 等資訊
MIN_WAKE_AREA = 100       # 尾流的最小面積像素(過濾海面反光雜訊)

# =========================================================
# 6) 共享視覺狀態 (改為 Wake 狀態)
# =========================================================
vision_lock = threading.Lock()
vision_state = {
    "connected": False,
    "fps": 0.0,
    "wake_detected": False,
    "wake_bbox": None,
    "wake_area": 0.0,
    "wake_center_offset": None,
    "last_update": 0.0
}

# =========================================================
# 7) 最新影格共享區（關鍵：只保留最新一幀）
# =========================================================
frame_lock = threading.Lock()
latest_frame = None
latest_frame_id = 0
latest_frame_time = 0.0


# =========================================================
# 工具函式：TCP 接收固定長度
# =========================================================
def recv_exact(conn, size):
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


# =========================================================
# 執行緒 1：只負責收最新影格
# =========================================================
def tcp_frame_receiver_thread():
    global latest_frame, latest_frame_id, latest_frame_time, vision_state

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[TCP] Waiting for Unity connection on {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[TCP] Connected by {addr}")

    with vision_lock:
        vision_state["connected"] = True

    try:
        while True:
            header = recv_exact(conn, 12)
            if header is None:
                print("[TCP] Connection closed.")
                break

            width, height, data_len = struct.unpack("iii", header)

            jpg_bytes = recv_exact(conn, data_len)
            if jpg_bytes is None:
                print("[TCP] Failed to receive image bytes.")
                break

            img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 只保留最新一幀
            with frame_lock:
                latest_frame = frame
                latest_frame_id += 1
                latest_frame_time = time.time()

    except Exception as e:
        print(f"[TCP] Error: {e}")

    finally:
        with vision_lock:
            vision_state["connected"] = False
            vision_state["wake_detected"] = False

        try:
            conn.close()
        except:
            pass
        try:
            server.close()
        except:
            pass

        print("[TCP] Shutdown complete.")


# =========================================================
# 執行緒 2：OpenCV 傳統視覺處理 (找白色尾流)
# =========================================================
def cv_processing_thread():
    global latest_frame, latest_frame_id, latest_frame_time, vision_state

    print("[OpenCV] Visual processing thread started.")

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0
    last_processed_frame_id = -1

    try:
        while True:
            frame = None
            frame_id = -1
            frame_ts = 0.0

            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
                    frame_id = latest_frame_id
                    frame_ts = latest_frame_time

            if frame is None:
                time.sleep(0.001)
                continue

            # 沒新幀就不重複推
            if frame_id == last_processed_frame_id:
                time.sleep(0.001)
                continue

            last_processed_frame_id = frame_id
            
            h, w = frame.shape[:2]
            display_frame = frame.copy() if SHOW_WINDOW else None

            # --- 影像處理核心開始 ---
            # 1. 轉換色彩空間 (BGR to HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 2. 定義「白色」的閥值 (視 Unity 光線情況調整)
            # 這裡設定飽和度低 (白)、亮度高 (亮)
            lower_white = np.array([0, 0, 240])   
            upper_white = np.array([180, 50, 255]) 
            
            # 3. 提取白色區域的 Mask
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # ==========================================
            # 🌟 新增：設定 ROI (感興趣區域)，切除天空與自己船身
            # ==========================================
            # 1. 屏蔽天空：將畫面上半部約 45% 直接塗黑
            sky_crop = int(h * 0.5)
            mask[0:sky_crop, :] = 0
            
            # 2. 屏蔽自己船身：將畫面下半部約 25% 直接塗黑 (保留 0~75% 的區域)
            boat_crop = int(h * 0.75)
            mask[boat_crop:h, :] = 0
            
            # ==========================================
            # 🌟 新增：精細化形態學「去雜訊 + 橋接」組合拳
            # ==========================================
            
            # 步驟 1: 開運算 (MORPH_OPEN) —— 先擦除孤立的小反光雜點
            # 用一個很小的 3x3 正方形 Kernel
            kernel_open = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            
            # 步驟 2: 閉運算 (MORPH_CLOSE) —— 後橋接斷裂的尾流
            # 用一個「極度扁平」的 Kernel (高度 1，寬度 30)
            # 這把刷子只會黏左右，不會黏上下，完美過濾上下雜訊！
            kernel_close = np.ones((1, 30), np.uint8) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            # ==========================================
            
            # 4. 尋找輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            max_area = 0

            # 找出最大的白色區塊 (假設最大的白色區塊就是尾流)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area and area > MIN_WAKE_AREA:
                    max_area = area
                    best_cnt = cnt

            # --- 計算與更新狀態 ---
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 1.0 / dt

            with vision_lock:
                vision_state["fps"] = fps
                vision_state["last_update"] = current_time

                if best_cnt is not None:
                    # 計算特徵重心
                    M = cv2.moments(best_cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        center_offset = (cx - (w / 2.0)) / (w / 2.0)
                        x, y, bw, bh = cv2.boundingRect(best_cnt)

                        vision_state["wake_detected"] = True
                        vision_state["wake_bbox"] = (x, y, x+bw, y+bh)
                        vision_state["wake_area"] = max_area
                        vision_state["wake_center_offset"] = center_offset

                        if SHOW_WINDOW:
                            # 畫出綠色外框與紅色中心點
                            cv2.rectangle(display_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(display_frame, f"Area: {max_area:.0f}", (x, max(y - 10, 20)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    vision_state["wake_detected"] = False
                    vision_state["wake_bbox"] = None
                    vision_state["wake_area"] = 0.0
                    vision_state["wake_center_offset"] = None

            if SHOW_WINDOW and display_frame is not None:
                # 在左上角疊加黑白 Mask 預覽，方便調試閥值
                mask_small = cv2.resize(mask, (0, 0), fx=0.3, fy=0.3)
                mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                h_m, w_m = mask_color.shape[:2]
                display_frame[0:h_m, 0:w_m] = mask_color

                if SHOW_OVERLAY_TEXT:
                    delay_ms = int((current_time - frame_ts) * 1000.0)
                    cv2.putText(display_frame, f"CV FPS: {fps:.1f}", (w_m + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Delay: {delay_ms} ms", (w_m + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC 鍵離開
                    break

    except Exception as e:
        print(f"[OpenCV] Error: {e}")

    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("[OpenCV] Shutdown complete.")


# =========================================================
# 視覺輔助判斷：前方中央是否有尾流 (距離太近)
# =========================================================
def is_front_boat_danger():
    with vision_lock:
        if not vision_state["wake_detected"]:
            return False

        offset = vision_state["wake_center_offset"]
        area = vision_state["wake_area"]

        if offset is None:
            return False

        in_center = abs(offset) <= VISION_CENTER_X_TOL
        near_enough = area >= VISION_WAKE_AREA_TH

        return in_center and near_enough


# =========================================================
# 單艘船控制邏輯（完全不動，完美沿用你的心血）
# =========================================================
def process_boat(sock, tx_port, boat_name, is_left):
    latest_data = None
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            latest_data = data
        except BlockingIOError:
            break

    if latest_data is None:
        return None

    state = json.loads(latest_data.decode("utf-8"))

    # 若還沒抓到 leader
    if 'leader_x' not in state:
        sock.sendto(
            bytes(json.dumps({"throttle": 0.0, "steer": 0.0}), "utf-8"),
            (UDP_IP, tx_port)
        )
        return None

    # =====================================================
    # 計算 V 字隊形目標點
    # =====================================================
    leader_x = state['leader_x']
    leader_z = state['leader_z']

    actual_leader_yaw = state['leader_yaw'] + OFFSET
    leader_yaw_rad = math.radians(actual_leader_yaw)

    back_x = -math.sin(leader_yaw_rad) * FORMATION_BACK_DIST
    back_z = -math.cos(leader_yaw_rad) * FORMATION_BACK_DIST

    right_x = math.cos(leader_yaw_rad) * FORMATION_SIDE_DIST
    right_z = -math.sin(leader_yaw_rad) * FORMATION_SIDE_DIST

    if is_left:
        target_x = leader_x + back_x - right_x
        target_z = leader_z + back_z - right_z
    else:
        target_x = leader_x + back_x + right_x
        target_z = leader_z + back_z + right_z

    # =====================================================
    # 追蹤虛擬目標點
    # =====================================================
    dx = target_x - state['x']
    dz = target_z - state['z']
    dist = math.sqrt(dx**2 + dz**2)

    target_angle = math.degrees(math.atan2(dx, dz))
    my_angle = state['yaw'] + OFFSET

    diff = target_angle - my_angle
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360

    # 轉向
    if abs(diff) < DEADZONE:
        steer = 0.0
    else:
        steer = diff * KP_STEER

    steer = max(min(steer, 1.0), -1.0)

    # 基本油門
    dist_error = dist - SAFE_DIST
    if dist_error > 0:
        throttle = dist_error * 0.05
    else:
        throttle = 0.0

    throttle = max(min(throttle, 1.0), 0.0)

    # 大角度自動收油門
    if abs(diff) > 45:
        throttle = 0.0
    elif abs(diff) > 20:
        throttle = throttle * 0.3

    # =====================================================
    # 視覺輔助：若前方中央偵測到大面積尾流，強制保守限速
    # =====================================================
    vision_brake = False
    if ENABLE_VISION_ASSIST and is_front_boat_danger():
        throttle = throttle * VISION_THROTTLE_SCALE
        vision_brake = True

    # 發送指令
    msg = json.dumps({"throttle": throttle, "steer": steer})
    sock.sendto(bytes(msg, "utf-8"), (UDP_IP, tx_port))

    speed_mps = state.get('speed', 0.0)

    return {
        "dist": dist,
        "throttle": throttle,
        "diff": diff,
        "speed_knots": speed_mps * 1.94384,
        "vision_brake": vision_brake
    }


# =========================================================
# 主程式
# =========================================================
def main():
    print("=======================================")
    print("雙船 V字隊形 + OpenCV 尾流特徵即時版 啟動")
    print("=======================================")

    tcp_thread = threading.Thread(target=tcp_frame_receiver_thread, daemon=True)
    tcp_thread.start()

    cv_thread = threading.Thread(target=cv_processing_thread, daemon=True)
    cv_thread.start()

    last_print_time = time.time()

    while True:
        try:
            res_left = process_boat(sock_left, PORT_LEFT_TX, "Left", True)
            res_right = process_boat(sock_right, PORT_RIGHT_TX, "Right", False)

            current_time = time.time()
            if current_time - last_print_time > 0.1:
                print_str = ""

                if res_left:
                    print_str += (
                        f"[左護法] 距目標: {res_left['dist']:5.1f}m | "
                        f"油門: {res_left['throttle']:4.2f} | "
                        f"角差: {res_left['diff']:6.1f}"
                    )
                    if res_left["vision_brake"]:
                        print_str += " | 視覺限速:ON"

                if res_right:
                    if print_str != "":
                        print_str += " || "
                    print_str += (
                        f"[右護法] 距目標: {res_right['dist']:5.1f}m | "
                        f"油門: {res_right['throttle']:4.2f} | "
                        f"角差: {res_right['diff']:6.1f}"
                    )
                    if res_right["vision_brake"]:
                        print_str += " | 視覺限速:ON"

                with vision_lock:
                    vs = vision_state.copy()

                if print_str != "":
                    if vs["connected"]:
                        print_str += (
                            f" || [CV] wake={vs['wake_detected']} "
                            f"area={vs['wake_area']:>5.0f} "
                            f"fps={vs['fps']:.1f}"
                        )
                    else:
                        print_str += " || [CV] 未連線"

                    print(print_str)

                last_print_time = current_time

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[Main] 使用者中止程式")
            break
        except Exception as e:
            print(f"[Main] Error: {e}")
            time.sleep(0.01)


if __name__ == "__main__":
    main()