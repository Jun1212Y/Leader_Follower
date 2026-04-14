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
# FORMATION_BACK_DIST = 20.0
# FORMATION_SIDE_DIST = 15.0
POS_LEFT  = (20.0, -10.0)
POS_RIGHT = (-20.0, -10.0)
# SAFE_DIST = 5.0

# PID　Parameters
DEADZONE = 6.0
KP_STEER = 0.02
KI_STEER = 0.0
KD_STEER = 0.008

KP_THROTTLE = 0.30
KI_THROTTLE = 0.005
KD_THROTTLE = 0.0

HEADING_I_LIMIT = 30.0
DIST_I_LIMIT = 15.0
LEADER_VEL_ALPHA = 0.45

FORMATION_POS_GAIN = 1.2
RIGID_SPEED_MAX = 20.0
MIN_GUIDE_VEC_NORM = 1e-6
CATCHUP_DIST_TH = 12.0
CATCHUP_SPEED_GAIN = 0.36
CATCHUP_SPEED_MAX_BOOST = 4.0
CATCHUP_THROTTLE_GAIN = 0.04
CATCHUP_THROTTLE_MAX = 0.18

LEADER_STOP_SPEED_TH = 0.2
FORMATION_HOLD_DIST_TH = 2.0
# =========================================================
# 多智能體避障 (Inter-Agent APF) 參數
# =========================================================
K_ATT = 0.3           # 引力係數 (前往目標點的渴望程度)
K_REP_AGENT = 1200.0  # 隊友斥力係數 (設大一點，確保互相排斥的力道夠強)
R_COL = 20.0          # 互斥半徑 (當兩艘船距離小於 12 公尺時，開始產生斥力推開彼此)
K_REP_LEADER = 1600.0
R_LEADER_COL = 16.0
LOCAL_MIN_TOTAL_FORCE_TH = 0.35
LOCAL_MIN_TARGET_DIST_TH = 6.0
LOCAL_MIN_PROGRESS_TH = 0.15
LOCAL_MIN_STUCK_TIME = 0.8
ESCAPE_TANGENT_GAIN = 3.0

# =========================================================
# 4) 視覺輔助控制參數（沿用原本邏輯，改為偵測尾流）
# =========================================================
ENABLE_VISION_ASSIST = False
VISION_CENTER_X_TOL = 0.25   # 尾流在畫面中央的容許誤差比例
VISION_WAKE_AREA_TH = 1500   # 尾流面積超過多少代表距離太近 (需根據畫面調整)
VISION_THROTTLE_SCALE = 0.35

# =========================================================
# 5) OpenCV 傳統視覺處理參數
# =========================================================
SHOW_WINDOW = False        # 顯示處理後的畫面
SHOW_OVERLAY_TEXT = False  # 顯示 FPS 等資訊
MIN_WAKE_AREA = 100       # 尾流的最小面積像素(過濾海面反光雜訊)

#PID Dictionary
pid_state = {
    "Left": {
        "prev_time": None,
        "heading_integral": 0.0,
        "prev_heading_error": 0.0,
        "dist_integral": 0.0,
        "prev_dist_error": 0.0,
        "stuck_time": 0.0,
        "last_target_dist": None,
        "escape_sign": 1.0,
        "prev_leader_x": None,
        "prev_leader_z": None,
        "prev_leader_yaw": None,
        "leader_vx_f": 0.0,
        "leader_vz_f": 0.0,
        "leader_omega_f": 0.0,
    },
    "Right": {
        "prev_time": None,
        "heading_integral": 0.0,
        "prev_heading_error": 0.0,
        "dist_integral": 0.0,
        "prev_dist_error": 0.0,
        "stuck_time": 0.0,
        "last_target_dist": None,
        "escape_sign": -1.0,
        "prev_leader_x": None,
        "prev_leader_z": None,
        "prev_leader_yaw": None,
        "leader_vx_f": 0.0,
        "leader_vz_f": 0.0,
        "leader_omega_f": 0.0,
    }
}

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

def wrap_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

# =========================================================
# 單艘船控制邏輯
# =========================================================
def process_boat_rigid(sock, tx_port, boat_name, target_dx, target_dz, other_boat_pos=None):
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
    pid = pid_state[boat_name]

    prev_time = pid["prev_time"]
    heading_integral = pid["heading_integral"]
    prev_heading_error = pid["prev_heading_error"]
    dist_integral = pid["dist_integral"]
    prev_dist_error = pid["prev_dist_error"]
    stuck_time = pid["stuck_time"]
    last_target_dist = pid["last_target_dist"]
    escape_sign = pid["escape_sign"]
    prev_leader_x = pid["prev_leader_x"]
    prev_leader_z = pid["prev_leader_z"]
    prev_leader_yaw = pid["prev_leader_yaw"]
    leader_vx_f = pid["leader_vx_f"]
    leader_vz_f = pid["leader_vz_f"]
    leader_omega_f = pid["leader_omega_f"]


    current_time = time.time()
    if prev_time is None:
        dt = 0.01
        first_time = True
    else:
        dt = current_time - prev_time
        if dt <= 0:
            dt = 0.01
        first_time = False

    if "leader_x" not in state:
        sock.sendto(
            bytes(json.dumps({"throttle": 0.0, "steer": 0.0}), "utf-8"),
            (UDP_IP, tx_port)
        )
        return None

    leader_x = state["leader_x"]
    leader_z = state["leader_z"]
    actual_leader_yaw = state["leader_yaw"] + OFFSET # OFFSET Change Forward with Backword
    leader_yaw_rad = math.radians(actual_leader_yaw)

    if prev_leader_x is None or first_time:
        leader_vx_meas = 0.0
        leader_vz_meas = 0.0
        leader_omega_meas = 0.0
    else:
        leader_vx_meas = (leader_x - prev_leader_x) / dt  # Measurement of Leader Velocity
        leader_vz_meas = (leader_z - prev_leader_z) / dt
        leader_dyaw = wrap_angle_deg(actual_leader_yaw - prev_leader_yaw)
        leader_omega_meas = math.radians(leader_dyaw) / dt # Measurement of Leader Omega(角速度)

    leader_vx = LEADER_VEL_ALPHA * leader_vx_meas + (1.0 - LEADER_VEL_ALPHA) * leader_vx_f # Smoothing Velocity with Alpha Blending
    leader_vz = LEADER_VEL_ALPHA * leader_vz_meas + (1.0 - LEADER_VEL_ALPHA) * leader_vz_f # Avoid The Jitter Effect
    leader_omega = LEADER_VEL_ALPHA * leader_omega_meas + (1.0 - LEADER_VEL_ALPHA) * leader_omega_f

    leader_speed = math.hypot(leader_vx, leader_vz)

    offset_world_x = target_dx * math.cos(leader_yaw_rad) + target_dz * math.sin(leader_yaw_rad)
    offset_world_z = -target_dx * math.sin(leader_yaw_rad) + target_dz * math.cos(leader_yaw_rad)

    target_x = leader_x + offset_world_x
    target_z = leader_z + offset_world_z
    target_vx = leader_vx - leader_omega * offset_world_z
    target_vz = leader_vz + leader_omega * offset_world_x

    my_x = state["x"]
    my_z = state["z"]
    my_speed = state.get("speed", 0.0)

    pos_error_x = target_x - my_x
    pos_error_z = target_z - my_z
    target_dist = math.hypot(pos_error_x, pos_error_z)

    cmd_vx = target_vx + FORMATION_POS_GAIN * pos_error_x
    cmd_vz = target_vz + FORMATION_POS_GAIN * pos_error_z
    desired_speed = min(math.hypot(cmd_vx, cmd_vz), RIGID_SPEED_MAX)

    F_att_x = K_ATT * pos_error_x
    F_att_z = K_ATT * pos_error_z

    F_rep_agent_x = 0.0
    F_rep_agent_z = 0.0
    F_rep_leader_x = 0.0
    F_rep_leader_z = 0.0
    dist_to_other = None
    other_x = None
    other_z = None

    if other_boat_pos is not None:
        other_x, other_z = other_boat_pos
        dist_to_other = math.hypot(my_x - other_x, my_z - other_z)
        if 0 < dist_to_other < R_COL:
            rep_mag = K_REP_AGENT * ((R_COL - dist_to_other) / R_COL) / (dist_to_other + 2.0)
            F_rep_agent_x = rep_mag * (my_x - other_x) / dist_to_other
            F_rep_agent_z = rep_mag * (my_z - other_z) / dist_to_other

    dist_to_leader = math.hypot(my_x - leader_x, my_z - leader_z)

    if 0 < dist_to_leader < R_LEADER_COL:
        rep_mag_leader = K_REP_LEADER * (1.0 / dist_to_leader - 1.0 / R_LEADER_COL) / (dist_to_leader ** 2 + 1e-4)
        F_rep_leader_x = rep_mag_leader * (my_x - leader_x) / dist_to_leader
        F_rep_leader_z = rep_mag_leader * (my_z - leader_z) / dist_to_leader

    guide_x = cmd_vx + F_rep_agent_x + F_rep_leader_x
    guide_z = cmd_vz + F_rep_agent_z + F_rep_leader_z
    guide_norm = math.hypot(guide_x, guide_z)

    progress = 0.0 if last_target_dist is None else (last_target_dist - target_dist)
    in_local_min = (
        target_dist > LOCAL_MIN_TARGET_DIST_TH
        and guide_norm < LOCAL_MIN_TOTAL_FORCE_TH
        and progress < LOCAL_MIN_PROGRESS_TH
    )

    if in_local_min:
        stuck_time += dt
    else:
        stuck_time = 0.0

    if stuck_time >= LOCAL_MIN_STUCK_TIME:
        obs_x = 0.0
        obs_z = 0.0
        obs_weight = 0.0

        if dist_to_leader < R_LEADER_COL and dist_to_leader > 0:
            obs_x += (my_x - leader_x) / dist_to_leader
            obs_z += (my_z - leader_z) / dist_to_leader
            obs_weight += 1.0

        if dist_to_other is not None and dist_to_other < R_COL and dist_to_other > 0:
            obs_x += (my_x - other_x) / dist_to_other
            obs_z += (my_z - other_z) / dist_to_other
            obs_weight += 1.0

        if obs_weight == 0.0:
            obs_dist = max(math.hypot(F_att_x, F_att_z), 1e-6)
            obs_x = F_att_x / obs_dist
            obs_z = F_att_z / obs_dist
        else:
            obs_norm = max(math.hypot(obs_x, obs_z), 1e-6)
            obs_x /= obs_norm
            obs_z /= obs_norm

        tangent_x = -obs_z * escape_sign
        tangent_z = obs_x * escape_sign
        guide_x += ESCAPE_TANGENT_GAIN * tangent_x
        guide_z += ESCAPE_TANGENT_GAIN * tangent_z
        guide_norm = math.hypot(guide_x, guide_z)

    if leader_speed < LEADER_STOP_SPEED_TH and target_dist < FORMATION_HOLD_DIST_TH:
        target_angle = actual_leader_yaw
    elif guide_norm < MIN_GUIDE_VEC_NORM:
        target_angle = actual_leader_yaw
    else:
        target_angle = math.degrees(math.atan2(guide_x, guide_z))

    my_angle = state["yaw"] + OFFSET
    diff = wrap_angle_deg(target_angle - my_angle)

    my_angle_rad = math.radians(my_angle)
    forward_x = math.sin(my_angle_rad)
    forward_z = math.cos(my_angle_rad)
    
    desired_forward_speed = guide_x * forward_x + guide_z * forward_z
    desired_forward_speed = max(0.0, min(desired_forward_speed, RIGID_SPEED_MAX))

    catchup_throttle_bonus = 0.0

    if target_dist > CATCHUP_DIST_TH:
        catchup_gap = target_dist - CATCHUP_DIST_TH
        desired_forward_speed = max(
            desired_forward_speed,
            leader_speed + CATCHUP_SPEED_GAIN * catchup_gap
        )
        desired_forward_speed = min(desired_forward_speed, RIGID_SPEED_MAX + CATCHUP_SPEED_MAX_BOOST)
        catchup_throttle_bonus = min(CATCHUP_THROTTLE_MAX, CATCHUP_THROTTLE_GAIN * catchup_gap)

    if dist_to_other is not None and dist_to_other < 12.0:
        proximity_scale = max(0.25, dist_to_other / 12.0)
        desired_forward_speed *= proximity_scale

    if abs(diff) < DEADZONE:
        steer = 0.0
        prev_heading_error = diff
    else:
        p_term = KP_STEER * diff
        heading_integral += diff * dt
        heading_integral = max(min(heading_integral, HEADING_I_LIMIT), -HEADING_I_LIMIT)
        i_term = KI_STEER * heading_integral

        if first_time:
            heading_derivative = 0.0
        else:
            heading_derivative = (diff - prev_heading_error) / dt
        d_term = KD_STEER * heading_derivative

        steer = p_term + i_term + d_term
        prev_heading_error = diff

    steer = max(min(steer, 1.0), -1.0)

    speed_error = desired_forward_speed - my_speed
    if desired_forward_speed > 0.01:
        p_term_th = KP_THROTTLE * speed_error
        dist_integral += speed_error * dt
        dist_integral = max(min(dist_integral, DIST_I_LIMIT), -DIST_I_LIMIT)
        i_term_th = KI_THROTTLE * dist_integral

        if first_time:
            dist_derivative = 0.0
        else:
            dist_derivative = (speed_error - prev_dist_error) / dt
        d_term_th = KD_THROTTLE * dist_derivative

        throttle = p_term_th + i_term_th + d_term_th
        prev_dist_error = speed_error
    else:
        throttle = 0.0
        dist_integral = 0.0
        prev_dist_error = 0.0

    throttle += catchup_throttle_bonus
    throttle = max(min(throttle, 1.0), 0.0)

    # 角度越大，油門越小。
    turn_scale = max(0.25, 1.0 - abs(diff) / 90.0)
    throttle = throttle * turn_scale


    pid["prev_time"] = current_time
    pid["heading_integral"] = heading_integral
    pid["prev_heading_error"] = prev_heading_error
    pid["dist_integral"] = dist_integral
    pid["prev_dist_error"] = prev_dist_error
    pid["stuck_time"] = stuck_time
    pid["last_target_dist"] = target_dist
    pid["prev_leader_x"] = leader_x
    pid["prev_leader_z"] = leader_z
    pid["prev_leader_yaw"] = actual_leader_yaw
    pid["leader_vx_f"] = leader_vx
    pid["leader_vz_f"] = leader_vz
    pid["leader_omega_f"] = leader_omega

    # =====================================================
    # 視覺輔助：若前方中央偵測到大面積尾流，強制保守限速
    # =====================================================
    vision_brake = False
    if ENABLE_VISION_ASSIST and is_front_boat_danger():
        throttle = throttle * VISION_THROTTLE_SCALE
        vision_brake = True

    msg = json.dumps({"throttle": throttle, "steer": steer})
    sock.sendto(bytes(msg, "utf-8"), (UDP_IP, tx_port))

    speed_mps = state.get("speed", 0.0)
    return {
        "dist": guide_norm,
        "target_dist": target_dist,
        "throttle": throttle,
        "diff": diff,
        "speed_knots": speed_mps * 1.94384,
        "follower_speed": my_speed,
        "leader_speed": leader_speed,
        "catchup_bonus": catchup_throttle_bonus,
        "vision_brake": vision_brake,
        "pos": (state["x"], state["z"])
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
    last_pos_left = None
    last_pos_right = None

    while True:
        try:
           # 處理左護法 (傳入右護法的座標作為障礙物)
            res_left = process_boat_rigid(sock_left, PORT_LEFT_TX, "Left", POS_LEFT[0], POS_LEFT[1], last_pos_right)
            if res_left:
                last_pos_left = res_left["pos"] # 記錄左護法算完後的座標

            # 處理右護法 (傳入左護法的座標作為障礙物)
            res_right = process_boat_rigid(sock_right, PORT_RIGHT_TX, "Right", POS_RIGHT[0], POS_RIGHT[1], last_pos_left)
            if res_right:
                last_pos_right = res_right["pos"] # 記錄右護法算完後的座標

            current_time = time.time()
            if current_time - last_print_time > 0.1:
                print_str = ""

                if res_left:
                    print_str += (
                        f"[Leader {res_left['leader_speed']:4.1f} m/s | "
                        f"Left {res_left['follower_speed']:4.1f} m/s | "
                        f"Gap {res_left['target_dist']:5.1f} m | "
                        f"Thr {res_left['throttle']:4.2f} | "
                        f"Catch {res_left['catchup_bonus']:4.2f} | "
                        f"Yaw {res_left['diff']:6.1f}]"
                    )
                    if res_left["vision_brake"]:
                        print_str += " | 視覺限速:ON"

                if res_right:
                    if print_str != "":
                        print_str += " || "
                    print_str += (
                        f"[Leader {res_right['leader_speed']:4.1f} m/s | "
                        f"Right {res_right['follower_speed']:4.1f} m/s | "
                        f"Gap {res_right['target_dist']:5.1f} m | "
                        f"Thr {res_right['throttle']:4.2f} | "
                        f"Catch {res_right['catchup_bonus']:4.2f} | "
                        f"Yaw {res_right['diff']:6.1f}]"
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
