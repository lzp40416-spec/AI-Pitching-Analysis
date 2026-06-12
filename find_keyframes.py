import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. 通用角度計算函數 ---
def calculate_angle(a, b, c):
    """計算三點之間的夾角 (單位：度)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

# --- 2. 系統初始化與掃描 ---
mp_pose = mp.solutions.pose
video_path = "xc.mp4" 
pose_history = []

print("🎥 [系統初始化] 正在提取骨架與 3D 深度特徵...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 啟用 75% 信心度閾值，過濾動態模糊雜訊
with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            pose_history.append({
                'frame': frame_idx,
                'left_shoulder': [lm[11].x, lm[11].y],
                'right_shoulder': [lm[12].x, lm[12].y],
                'left_shoulder_z': lm[11].z,
                'right_shoulder_z': lm[12].z,
                
                'left_elbow': [lm[13].x, lm[13].y],      
                'right_elbow': [lm[14].x, lm[14].y],
                'left_wrist': [lm[15].x, lm[15].y],
                'right_wrist': [lm[16].x, lm[16].y],
                'right_index': [lm[20].x, lm[20].y],
                
                'left_hip': [lm[23].x, lm[23].y],
                'right_hip': [lm[24].x, lm[24].y],
                'left_hip_z': lm[23].z,
                'right_hip_z': lm[24].z,
                
                'left_knee': [lm[25].x, lm[25].y],
                'right_knee': [lm[26].x, lm[26].y], 
                'left_ankle': [lm[27].x, lm[27].y], 
                'right_ankle': [lm[28].x, lm[28].y], # 全身比例尺基準點
                
                'stride_dist': abs(lm[27].x - lm[28].x),
                'left_knee_y': lm[25].y
            })
        frame_idx += 1
cap.release()

print("🧠 [邏輯引擎啟動] 正在定位 6 大節點與萃取力學特徵...")

# --- 3. 核心運算：定位 6 大節點 ---

# 📌 節點 1：抬腿最高點
f1_data = min(pose_history, key=lambda x: x['left_knee_y'])
f1 = f1_data['frame']

# 📌 節點 2：前腳著地
f2_candidates = [d for d in pose_history if d['frame'] > f1]
f2_data = max(f2_candidates, key=lambda x: x['stride_dist'])
f2 = f2_data['frame']

# 📌 節點 1.5：雙手分離
f1_5 = f1
for data in pose_history:
    if f1 < data['frame'] < f2:
        wrist_dist = math.dist(data['left_wrist'], data['right_wrist'])
        sh_width = math.dist(data['left_shoulder'], data['right_shoulder'])
        if wrist_dist > (sh_width * 1.2):
            f1_5 = data['frame']
            break

# 📌 節點 4：放球瞬間
f4_raw = f2 + 10 
max_ext_angle = -1.0

for data in pose_history:
    if f2 < data['frame'] <= f2 + 45:
        # [防護網] 手腕必須在肩膀前方、手腕不得低於手肘、手指必須在手腕前方
        wrist_in_front = data['right_wrist'][0] > data['right_shoulder'][0]
        wrist_above_elbow = data['right_wrist'][1] <= data['right_elbow'][1]
        finger_in_front = data['right_index'][0] > data['right_wrist'][0]
        
        if wrist_in_front and wrist_above_elbow and finger_in_front:
            angle = calculate_angle(data['right_shoulder'], data['right_elbow'], data['right_wrist'])
            if angle > max_ext_angle:
                max_ext_angle = angle
                f4_raw = data['frame']

# 實務校正：倒退一幀作為真實放球點
f4 = max(f2 + 1, f4_raw - 1)

# 📌 節點 3：最大肩外旋 (MER)
f3 = f2 + 1
min_y_diff = 999.0
for data in pose_history:
    if f2 < data['frame'] < f4:
        wrist_behind_elbow = data['right_wrist'][0] < data['right_elbow'][0]
        finger_behind_wrist = data['right_index'][0] < data['right_wrist'][0]
        finger_behind_hip = data['right_index'][0] < data['right_hip'][0]
        
        if wrist_behind_elbow and finger_behind_wrist and finger_behind_hip:
            y_diff = abs(data['right_wrist'][1] - data['right_elbow'][1])
            if y_diff < min_y_diff:
                min_y_diff = y_diff
                f3 = data['frame']

# 📌 節點 5：減速期結束 (後腳高踢，Y軸超越)
f5 = f4 + 10 
for data in pose_history:
    if f4 < data['frame'] <= f4 + 45: 
        if data['right_knee'][1] < data['left_knee'][1]:
            f5 = data['frame']
            break

# --- 4. 萃取 10 大黃金力學特徵 ---

node2_data = next(d for d in pose_history if d['frame'] == f2)
node4_data = next(d for d in pose_history if d['frame'] == f4)
node5_data = next((d for d in pose_history if d['frame'] == f5), pose_history[-1])

# [特徵 1] 髖肩分離度 (Node 2)
hip_twist = abs(node2_data['left_hip_z'] - node2_data['right_hip_z'])
shoulder_twist = abs(node2_data['left_shoulder_z'] - node2_data['right_shoulder_z'])
separation_index = hip_twist - shoulder_twist 

# [特徵 2] 手臂延遲預警 Late Cocking (Node 2)
is_late_cocking = node2_data['right_wrist'][1] > node2_data['right_shoulder'][1]

# [特徵 3] 手套手鎖定角 (Node 4)
glove_arm_angle = calculate_angle(node4_data['left_shoulder'], node4_data['left_elbow'], node4_data['left_wrist'])

# [特徵 4] 放球點延伸比例 (Node 4)
release_ext_x = abs(node4_data['right_wrist'][0] - node4_data['left_ankle'][0])
body_len_n4 = abs(node4_data['left_ankle'][1] - node4_data['right_shoulder'][1])
ext_ratio = release_ext_x / body_len_n4 if body_len_n4 > 0 else 0

# [特徵 5] 前腳支撐煞車角 (Node 5)
bracing_angle = calculate_angle(node5_data['left_hip'], node5_data['left_knee'], node5_data['left_ankle'])

# [特徵 6] 抬腿平衡/軀幹傾斜角 (Node 1)
n1_data = next(d for d in pose_history if d['frame'] == f1)
mid_shoulder_n1_x = (n1_data['left_shoulder'][0] + n1_data['right_shoulder'][0]) / 2
mid_shoulder_n1_y = (n1_data['left_shoulder'][1] + n1_data['right_shoulder'][1]) / 2
mid_hip_n1_x = (n1_data['left_hip'][0] + n1_data['right_hip'][0]) / 2
mid_hip_n1_y = (n1_data['left_hip'][1] + n1_data['right_hip'][1]) / 2
dx1 = mid_shoulder_n1_x - mid_hip_n1_x 
dy1 = mid_hip_n1_y - mid_shoulder_n1_y
lift_torso_tilt = math.degrees(math.atan2(dx1, dy1))

# [特徵 7] 蓄力手肘屈曲角 (Node 2 -> Node 3)
best_cocking_angle = 0
min_diff_to_90 = 999.0
for d in pose_history:
    if f2 <= d['frame'] <= f3:
        if d['right_index'][1] < d['right_elbow'][1]:
            ang = calculate_angle(d['right_shoulder'], d['right_elbow'], d['right_wrist'])
            if abs(ang - 90) < min_diff_to_90:
                min_diff_to_90 = abs(ang - 90)
                best_cocking_angle = ang

# [特徵 8] 放球軀幹前傾角 (Node 4)
mid_shoulder_n4_x = (node4_data['left_shoulder'][0] + node4_data['right_shoulder'][0]) / 2
mid_shoulder_n4_y = (node4_data['left_shoulder'][1] + node4_data['right_shoulder'][1]) / 2
mid_hip_n4_x = (node4_data['left_hip'][0] + node4_data['right_hip'][0]) / 2
mid_hip_n4_y = (node4_data['left_hip'][1] + node4_data['right_hip'][1]) / 2
dx4 = mid_shoulder_n4_x - mid_hip_n4_x
dy4 = mid_hip_n4_y - mid_shoulder_n4_y
release_trunk_tilt = math.degrees(math.atan2(dx4, dy4))

# [特徵 9] 跨步期下潛比例 (Node 1 -> Node 2)
hip_y_n1 = mid_hip_n1_y 
max_hip_y = hip_y_n1    
for d in pose_history:
    if f1 <= d['frame'] <= f2:
        current_hip_y = (d['left_hip'][1] + d['right_hip'][1]) / 2
        if current_hip_y > max_hip_y:
            max_hip_y = current_hip_y
body_len_n1 = abs(((n1_data['left_ankle'][1] + n1_data['right_ankle'][1])/2) - mid_shoulder_n1_y)
drop_ratio = (max_hip_y - hip_y_n1) / body_len_n1 if body_len_n1 > 0 else 0
drop_percentage = drop_ratio * 100

# [特徵 10] 跨步幅度比例 (Node 2)
stride_len = abs(node2_data['left_ankle'][0] - node2_data['right_ankle'][0])
stride_ratio = stride_len / body_len_n1 if body_len_n1 > 0 else 0
stride_percentage = stride_ratio * 100

# [特徵 11] 放球肩外展角/手肘下掉 (Node 4)
shoulder_abduction_angle = calculate_angle(node4_data['right_hip'], node4_data['right_shoulder'], node4_data['right_elbow'])


# --- 5. 截圖輸出模組 ---
print("📸 [輸出模組] 正在匯出 6 大關鍵幀影像...")
cap = cv2.VideoCapture(video_path)
for f_num, name in [(f1, "N1_Lift.jpg"), (f1_5, "N1.5_Break.jpg"), (f2, "N2_Strike.jpg"), 
                    (f3, "N3_MER.jpg"), (f4, "N4_Release.jpg"), (f5, "N5_Finish.jpg")]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
    ret, img = cap.read()
    if ret: cv2.imwrite(name, img)
cap.release()

# ==========================================
# 🌟 數據直觀化轉換邏輯
# ==========================================
sep_score = max(0, min(100, int((separation_index + 0.08) / 0.16 * 100)))
ext_percentage = ext_ratio * 100

if glove_arm_angle < 100: glove_status = "✅ 完美鎖定"
elif glove_arm_angle < 130: glove_status = "⚠️ 微幅外開"
else: glove_status = "❌ 嚴重開掉 (力量流失)"

if bracing_angle > 150: brace_status = "✅ 完美支撐"
else: brace_status = "⚠️ 煞車不足 (膝蓋彎曲)"


# --- 6. 系統報告 (深度力學儀表板) ---
print("\n" + "="*75)
print("🎯 棒球投手動作分析系統 - 深度姿勢優化與受傷預警報告")
print("="*75)
print(f"【時間軸與動作節點】")
print(f"📍 N1(抬腿): {f1} | N1.5(分離): {f1_5} | N2(著地): {f2} | N3(MER): {f3} | N4(放球): {f4} | N5(減速): {f5}")
print("-" * 75)

print("【💪 動能蓄積與轉移 (下盤分析)】")
if -5 < lift_torso_tilt < 5:
    print(f"🔹 抬腿平衡   : {lift_torso_tilt:.1f} 度 (✅ 重心完美，脊椎直立)")
elif lift_torso_tilt < -5:
    print(f"🔹 抬腿平衡   : {lift_torso_tilt:.1f} 度 (⚠️ 過度後仰，易導致放球點飄忽)")
else:
    print(f"🔹 抬腿平衡   : {lift_torso_tilt:.1f} 度 (⚠️ 過度前傾，重心提早崩潰)")

if drop_percentage > 8:
    print(f"🔹 下盤驅動   : 下潛 {drop_percentage:.1f}% (✅ 成功利用後腳推蹬 Drop & Drive)")
else:
    print(f"🔹 下盤驅動   : 下潛 {drop_percentage:.1f}% (⚠️ 下潛不足，Tall & Fall 易流失動能)")

if stride_percentage >= 85:
    print(f"🔹 跨步幅度   : 身體長度的 {stride_percentage:.1f}% (✅ 跨步極佳，完美延伸動力鏈)")
elif 70 <= stride_percentage < 85:
    print(f"🔹 跨步幅度   : 身體長度的 {stride_percentage:.1f}% (⚠️ 跨步偏小，可能限制球威)")
else:
    print(f"🔹 跨步幅度   : 身體長度的 {stride_percentage:.1f}% (❌ 跨步嚴重不足，導致放球點過早)")


print("\n【⚾ 力量釋放與煞車 (上盤分析)】")
print(f"🔹 肩髖分離度 : {sep_score} 分 / 100 分 (原始 Z 軸差: {separation_index:.4f})")
print(f"🔹 手套手狀態 : {glove_arm_angle:.1f} 度 ({glove_status})")
print(f"🔹 放球延伸度 : 往前延伸等同於 {ext_percentage:.1f}% 的半身高度")

if release_trunk_tilt > 20:
    print(f"🔹 軀幹前傾角 : {release_trunk_tilt:.1f} 度 (✅ 軀幹完美前壓，釋放力量)")
else:
    print(f"🔹 軀幹前傾角 : {release_trunk_tilt:.1f} 度 (❌ 直立放球，過度依賴手臂發力)")

print(f"🔹 前腳煞車角 : {bracing_angle:.1f} 度 ({brace_status})")


print("\n【🚨 隱性受傷風險與關節壓力預警】")
print(f"⚠️ 手臂延遲   : {'❌ 偵測到 Late Cocking (高風險)' if is_late_cocking else '✅ 舉起時機標準'}")

if best_cocking_angle == 0:
    print(f"⚠️ 蓄力手肘角(投球手) : 無法成功偵測，動作可能被遮擋。")
elif 85 < best_cocking_angle < 115:
    print(f"⚠️ 蓄力手肘角(投球手) : {best_cocking_angle:.1f} 度 (✅ 位於安全範圍，韌帶壓力正常)")
else:
    print(f"⚠️ 蓄力手肘角(投球手): {best_cocking_angle:.1f} 度 (❌ 角度過大/過小，UCL 承受極大撕裂張力！)")

# 🌟 修改版：放球點手肘下掉診斷 (專注於抓出推鉛球的錯誤)
if shoulder_abduction_angle <= 85:
    # 嚴格抓出小於 85 度的手肘下掉
    print(f"⚠️ 放球手肘角 : {shoulder_abduction_angle:.1f} 度 (❌ 手肘下掉！推鉛球發力，極易傷及肩關節與 UCL)")

print("="*75)
print("✅ 資料處理完畢！")