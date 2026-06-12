import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """向量內積公式計算夾角 (單位：度)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b #向量
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2) # 向量長度
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi


class PitchingAnalyzer:
    """
    骨架座標提取
    負責：骨架提取、自動預判節點、生物力學特徵運算
    """
    # Mediapipe Pose 模型初始化
    # 偵測、追踪信心度75%
    def __init__(self, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        self.mp_pose = mp.solutions.pose # mp人體骨架模組
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_pose_history(self, video_path):
        """
        階段一：掃描影片，提取每一幀的 3D 骨架座標
        :param video_path: 影片檔案路徑
        :return: pose_history (List of Dictionaries)
        """
        pose_history = []
        cap = cv2.VideoCapture(video_path) # 呼叫OpenCV
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB色彩轉換
            results = self.pose.process(image_rgb) # 丟進骨架偵測引擎


            # 定義關鍵節點
            # (肩、肘、腕、手指、髖、膝、踝)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                pose_history.append({
                    'frame': frame_idx,
                    'left_shoulder': [lm[11].x, lm[11].y], 'right_shoulder': [lm[12].x, lm[12].y],
                    'left_shoulder_z': lm[11].z, 'right_shoulder_z': lm[12].z,
                    'left_elbow': [lm[13].x, lm[13].y], 'right_elbow': [lm[14].x, lm[14].y],
                    'left_wrist': [lm[15].x, lm[15].y], 'right_wrist': [lm[16].x, lm[16].y],
                    'right_index': [lm[20].x, lm[20].y],
                    'left_hip': [lm[23].x, lm[23].y], 'right_hip': [lm[24].x, lm[24].y],
                    'left_hip_z': lm[23].z, 'right_hip_z': lm[24].z,
                    'left_knee': [lm[25].x, lm[25].y], 'right_knee': [lm[26].x, lm[26].y],
                    'left_ankle': [lm[27].x, lm[27].y], 'right_ankle': [lm[28].x, lm[28].y],
                    'stride_dist': abs(lm[27].x - lm[28].x),
                    'left_knee_y': lm[25].y
                })
            frame_idx += 1
            
        cap.release()
        return pose_history

    def auto_detect_keyframes(self, pose_history):
        """
        階段二：投手動作偵測引擎，自動判別 6 大關鍵節點
        :param pose_history: extract_pose_history 產出的骨架歷史資料
        :return: keyframes (Dictionary) 包含 f1 到 f5 的幀數
        """
        if not pose_history:
            return {}

        # 抬腿最高點f1
        # 左膝最高點
        f1 = min(pose_history, key=lambda x: x['left_knee_y'])['frame']
        
        # 前腳著地f2
        # 兩腳踝最大距離
        f2_candidates = [d for d in pose_history if d['frame'] > f1]
        f2 = max(f2_candidates, key=lambda x: x['stride_dist'])['frame'] if f2_candidates else f1 + 10

        # 雙手分離f1_5
        # 手腕距離大於肩寬1.2倍以上
        f1_5 = f1
        for data in pose_history:
            if f1 < data['frame'] < f2:
                wrist_dist = math.dist(data['left_wrist'], data['right_wrist'])
                sh_width = math.dist(data['left_shoulder'], data['right_shoulder'])
                if wrist_dist > (sh_width * 1.2):
                    f1_5 = data['frame']
                    break

        # 放球點f4
        # 手肘伸展最大角度
        f4_raw = f2 + 10 
        max_ext_angle = -1.0
        for data in pose_history:
            if f2 < data['frame'] <= f2 + 45:
                if (data['right_wrist'][0] > data['right_shoulder'][0] and 
                    data['right_wrist'][1] <= data['right_elbow'][1] and 
                    data['right_index'][0] > data['right_wrist'][0]):
                    angle = calculate_angle(data['right_shoulder'], data['right_elbow'], data['right_wrist'])
                    if angle > max_ext_angle:
                        max_ext_angle = angle
                        f4_raw = data['frame']
        f4 = max(f2 + 1, f4_raw - 1)

        # 最大肩外旋 MER f3
        # 手腕與手肘最接近水平狀態
        f3 = f2 + 1
        min_y_diff = 999.0
        for data in pose_history:
            if f2 < data['frame'] < f4:
                if (data['right_wrist'][0] < data['right_elbow'][0] and 
                    data['right_index'][0] < data['right_wrist'][0] and 
                    data['right_index'][0] < data['right_hip'][0]):
                    y_diff = abs(data['right_wrist'][1] - data['right_elbow'][1])
                    if y_diff < min_y_diff:
                        min_y_diff = y_diff
                        f3 = data['frame']

        # 前腳煞車f5
        # 右膝高度超過左膝
        f5 = f4 + 10 
        for data in pose_history:
            if f4 < data['frame'] <= f4 + 45: 
                if data['right_knee'][1] < data['left_knee'][1]:
                    f5 = data['frame']
                    break

        return {"f1": f1, "f1_5": f1_5, "f2": f2, "f3": f3, "f4": f4, "f5": f5}


    def calculate_biomechanics(self, pose_history, keyframes):
        """
        階段三：根據「最終確認的節點」計算 11 大力學特徵
        """
        f1, f2, f3, f4, f5 = keyframes['f1'], keyframes['f2'], keyframes['f3'], keyframes['f4'], keyframes['f5']
        
        # --- 防崩潰 --- 抓取不到關鍵幀，就用第一幀或最後一幀補上
        n1_data = next((d for d in pose_history if d['frame'] == f1), pose_history[0])
        n2_data = next((d for d in pose_history if d['frame'] == f2), pose_history[0])
        n4_data = next((d for d in pose_history if d['frame'] == f4), pose_history[-1])
        n5_data = next((d for d in pose_history if d['frame'] == f5), pose_history[-1])

        
        # 前置作業：建立身高與下潛比例尺
        # ==========================================
        mid_sh_n1_y = (n1_data['left_shoulder'][1] + n1_data['right_shoulder'][1]) / 2 # 肩膀中心點
        hip_y_n1 = (n1_data['left_hip'][1] + n1_data['right_hip'][1]) / 2 # 髖部中心點
        max_hip_y = max([ (d['left_hip'][1] + d['right_hip'][1]) / 2 for d in pose_history if f1 <= d['frame'] <= f2 ], default=hip_y_n1)
        # 髖部下淺最低點
        body_len_n1 = abs(((n1_data['left_ankle'][1] + n1_data['right_ankle'][1])/2) - mid_sh_n1_y)
        # 身高比例尺

        
        #第一類：發力與協調性
        # ==========================================
        
        # 1. 抬腿平衡
        # F1 計算肩膀中線與髖部中線的X座標頃斜角度
        mid_sh_n1_x = (n1_data['left_shoulder'][0] + n1_data['right_shoulder'][0]) / 2
        mid_hp_n1_x = (n1_data['left_hip'][0] + n1_data['right_hip'][0]) / 2
        lift_torso_tilt = math.degrees(math.atan2(mid_sh_n1_x - mid_hp_n1_x, hip_y_n1 - mid_sh_n1_y))

        # 2. 下潛幅度
        # F1-F2 髖部最高點-髖部最低點/身高*100%
        drop_percentage = ((max_hip_y - hip_y_n1) / body_len_n1 * 100) if body_len_n1 > 0 else 0

        # 3. 跨步幅度
        # F2 擷取左右腳踝的距離/身高*100%
        stride_len = abs(n2_data['left_ankle'][0] - n2_data['right_ankle'][0])
        stride_percentage = (stride_len / body_len_n1 * 100) if body_len_n1 > 0 else 0


        # 第二類：力量轉移與結束
        # ==========================================

        # 4. 髖肩分離度
        # F2 左右髖部的Z軸差-左右肩膀的Z軸差
        separation_index = abs(n2_data['left_hip_z'] - n2_data['right_hip_z']) - abs(n2_data['left_shoulder_z'] - n2_data['right_shoulder_z'])

        # 5. 手套手穩定度
        # F4 放球點，計算手套手的夾角
        glove_arm_angle = calculate_angle(n4_data['left_shoulder'], n4_data['left_elbow'], n4_data['left_wrist'])

        # 6. 放球點延伸比例
        # F4 右手腕與左腳腳踝的水平距離/身高
        body_len_n4 = abs(n4_data['left_ankle'][1] - n4_data['right_shoulder'][1])
        release_ext_x = abs(n4_data['right_wrist'][0] - n4_data['left_ankle'][0])
        ext_ratio = release_ext_x / body_len_n4 if body_len_n4 > 0 else 0

        # 7. 軀幹前傾角
        # F4 計算肩膀中線與髖部中線的頃斜角
        mid_sh_n4_x, mid_sh_n4_y = (n4_data['left_shoulder'][0] + n4_data['right_shoulder'][0]) / 2, (n4_data['left_shoulder'][1] + n4_data['right_shoulder'][1]) / 2
        mid_hp_n4_x, mid_hp_n4_y = (n4_data['left_hip'][0] + n4_data['right_hip'][0]) / 2, (n4_data['left_hip'][1] + n4_data['right_hip'][1]) / 2
        release_trunk_tilt = math.degrees(math.atan2(mid_sh_n4_x - mid_hp_n4_x, mid_hp_n4_y - mid_sh_n4_y))

        # 8. 前腳煞車
        # F5 右腳膝蓋的夾角
        bracing_angle = calculate_angle(n5_data['left_hip'], n5_data['left_knee'], n5_data['left_ankle'])


        # 第三類：受傷防護警告
        # ==========================================

        # 9. 手臂延遲
        # F2前腳著地時，檢查右手腕有沒有低於右肩膀
        is_late_cocking = 1 if n2_data['right_wrist'][1] > n2_data['right_shoulder'][1] else 0
    
        # 10. 蓄力手肘角
        # F2和F3之間，尋找最接近90度的手肘夾角
        best_cocking_angle = 0
        min_diff = 999.0
        for d in pose_history:
            if f2 <= d['frame'] <= f3 and d['right_index'][1] < d['right_elbow'][1]:
                ang = calculate_angle(d['right_shoulder'], d['right_elbow'], d['right_wrist'])
                if abs(ang - 90) < min_diff:
                    min_diff = abs(ang - 90)
                    best_cocking_angle = ang

        # 11. 手肘下掉 (肩外展角)
        # F4檢查手臂與身體的夾角
        shoulder_abduction_angle = calculate_angle(n4_data['right_hip'], n4_data['right_shoulder'], n4_data['right_elbow'])


        # 打包回傳給前端或資料庫
        # ==========================================
        return {
            "lift_torso_tilt": round(lift_torso_tilt, 1),
            "drop_percentage": round(drop_percentage, 1),
            "stride_percentage": round(stride_percentage, 1),
            "hip_shoulder_sep": round(separation_index, 4),
            
            "glove_arm_angle": round(glove_arm_angle, 1),
            "extension_ratio": round(ext_ratio, 3),
            "release_trunk_tilt": round(release_trunk_tilt, 1),
            "bracing_angle": round(bracing_angle, 1),
            
            "is_late_cocking": is_late_cocking,
            "cocking_elbow_angle": round(best_cocking_angle, 1),
            "shoulder_abduction_angle": round(shoulder_abduction_angle, 1)
        }