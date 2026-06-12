import cv2      
import math     
import sys      
import os       


# 0. 環境路徑設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.analyzer import PitchingAnalyzer
from database.db_manager import DBManager


# 1. 載入影片
# ==========================================
video_path = "videos/江少慶.mp4" # 影片路徑

# 實例化力學引擎與資料庫
analyzer = PitchingAnalyzer() 
db = DBManager()              

print("🎥 [系統初始化] 核心引擎正在背景掃描影片...")
pose_history = analyzer.extract_pose_history(video_path)

print("🤖 [AI 預判] 引擎正在計算 6 大節點初稿...")
suggested_keys = analyzer.auto_detect_keyframes(pose_history)

print("📺 [載入畫面] 準備互動校閱介面...")
video_frames = []
cap = cv2.VideoCapture(video_path)

# 抓取影片的寬高
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 抓取影片的 FPS 與總幀數
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

# 把影片存進RAM陣列
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    video_frames.append(frame)
cap.release()


# ==========================================
# 2. 人工校閱模組 (UI 邏輯)
# ==========================================
dragging_point = None 
current_f_idx = 0     

def mouse_event(event, x, y, flags, param):
    """滑鼠操作模組：負責處理點擊與拖曳骨架節點"""
    global dragging_point, current_f_idx
    if current_f_idx >= len(pose_history): return
    
    # 將像素座標 (x, y) 轉回 0~1 的正規化座標(MediaPipe 存的是正規化座標)
    norm_x, norm_y = x / width, y / height
    
    # 定義可以被滑鼠拖曳的 10 個節點 (肩膀、手肘、手腕、髖、膝、踝)
    draggable_keys = ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle', 'right_ankle']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for key in draggable_keys:
            # 算出該節點的實際像素座標
            px, py = int(pose_history[current_f_idx][key][0] * width), int(pose_history[current_f_idx][key][1] * height)
            if math.dist([x, y], [px, py]) < 15:
                dragging_point = key
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        # 當滑鼠移動，直接覆寫 pose_history 記憶體裡面的座標值！
        if dragging_point:
            pose_history[current_f_idx][dragging_point] = [norm_x, norm_y] 
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

# 建立 OpenCV 視窗
cv2.namedWindow("AI Pitching Coach - Verification")
cv2.setMouseCallback("AI Pitching Coach - Verification", mouse_event)

# 定義校閱的 6 大節點
nodes_order = [
    ("Node 1 (Lift)", "f1"), ("Node 1.5 (Break)", "f1_5"), 
    ("Node 2 (Strike)", "f2"), ("Node 3 (MER)", "f3"), 
    ("Node 4 (Release)", "f4"), ("Node 5 (Finish)", "f5")
]

final_keys = {} 
node_idx = 0    

# 外層迴圈： 6 大階段
while node_idx < len(nodes_order):
    ui_name, dict_key = nodes_order[node_idx]
    
    # 載入記憶進度 (如果按 B 退回上一動)，或者載入引擎預判的值
    if dict_key in final_keys:
        current_f_idx = final_keys[dict_key]
    else:
        current_f_idx = suggested_keys[dict_key]
    
    # 內層迴圈：負責不斷刷新當下這一幀的畫面 
    while True:
        display_frame = video_frames[current_f_idx].copy() 
        
        # UI 提示文字
        cv2.putText(display_frame, f"Reviewing: {ui_name} ({node_idx+1}/6)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {current_f_idx} (A/D: Move | ENTER: Next | B: Back)", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "[Mouse Drag] to fix bad keypoints", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        data = pose_history[current_f_idx]
        
        # 內部繪圖小函數：畫圓點
        def draw_pt(key, color=(0,0,255)):
            cx, cy = int(data[key][0]*width), int(data[key][1]*height)
            radius = 10 if dragging_point == key else 6 
            cv2.circle(display_frame, (cx, cy), radius, color, -1)
            
        # 內部繪圖小函數：畫骨架連線
        def draw_line(k1, k2, color=(255,105,180)):
            cv2.line(display_frame, (int(data[k1][0]*width), int(data[k1][1]*height)), 
                     (int(data[k2][0]*width), int(data[k2][1]*height)), color, 3)

        # 畫出手臂與下盤的骨架連線
        draw_line('right_shoulder', 'right_elbow'); draw_line('right_elbow', 'right_wrist')
        draw_line('left_shoulder', 'left_elbow'); draw_line('left_elbow', 'left_wrist')
        draw_line('left_hip', 'left_knee'); draw_line('left_knee', 'left_ankle')
        
        # 畫出 10 個核心控制點
        for k in ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle', 'right_ankle']:
            draw_pt(k)

        cv2.imshow("AI Pitching Coach - Verification", display_frame)
        
        # 鍵盤事件監聽
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a') and current_f_idx > 0: current_f_idx -= 1 # A鍵：畫面倒退一幀
        elif key == ord('d') and current_f_idx < len(video_frames) - 1: current_f_idx += 1 # D鍵：畫面快進一幀
        elif key == 13: # Enter鍵：鎖定此幀為標準答案，進入下一個階段
            final_keys[dict_key] = current_f_idx
            node_idx += 1
            break
        elif key == ord('b') or key == ord('B'): # B鍵：發現上一步標錯了，退回上一個階段重標
            if node_idx > 0:
                final_keys[dict_key] = current_f_idx 
                node_idx -= 1 
                break

cv2.destroyAllWindows()

# ==========================================
# 3. 呼叫引擎結算與診斷報告
# ==========================================
print("\n⚙️ 正在呼叫引擎進行深度物理運算...")
# 將手動微調過的骨架，丟給力學引擎
features = analyzer.calculate_biomechanics(pose_history, final_keys)

# 資料直觀化 (將原始數值轉換為 100分制)
sep_score = max(0, min(100, int((features['hip_shoulder_sep'] + 0.08) / 0.16 * 100)))

if features['glove_arm_angle'] < 100: glove_status = "✅ 完美鎖定"
elif features['glove_arm_angle'] < 130: glove_status = "⚠️ 微幅外開"
else: glove_status = "❌ 嚴重開掉 (力量流失)"

if features['bracing_angle'] > 150: brace_status = "✅ 完美支撐"
else: brace_status = "⚠️ 煞車不足 (膝蓋彎曲)"


print("\n" + "="*75)
print("🎯 系統診斷報告 (人工校閱確認版)")
print("="*75)
print(f"【時間軸與最終確認節點】")
print(f"📍 N1: {final_keys['f1']} | N1.5: {final_keys['f1_5']} | N2: {final_keys['f2']} | N3: {final_keys['f3']} | N4: {final_keys['f4']} | N5: {final_keys['f5']}")
print("-" * 75)

print("【💪 動能蓄積與轉移 (下盤分析)】")
if -5 < features['lift_torso_tilt'] < 5:
    print(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (✅ 重心完美，脊椎直立)")
elif features['lift_torso_tilt'] < -5:
    print(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (⚠️ 過度後仰，易導致放球點飄忽)")
else:
    print(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (⚠️ 過度前傾，重心提早崩潰)")

if features['drop_percentage'] > 8:
    print(f"🔹 下盤驅動   : 下潛 {features['drop_percentage']}% (✅ 成功利用後腳推蹬 Drop & Drive)")
else:
    print(f"🔹 下盤驅動   : 下潛 {features['drop_percentage']}% (⚠️ 下潛不足，Tall & Fall 易流失動能)")

if features['stride_percentage'] >= 85:
    print(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (✅ 跨步極佳，完美延伸動力鏈)")
elif 70 <= features['stride_percentage'] < 85:
    print(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (⚠️ 跨步偏小，可能限制球威)")
else:
    print(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (❌ 跨步嚴重不足，導致放球點過早)")

print("\n【⚾ 力量釋放與煞車 (上盤分析)】")
print(f"🔹 髖肩分離度 : {sep_score} 分 / 100 分 (原始 Z 軸差: {features['hip_shoulder_sep']})")
print(f"🔹 手套手狀態 : {features['glove_arm_angle']} 度 ({glove_status})")
print(f"🔹 放球延伸度 : 往前延伸等同於 {features['extension_ratio']*100:.1f}% 的半身高度")

if features['release_trunk_tilt'] > 20:
    print(f"🔹 軀幹前傾角 : {features['release_trunk_tilt']} 度 (✅ 軀幹完美前壓，釋放力量)")
else:
    print(f"🔹 軀幹前傾角 : {features['release_trunk_tilt']} 度 (❌ 直立放球，過度依賴手臂發力)")

print(f"🔹 前腳煞車角 : {features['bracing_angle']} 度 ({brace_status})")

print("\n【🚨 隱性受傷風險與關節壓力預警】")
print(f"⚠️ 手臂延遲   : {'❌ 偵測到 Late Cocking (高風險)' if features['is_late_cocking'] else '✅ 舉起時機標準'}")

if features['cocking_elbow_angle'] == 0:
    print(f"⚠️ 蓄力手肘角 : 無法成功偵測，動作可能被遮擋。")
elif 85 < features['cocking_elbow_angle'] < 115:
    print(f"⚠️ 蓄力手肘角 : {features['cocking_elbow_angle']} 度 (✅ 位於安全範圍，韌帶壓力正常)")
else:
    print(f"⚠️ 蓄力手肘角 : {features['cocking_elbow_angle']} 度 (❌ 角度過大/過小，UCL 承受極大撕裂張力！)")

if features['shoulder_abduction_angle'] >= 85:
    print(f"⚠️ 放球手肘角 : {features['shoulder_abduction_angle']} 度 (✅ 手肘高度完美，肩膀分擔受力)")
else:
    print(f"⚠️ 放球手肘角 : {features['shoulder_abduction_angle']} 度 (❌ 手肘下掉！推鉛球發力，極易傷及肩關節與 UCL)")

print("="*75)

# ==========================================
# 4. 寫入標準資料庫
# ==========================================
print("\n📝 準備歸檔，請輸入投手分類資訊：")

# 自動抓取檔名作為投手名稱 (例如 "江少慶.mp4" -> "江少慶")
pitcher_name = os.path.splitext(os.path.basename(video_path))[0]
print(f"🤖 目前投手: {pitcher_name}")

# 【物理意義】電腦視覺無法準確得知投手的慣用手與發力流派，因此需要人類補足這兩個 Metadata
# 1. 輸入慣用手
h_choice = input("👉 慣用手 (輸入 R 代表右投 / L 代表左投): ").upper()
handedness = "R" if h_choice == "R" else "L"

# 2. 輸入出手點
print("👉 出手點類型: [1]高壓(Overhand)  [2]上肩(3/4)  [3]側投(Sidearm)  [4]低肩(Submarine)")
s_choice = input("請選擇編號 (1-4): ")
slots = {"1": "高壓", "2": "3/4", "3": "側投", "4": "低肩"}
arm_slot = slots.get(s_choice, "Unknown")

# 3. 按照順序呼叫寫入函數，將這筆「黃金標準資料」存入 SQLite 供前端 Web App 比對用
db.insert_pro_data(
    name=pitcher_name,
    handedness=handedness,
    arm_slot=arm_slot,
    filename=video_path,
    features=features,
    keyframes=final_keys,      # 傳入微調後的關鍵幀字典
    fps=fps,                   # 傳入影片 FPS
    total_frames=total_frames, # 傳入總幀數
    is_verified=True           # 標記這筆資料是經過「人工雙重驗證」的完美資料
)

db.close()
print("✅ 資料歸檔完畢！")

# ==========================================
# 5. 自動歸檔：將處理完的影片搬移到專屬資料夾
# ==========================================
# 【工程意義】Data Pipeline 技巧：自動將處理過的檔案移走，
# 這樣下次執行程式時，/videos 資料夾永遠只會有「還沒處理的新影片」，防止重複作業。

processed_dir = "processed_videos" # 設定已處理檔案的資料夾

# 如果資料夾不存在，自動建立
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

video_filename = os.path.basename(video_path)
new_video_path = os.path.join(processed_dir, video_filename)

try:
    # 執行檔案搬移 (從 videos/ 搬到 processed_videos/)
    os.replace(video_path, new_video_path)
    print(f"📦 [系統] 影片已自動移至「已處理區」: {new_video_path}")
except Exception as e:
    print(f"⚠️ [系統警告] 影片搬移失敗，請確認檔案是否被其他程式占用: {e}")