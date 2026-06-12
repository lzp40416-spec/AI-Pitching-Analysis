from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import shutil
import os
import cv2
import mediapipe as mp
import tempfile
import json
import asyncio
import sqlite3

# 引入力學引擎
from core.analyzer import PitchingAnalyzer

# ==========================================
# 模組一：伺服器基礎建設與環境設定
# ==========================================
app = FastAPI(title="⚾ AI 棒球投手分析 API")

# 設定 CORS：允許前端 (localhost:3000) 跨網域呼叫此後端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 開放靜態資料夾權限：讓前端可直接讀取/播放的影片
os.makedirs("output_videos", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="output_videos"), name="outputs")

os.makedirs("processed_videos", exist_ok=True) 
app.mount("/videos", StaticFiles(directory="processed_videos"), name="videos")


# 初始化力學演算法
analyzer = PitchingAnalyzer()

# ==========================================
# 模組二：API 端點與檔案接收暫存
# ==========================================
@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    in_path = ""
    out_path = ""
    
    try:
        # 將前端上傳的影片存入系統暫存區
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_in:
            shutil.copyfileobj(file.file, tfile_in)
            in_path = tfile_in.name

        # 設定畫完骨架後的影片格式 (轉為網頁相容性高的 webm)
        output_filename = f"processed_{file.filename}.webm"
        out_path = os.path.join("output_videos", output_filename)
        
    except Exception as e:
        return {"status": "error", "message": f"檔案上傳失敗: {str(e)}"}

    # ==========================================
    # 模組三：力學引擎運算
    # ==========================================
    async def event_generator():
        try:
            # 狀態回報
            yield json.dumps({"status": "processing", "progress": 2, "msg": "🚀 開始讀取影片..."}) + "\n"
            
            # 1. 讀取影片基礎資訊 (寬、高、FPS)
            cap = cv2.VideoCapture(in_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 2. 設定影片寫出器 (VP09 編碼)
            fourcc = cv2.VideoWriter_fourcc(*'vp09')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            
            pose_history = []
            frame_idx = 0
            
            # 3. MediaPipe 骨架擷取、繪製、轉檔
            with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break
                        
                    # 色彩轉換與骨架運算
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        # 繪製骨架
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                        lm = results.pose_landmarks.landmark
                        
                        # 定義關鍵關節
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

                    out.write(image)
                    frame_idx += 1
                    
                    # 每 3 幀回傳一次進度給前端 (更新進度條)
                    if frame_idx % 3 == 0 or frame_idx == total_frames:
                        current_progress = int((frame_idx / total_frames) * 90)
                        yield json.dumps({
                            "status": "processing", 
                            "progress": current_progress, 
                            "msg": f"骨架演算中 ({frame_idx}/{total_frames})"
                        }) + "\n"
                        await asyncio.sleep(0.001) 

            cap.release()
            out.release()

            # ==========================================
            # 模組四：使用者進階力學分析與診斷
            # ==========================================
            yield json.dumps({"status": "processing", "progress": 92, "msg": "⚙️ 骨架萃取完畢，進行力學診斷..."}) + "\n"
            await asyncio.sleep(0.001)

            # 呼叫引擎：找 F1~F5 關鍵幀與原始角度，並計算進階數據
            final_keys = analyzer.auto_detect_keyframes(pose_history)
            features = analyzer.calculate_biomechanics(pose_history, final_keys)

            yield json.dumps({"status": "processing", "progress": 95, "msg": "🔍 正在資料庫中搜尋最佳匹配模板..."}) + "\n"
            await asyncio.sleep(0.001)

            # 翻譯診斷報告 (使用者版 - 保留評價 ✅❌)
            report = []
            report.append("【時間軸與最終確認節點】")
            report.append(f"📍 N1: {final_keys.get('f1','-')} | N1.5: {final_keys.get('f1_5','-')} | N2: {final_keys.get('f2','-')} | N3: {final_keys.get('f3','-')} | N4: {final_keys.get('f4','-')} | N5: {final_keys.get('f5','-')}")
            report.append("-" * 65)

            # --- 下盤分析區塊 ---
            report.append("【💪 動能蓄積與轉移 (下盤分析)】")
            if -5 < features['lift_torso_tilt'] < 5: 
                report.append(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (✅ 重心完美，脊椎直立)")
            elif features['lift_torso_tilt'] < -5: 
                report.append(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (⚠️ 過度後仰，易導致放球點飄忽)")
            else: 
                report.append(f"🔹 抬腿平衡   : {features['lift_torso_tilt']} 度 (⚠️ 過度前傾，重心提早崩潰)")

            if features['drop_percentage'] > 8: 
                report.append(f"🔹 下盤驅動   : 下潛 {features['drop_percentage']}% (✅ 成功利用後腳推蹬 Drop & Drive)")
            else: 
                report.append(f"🔹 下盤驅動   : 下潛 {features['drop_percentage']}% (⚠️ 下潛不足，Tall & Fall 易流失動能)")

            if features['stride_percentage'] >= 85: 
                report.append(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (✅ 跨步極佳，完美延伸動力鏈)")
            elif 70 <= features['stride_percentage'] < 85: 
                report.append(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (⚠️ 跨步偏小，可能限制球威)")
            else: 
                report.append(f"🔹 跨步幅度   : 身體長度的 {features['stride_percentage']}% (❌ 跨步嚴重不足，導致放球點過早)")

            # --- 上盤分析區塊 (🌟 已將變數移入並展開) ---
            report.append("\n【⚾ 力量釋放與煞車 (上盤分析)】")
            
            # 1. 髖肩分離度
            sep_score = max(0, min(100, int((features['hip_shoulder_sep'] + 0.08) / 0.16 * 100)))
            report.append(f"🔹 髖肩分離度 : {sep_score} 分 / 100 分 (原始 Z 軸差: {features['hip_shoulder_sep']})")
            
            # 2. 手套手狀態
            if features['glove_arm_angle'] < 100:
                report.append(f"🔹 手套手狀態 : {features['glove_arm_angle']} 度 (✅ 完美鎖定)")
            elif features['glove_arm_angle'] < 130:
                report.append(f"🔹 手套手狀態 : {features['glove_arm_angle']} 度 (⚠️ 微幅外開)")
            else:
                report.append(f"🔹 手套手狀態 : {features['glove_arm_angle']} 度 (❌ 嚴重開掉 (力量流失))")
                
            # 3. 放球延伸度
            report.append(f"🔹 放球延伸度 : 往前延伸等同於 {features['extension_ratio']*100:.1f}% 的半身高度")
            
            # 4. 軀幹前傾角
            if features['release_trunk_tilt'] > 20:
                report.append(f"🔹 軀幹前傾角 : {features['release_trunk_tilt']} 度 (✅ 軀幹完美前壓，釋放力量)")
            else:
                report.append(f"🔹 軀幹前傾角 : {features['release_trunk_tilt']} 度 (❌ 直立放球，過度依賴手臂發力)")
                
            # 5. 前腳煞車角
            if features['bracing_angle'] > 150:
                report.append(f"🔹 前腳煞車角 : {features['bracing_angle']} 度 (✅ 完美支撐)")
            else:
                report.append(f"🔹 前腳煞車角 : {features['bracing_angle']} 度 (⚠️ 煞車不足 (膝蓋彎曲))")

            # --- 受傷風險區塊 ---
            report.append("\n【🚨 隱性受傷風險與關節壓力預警】")
            report.append(f"⚠️ 手臂延遲   : {'❌ 偵測到 Late Cocking (高風險)' if features['is_late_cocking'] else '✅ 舉起時機標準'}")
            
            if features['cocking_elbow_angle'] == 0:
                report.append(f"⚠️ 蓄力手肘角 : 無法成功偵測，動作可能被遮擋。")
            elif 85 < features['cocking_elbow_angle'] < 115:
                report.append(f"⚠️ 蓄力手肘角 : {features['cocking_elbow_angle']} 度 (✅ 位於安全範圍，韌帶壓力正常)")
            else:
                report.append(f"⚠️ 蓄力手肘角 : {features['cocking_elbow_angle']} 度 (❌ 角度過大/過小，UCL 承受極大撕裂張力！)")

            if features['shoulder_abduction_angle'] >= 85:
                report.append(f"⚠️ 放球手肘角 : {features['shoulder_abduction_angle']} 度 (✅ 手肘高度完美，肩膀分擔受力)")
            else:
                report.append(f"⚠️ 放球手肘角 : {features['shoulder_abduction_angle']} 度 (❌ 手肘下掉！推鉛球發力，極易傷及肩關節與 UCL)")

            real_report_text = "\n".join(report)

            yield json.dumps({"status": "processing", "progress": 98, "msg": "📝 正在產生專業分析報告..."}) + "\n"
            await asyncio.sleep(0.001)

            # ==========================================
            # 模組五：職業投手配對
            # ==========================================
            pro_mock_data = None
            db_path = "pitching_data.db"
            
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row 
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM pro_pitchers")
                    pros = cursor.fetchall()
                    
                    best_match = None
                    min_diff = float('inf')
                    
                    # 簡單配對演算法：比對下潛、跨步、前傾角特徵
                    for pro in pros:
                        diff = (
                            abs(features['drop_percentage'] - pro['drop_percentage']) * 1.0 +
                            abs(features['stride_percentage'] - pro['stride_percentage']) * 0.8 +
                            abs(features['release_trunk_tilt'] - pro['release_trunk_tilt']) * 1.2
                        )
                        if diff < min_diff:
                            min_diff = diff
                            best_match = pro
                    
                    if best_match:
                        # 職業投手的報告
                        pro_sep_score = max(0, min(100, int((best_match['hip_shoulder_sep'] + 0.08) / 0.16 * 100)))
                        
                        pro_report = []
                        pro_report.append("【📝 AI 系統配對原因】")
                        pro_report.append("這是一位與您「下盤機制」與「放球前傾角」最相似的職業投手！藉由觀察他的節奏，可以幫助您優化動力鏈。\n")
                        
                        pro_report.append("【⚾ 投手基本資訊】")
                        pro_report.append(f"🔹 慣用手   : {best_match['handedness']}")
                        pro_report.append(f"🔹 出手點   : {best_match['arm_slot']}\n")

                        pro_report.append("【時間軸與最終確認節點】")
                        pro_report.append(f"📍 N1: {best_match['f1']} | N1.5: {best_match['f1_5']} | N2: {best_match['f2']} | N3: {best_match['f3']} | N4: {best_match['f4']} | N5: {best_match['f5']}")
                        pro_report.append("-" * 65)

                        pro_report.append("【💪 動能蓄積與轉移 (下盤分析)】")
                        pro_report.append(f"🔹 抬腿平衡   : {best_match['lift_torso_tilt']} 度")
                        pro_report.append(f"🔹 下盤驅動   : 下潛 {best_match['drop_percentage']}%")
                        pro_report.append(f"🔹 跨步幅度   : 身體長度的 {best_match['stride_percentage']}%")

                        pro_report.append("\n【⚾ 力量釋放與煞車 (上盤分析)】")
                        pro_report.append(f"🔹 髖肩分離度 : {pro_sep_score} 分 / 100 分 (原始 Z 軸差: {best_match['hip_shoulder_sep']})")
                        pro_report.append(f"🔹 手套手狀態 : {best_match['glove_arm_angle']} 度")
                        pro_report.append(f"🔹 放球延伸度 : 往前延伸等同於 {best_match['extension_ratio']*100:.1f}% 的半身高度")
                        pro_report.append(f"🔹 軀幹前傾角 : {best_match['release_trunk_tilt']} 度")
                        pro_report.append(f"🔹 前腳煞車角 : {best_match['bracing_angle']} 度")

                        pro_report.append("\n【🚨 隱性受傷風險與關節壓力預警】")
                        pro_report.append(f"⚠️ 手臂延遲   : {'有' if best_match['is_late_cocking'] else '無'}")
                        pro_report.append(f"⚠️ 蓄力手肘角 : {best_match['cocking_elbow_angle']} 度")
                        pro_report.append(f"⚠️ 放球手肘角 : {best_match['shoulder_abduction_angle']} 度")

                        pro_mock_data = {
                            "name": best_match['pitcher_name'],
                            "video_url": f"http://localhost:8000/{best_match['video_filename']}",
                            "fps": best_match['fps'],
                            "total_frames": best_match['total_frames'],
                            "keyframes": {
                                "f1": best_match['f1'],
                                "f1_5": best_match['f1_5'],
                                "f2": best_match['f2'],
                                "f3": best_match['f3'],
                                "f4": best_match['f4'],
                                "f5": best_match['f5']
                            },
                            "report": "\n".join(pro_report) # 裝入組裝好的字串
                        }
                    conn.close()
                except Exception as db_err:
                    print(f"⚠️ 資料庫配對失敗: {db_err}")

            # ==========================================
            # 模組六：打包回傳與清理
            # ==========================================
            yield json.dumps({
                "status": "success",
                "progress": 100,
                "report": real_report_text,
                "video_url": f"http://localhost:8000/outputs/{output_filename}",
                "keyframes": final_keys, 
                "fps": fps,              
                "total_frames": total_frames,
                "pro_data": pro_mock_data 
            }) + "\n"
            
        except Exception as e:
            print(f"❌ [系統錯誤] {str(e)}")
            yield json.dumps({"status": "error", "message": f"分析失敗: {str(e)}"}) + "\n"
            
        finally:
            # 清理作業：刪除上傳的暫存原始影片，釋放伺服器空間
            if in_path and os.path.exists(in_path):
                try: os.remove(in_path)
                except Exception: pass

    # 以 StreamingResponse (NDJSON 格式) 啟動串流回傳
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")