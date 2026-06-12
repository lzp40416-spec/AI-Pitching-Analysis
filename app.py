import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

st.title("⚾ AI 棒球投手投球機制分析系統")
st.write("請上傳投球影片，系統將進行全身骨架演算。")

uploaded_file = st.file_uploader("上傳投球影片 (支援 mp4, mov 格式)", type=["mp4", "mov"])

if uploaded_file is not None:
    # 🌟 關鍵修正 1：用 with 語法，確保寫入完畢後 Python 會立刻「放開(close)」檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_in:
        tfile_in.write(uploaded_file.read())
        in_path = tfile_in.name  # 把路徑存起來
        
    # 🌟 關鍵修正 2：先建好空檔案並取得路徑，然後立刻放開它
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tfile_out:
        out_path = tfile_out.name

    # 接下來全部改用 in_path 和 out_path
    cap = cv2.VideoCapture(in_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    st.info("🧠 AI 正在為每一格畫面繪製骨架，這需要一點時間，請稍候...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        current_frame = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
            out.write(image)
            
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"演算進度：{progress}% ({current_frame}/{total_frames} 格)")
            
    cap.release()
    out.release()
    st.success("✅ 骨架演算完成！請觀看下方分析影片。")
    
    # 讀取並播放在 out_path 的影片
    with open(out_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes, format='video/webm')
        
    # 🌟 關鍵修正 3：這一次 Python 已經沒有抓著檔案了，Windows 絕對會乖乖讓我們刪除！
    try:
        os.remove(in_path)
        os.remove(out_path)
        
        # 🕵️‍♂️ 深度確認：用 os.path.exists 檢查檔案「是不是真的消失了」
        if not os.path.exists(in_path) and not os.path.exists(out_path):
            # print() 會把這些字印在 VS Code 下方的黑畫面終端機裡，作為後台日誌！
            print("\n" + "="*40)
            print("✅ [系統後台日誌] 深度確認：刪除成功！")
            print(f"🗑️ 已拔除原始檔：{in_path}")
            print(f"🗑️ 已拔除骨架檔：{out_path}")
            print("="*40 + "\n")
        else:
            print("⚠️ [系統後台日誌] 警告：檔案似乎還有殘留，請檢查權限！")
            
        st.toast("🧹 暫存影片已從硬碟徹底清除！") 
    except Exception as e:
        # 如果真的有意外，這次我們把它顯示在網頁上，不要偷偷藏起來
        st.error(f"清理失敗，請手動檢查：{e}") 
        print(f"❌ [系統錯誤] 刪除暫存檔時發生例外狀況：{e}")