import sqlite3
import os

class DBManager:
    """
    棒球投手數據資料庫管理器 (SQLite 版)
    負責建立與連線資料庫，並將影像辨識引擎算出的力學特徵存入。
    """
    def __init__(self, db_name="pitching_data.db"):
        # 確保資料庫檔案永遠建立在專案的根目錄，避免路徑錯亂找不到檔案
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.db_path = os.path.join(base_dir, db_name)
        
        # 建立資料庫連線與操作游標
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # 初始化：確保基礎表格存在，並檢查是否需要為舊資料庫升級新欄位
        self.create_tables()
        self._check_and_add_column("handedness", "TEXT")
        self._check_and_add_column("arm_slot", "TEXT")
        
        # 🌟 新增：擴建前端同步所需的 8 個欄位
        self._check_and_add_column("fps", "INTEGER")
        self._check_and_add_column("total_frames", "INTEGER")
        self._check_and_add_column("f1", "INTEGER")
        self._check_and_add_column("f1_5", "INTEGER")
        self._check_and_add_column("f2", "INTEGER")
        self._check_and_add_column("f3", "INTEGER")
        self._check_and_add_column("f4", "INTEGER")
        self._check_and_add_column("f5", "INTEGER")

    def _check_and_add_column(self, column_name, column_type="REAL"):
        """
        【自動升級工具】
        安全檢查欄位是否存在。如果未來想新增特徵，只要在 __init__ 呼叫此函數，
        它就會自動幫舊表格擴建新欄位，且不會破壞既有資料。
        """
        # 取得目前 pro_pitchers 表格的所有欄位名稱
        self.cursor.execute("PRAGMA table_info(pro_pitchers)")
        existing_columns = [col[1] for col in self.cursor.fetchall()]
        
        # 若發現新欄位不在現有表格中，則使用 ALTER TABLE 擴建在最右側
        if column_name not in existing_columns:
            self.cursor.execute(f"ALTER TABLE pro_pitchers ADD COLUMN {column_name} {column_type}")
            self.conn.commit()

    def create_tables(self):
        """
        【建立資料表 Schema】
        如果 pitching_data.db 檔案不存在或全新建立，此語法會負責把地基打好。
        (註：此處的欄位宣告順序已設定為包含分類標籤的完美順序)
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pro_pitchers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_name TEXT,
                handedness TEXT,         -- 慣用手 (Right/Left)
                arm_slot TEXT,           -- 出手點 (Overhand/Sidearm等)
                video_filename TEXT,
                
                -- 下盤動力鏈
                lift_torso_tilt REAL,
                drop_percentage REAL,
                stride_percentage REAL,
                
                -- 上盤與煞車
                hip_shoulder_sep REAL,
                glove_arm_angle REAL,
                extension_ratio REAL,
                release_trunk_tilt REAL,
                bracing_angle REAL,
                
                -- 受傷預警指標
                is_late_cocking INTEGER,
                cocking_elbow_angle REAL,
                shoulder_abduction_angle REAL,
                
                -- 系統管理
                is_verified BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    # 🌟 修改：參數多接收了 keyframes, fps, total_frames
    def insert_pro_data(self, name, handedness, arm_slot, filename, features, keyframes, fps, total_frames, is_verified=True):
        """
        【寫入投手特徵】
        接收整理好的分類資訊與 11 項力學特徵字典，以及前端對決模式所需的關鍵幀與影片資訊，
        依序安全地寫入 SQLite 資料庫。
        """
        try:
            self.cursor.execute('''
                INSERT INTO pro_pitchers (
                    pitcher_name, handedness, arm_slot, video_filename, 
                    lift_torso_tilt, drop_percentage, stride_percentage, 
                    hip_shoulder_sep, glove_arm_angle, extension_ratio, 
                    release_trunk_tilt, bracing_angle, is_late_cocking, 
                    cocking_elbow_angle, shoulder_abduction_angle, is_verified,
                    fps, total_frames, f1, f1_5, f2, f3, f4, f5
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, 
                handedness, 
                arm_slot, 
                filename, 
                features['lift_torso_tilt'], 
                features['drop_percentage'], 
                features['stride_percentage'], 
                features['hip_shoulder_sep'], 
                features['glove_arm_angle'], 
                features['extension_ratio'], 
                features['release_trunk_tilt'], 
                features['bracing_angle'], 
                features['is_late_cocking'], 
                features['cocking_elbow_angle'], 
                features['shoulder_abduction_angle'], 
                is_verified,
                # 🌟 把新收到的資料對應塞進去
                fps, 
                total_frames, 
                keyframes.get('f1', 0), 
                keyframes.get('f1_5', 0), 
                keyframes.get('f2', 0), 
                keyframes.get('f3', 0), 
                keyframes.get('f4', 0), 
                keyframes.get('f5', 0)
            ))
            self.conn.commit()
            print(f"💾 [資料庫] 成功寫入: {name} ({handedness} | {arm_slot})")
        except Exception as e:
            # 若發生欄位不匹配或資料型態錯誤，會在此處報錯攔截
            print(f"❌ [資料庫錯誤] 寫入失敗: {e}")

    def close(self):
        """【關閉連線】執行完畢後釋放系統資源"""
        self.conn.close()

# --- 獨立測試區塊 ---
# 當直接執行此檔案 (python database/db_manager.py) 時才會觸發
if __name__ == "__main__":
    print("啟動資料庫測試...")
    db = DBManager()
    print(f"✅ 資料庫已成功連線並建立於: {db.db_path}")
    db.close()