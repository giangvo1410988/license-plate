import cv2
import numpy as np
import torch
import re
import collections
import function.helper as helper
import function.utils_rotate as utils_rotate
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import time
from ultralytics import YOLO 

FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLOV5_ROOT = ROOT / 'yolov5'
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

yolo_LP_detect = torch.hub.load('yolov5', 'custom',
                                path='model/LP_detector.pt',
                                force_reload=True, source='local').to('cuda')

yolo_license_plate = torch.hub.load('yolov5', 'custom',
                                    path='model/LP_ocr.pt',
                                    force_reload=True, source='local').to('cuda')


yolo_lp_track = YOLO('./checkpoint/yolo11n.pt').to('cuda')

def init_db():
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS registered_plates
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS plate_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    plate_number TEXT,
                    timestamp DATETIME,
                    type TEXT,
                    UNIQUE(track_id, plate_number)
                    )''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()
        
def init_temp_table():
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_plate_temp (
                track_id INTEGER,
                license_plate TEXT,
                count INTEGER,
                PRIMARY KEY (track_id, license_plate)
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error (init_temp_table): {e}")
    finally:
        conn.close()

def update_vehicle_plate_temp(track_id, license_plate):
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute("SELECT count FROM vehicle_plate_temp WHERE track_id = ? AND license_plate = ?", (track_id, license_plate))
        row = c.fetchone()
        if row:
            new_count = row[0] + 1
            c.execute("UPDATE vehicle_plate_temp SET count = ? WHERE track_id = ? AND license_plate = ?", (new_count, track_id, license_plate))
        else:
            c.execute("INSERT INTO vehicle_plate_temp (track_id, license_plate, count) VALUES (?, ?, ?)", (track_id, license_plate, 1))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error (update_vehicle_plate_temp): {e}")
    finally:
        conn.close()

def process_temp_table():
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute("""
            SELECT track_id, license_plate, MAX(count) 
            FROM vehicle_plate_temp 
            GROUP BY track_id
        """)
        rows = c.fetchall()
        for row in rows:
            track_id, license_plate, max_count = row
            registered = get_registered_plates()
            plate_type = "Registered" if (not registered.empty and license_plate in registered['plate_number'].values) else "New"
            add_plate_history(track_id, license_plate, plate_type)
        c.execute("DELETE FROM vehicle_plate_temp")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error (process_temp_table): {e}")
    finally:
        conn.close()

def add_registered_plate(plate_number):
    if not plate_number:
        return False
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute("INSERT INTO registered_plates (plate_number) VALUES (?)", (plate_number,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return False
    finally:
        conn.close()

def get_registered_plates():
    try:
        conn = sqlite3.connect('license_plate.db')
        df = pd.read_sql_query("SELECT * FROM registered_plates", conn)
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def delete_registered_plate(plate_number):
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute("DELETE FROM registered_plates WHERE plate_number = ?", (plate_number,))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

def add_plate_history(track_id, plate_number, plate_type):
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        # Kiểm tra nếu (track_id, plate_number) đã có trong bảng
        c.execute("SELECT 1 FROM plate_history WHERE track_id = ? AND plate_number = ?", (track_id, plate_number))
        if c.fetchone() is None:
            timestamp = datetime.now()
            c.execute("INSERT INTO plate_history (track_id, plate_number, timestamp, type) VALUES (?, ?, ?, ?)",
                      (track_id, plate_number, timestamp, plate_type))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()
        
def delete_plate_history():
    conn = None
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM plate_history")
        initial_count = c.fetchone()[0]
        
        c.execute("BEGIN TRANSACTION")
        
        c.execute("DELETE FROM plate_history")
        
        c.execute("DELETE FROM sqlite_sequence WHERE name='plate_history'")
        
        conn.commit()
        
        c.execute("SELECT COUNT(*) FROM plate_history")
        final_count = c.fetchone()[0]
        
        if final_count == 0:
            st.success(f"Successfully deleted {initial_count} records from history")
            return True
        else:
            st.warning("Delete operation completed but records may still exist")
            return False
            
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        if conn:
            conn.close() 


def get_plate_history():
    try:
        conn = sqlite3.connect('license_plate.db')
        df = pd.read_sql_query("""
            SELECT 
                ROW_NUMBER() OVER (ORDER BY timestamp DESC) as no,
                plate_number as 'License Plate',
                datetime(timestamp) as 'Time',
                type as 'Type'
            FROM plate_history
            ORDER BY timestamp DESC
        """, conn)
        if not df.empty:
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%d/%m/%Y %I:%M %p')
        return df
    except (sqlite3.Error, pd.Error) as e:
        st.error(f"Error getting history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def detect_license_plate(frame):
    polygon_points = np.array([(89, 607), (331, 344), (1296, 352), (1313, 641)], np.int32)
    output = []
    start_time = time.time()
    
    track_results = yolo_lp_track.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2])
    
    for result in track_results:
        boxes = result.boxes
        for box in boxes:
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:

                vehicle_roi = frame[y1:y2, x1:x2]
                

                lp_results = yolo_LP_detect(vehicle_roi, size=640)
                lp_detections = lp_results.pandas().xyxy[0].values.tolist()
                
                for det in lp_detections:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, det[:4])

                    lp_roi = vehicle_roi[lp_y1:lp_y2, lp_x1:lp_x2]
                    if lp_roi.size == 0:
                        continue


                    ocr_text = helper.read_plate(yolo_license_plate, lp_roi)
                    if ocr_text == "unknown":

                        flag = False
                        for cc in range(0, 2):
                            for ct in range(0, 2):
                                rotated = utils_rotate.deskew(lp_roi, cc, ct)
                                ocr_text = helper.read_plate(yolo_license_plate, rotated)
                                if ocr_text != "unknown":
                                    flag = True
                                    break
                            if flag:
                                break

                    if ocr_text != "unknown":
                        output.append({
                            "id": track_id,
                            "ocr": ocr_text,
                            "vehicle_bbox": [x1, y1, x2, y2],
                            "plate_bbox": [lp_x1, lp_y1, lp_x2, lp_y2]
                        })
    return output
