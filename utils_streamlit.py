import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
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

ocr_model = ocr = PaddleOCR(lang='en')


class LicensePlateTracker:
    def __init__(self):
        self.tracked_plates = []
        self.next_id = 0
        
    def update(self, detections):
        matched_tracks = set()
        
        # Process new detections
        for detection in detections:
            matched = False
            bbox = detection[:4]
            conf = detection[4]
            
            # Compare with existing tracks
            for track in self.tracked_plates:
                if self._iou(bbox, track['bbox']) > 0:  # IOU threshold
                    track['bbox'] = bbox
                    track['conf'] = conf
                    track['age'] = 0
                    matched_tracks.add(track['id'])
                    matched = True
                    break
            
            # Create new track if no match
            if not matched:
                self.tracked_plates.append({
                    'id': self.next_id,
                    'bbox': bbox,
                    'conf': conf,
                    'age': 0
                })
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Update age of unmatched tracks
        self.tracked_plates = [track for track in self.tracked_plates 
                             if track['id'] in matched_tracks or track['age'] < 5]
        
        # Increment age for unmatched tracks
        for track in self.tracked_plates:
            if track['id'] not in matched_tracks:
                track['age'] += 1
    
    def _iou(self, bbox1, bbox2):
        # Calculate intersection over union
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS registered_plates
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      plate_number TEXT UNIQUE)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS plate_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      plate_number TEXT,
                      timestamp DATETIME,
                      type TEXT)''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
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

def add_plate_history(plate_number, plate_type):
    try:
        conn = sqlite3.connect('license_plate.db')
        c = conn.cursor()
        timestamp = datetime.now()
        c.execute("INSERT INTO plate_history (plate_number, timestamp, type) VALUES (?, ?, ?)",
                  (plate_number, timestamp, plate_type))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
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
            # LIMIT 100
        if not df.empty:
            df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%d/%m/%Y %I:%M %p')
        return df
    except (sqlite3.Error, pd.Error) as e:
        st.error(f"Error getting history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def detect_license_plate(frame):
    """
    Xử lý 1 frame để phát hiện biển số, thực hiện OCR và cập nhật tracking.
    """
    polygon_points = np.array([(89, 607), (331, 344), (1296, 352), (1313, 641)], np.int32)
    output = []
    
    # Sử dụng tracker toàn cục được lưu trong session state
    start_time = time.time()
    
    tracker = st.session_state.lp_tracker
    
    results = yolo_LP_detect(frame, size=640)
    detections = results.pandas().xyxy[0].values.tolist()
    
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:
            filtered_detections.append(det)
    
    tracker.update(filtered_detections)
    
    for track in tracker.tracked_plates:
        bbox = track['bbox']
        track_id = track['id']
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        lp_result = helper.read_plate(yolo_license_plate, roi)
        
        if lp_result == "unknown":
            flag = False
            for cc in range(0, 2):
                for ct in range(0, 2):
                    rotated = utils_rotate.deskew(roi, cc, ct)
                    lp_result = helper.read_plate(yolo_license_plate, rotated)
                    if lp_result != "unknown":
                        flag = True
                        break
                if flag:
                    break
        
        if lp_result != "unknown":
            output.append({
                "id": track_id,
                "ocr": lp_result,
                "bbox": [x1, y1, x2, y2]
            })
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.3f}s")
    print("Memory Allocated:", torch.cuda.memory_allocated())
    print("Memory Reserved: ", torch.cuda.memory_reserved())
    
    return output

def format_plate_text(ocr_results):
    if ocr_results[0]:
        text_parts = []
        for detection in ocr_results[0]:
            text = detection[1][0] 
            filtered_text = re.sub(r'[^A-Za-z0-9]', '', text)
            if text.strip():
                text_parts.append(filtered_text)
        
        plate_text = "".join(text_parts)
        return plate_text
    return ""

# def detect_license_plate(frame):
#     """
#     Xử lý 1 frame để phát hiện biển số, thực hiện OCR sử dụng PaddleOCR và cập nhật tracking.
#     """
#     polygon_points = np.array([(89, 607), (331, 344), (1296, 352), (1313, 641)], np.int32)
#     output = []
    
#     start_time = time.time()
    
#     # Sử dụng tracker toàn cục được lưu trong session state
#     tracker = st.session_state.lp_tracker
    
#     # Phát hiện biển số trong frame bằng YOLO
#     results = yolo_LP_detect(frame, size=640)
#     detections = results.pandas().xyxy[0].values.tolist()
    
#     filtered_detections = []
#     for det in detections:
#         x1, y1, x2, y2 = map(int, det[:4])
#         center_x = (x1 + x2) // 2
#         center_y = (y1 + y2) // 2
#         if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:
#             filtered_detections.append(det)
    
#     # Cập nhật tracker với các detection đã được lọc
#     tracker.update(filtered_detections)
    
#     # Duyệt qua từng track để lấy ROI và thực hiện OCR
#     for track in tracker.tracked_plates:
#         bbox = track['bbox']
#         track_id = track['id']
#         x1, y1, x2, y2 = map(int, bbox)
#         roi = frame[y1:y2, x1:x2]
#         if roi.size == 0:
#             continue
        
#         # Sử dụng PaddleOCR để nhận dạng biển số trên ROI
#         ocr_results = ocr_model.ocr(roi, cls=True)
#         if ocr_results and len(ocr_results) > 0:
#             lp_text = ocr_results
#             lp_text = format_plate_text(ocr_results)
#         else:
#             lp_text = "unknown"
        
#         # Nếu PaddleOCR không nhận dạng được, thử xoay (deskew) ROI
#         if lp_text == "unknown":
#             flag = False
#             for cc in range(0, 2):
#                 for ct in range(0, 2):
#                     rotated = utils_rotate.deskew(roi, cc, ct)
#                     ocr_results_rot = ocr_model.ocr(rotated, cls=True)
#                     if ocr_results_rot and len(ocr_results_rot) > 0:
#                         lp_text = ocr_results_rot
#                         lp_text = format_plate_text(ocr_results)
#                         flag = True
#                         break
#                 if flag:
#                     break
        
#         # Nếu sau các bước, PaddleOCR đã nhận dạng được biển số, thêm vào output
#         if lp_text != "unknown":
#             output.append({
#                 "id": track_id,
#                 "ocr": lp_text,
#                 "bbox": [x1, y1, x2, y2]
#             })
    
#     end_time = time.time()
#     inference_time = end_time - start_time
#     print(f"Inference time: {inference_time:.3f}s")
#     print("Memory Allocated:", torch.cuda.memory_allocated())
#     print("Memory Reserved: ", torch.cuda.memory_reserved())
    
#     return output

