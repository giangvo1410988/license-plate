import streamlit as st
import cv2
from datetime import datetime
from PIL import Image
import queue
from utils_streamlit_cartrack import *
import time
import threading 

stop_event = threading.Event()

def init_session_state():
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()
    if 'added_plates' not in st.session_state:
        st.session_state.added_plates = []
    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=10)



init_session_state()


init_db()
init_temp_table()

st.set_page_config(layout="wide", page_title="License Plate Detection")


def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
        time.sleep(0.05)

def process_frame(frame):
    polygon_points = np.array([(89, 607), (331, 344), (1296, 352), (1313, 641)], np.int32)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected_data = detect_license_plate(frame_rgb)
    
    for data in detected_data:
        vehicle_bbox = data["vehicle_bbox"]
        plate_bbox = data["plate_bbox"]
        track_id = data["id"]
        ocr_text = data["ocr"]
        

        cv2.rectangle(frame_rgb, (vehicle_bbox[0], vehicle_bbox[1]),
                      (vehicle_bbox[2], vehicle_bbox[3]), (255, 255, 0), 2)

        cv2.rectangle(frame_rgb, (vehicle_bbox[0] + plate_bbox[0], vehicle_bbox[1] + plate_bbox[1]),
                      (vehicle_bbox[0] + plate_bbox[2], vehicle_bbox[1] + plate_bbox[3]), (0, 255, 0), 2)

        cv2.putText(frame_rgb, f"ID: {track_id} LP: {ocr_text}",
                    (vehicle_bbox[0], vehicle_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        update_vehicle_plate_temp(track_id, ocr_text)
    

    cv2.polylines(frame_rgb, [polygon_points.reshape((-1, 1, 2))],
                  isClosed=True, color=(0, 0, 255), thickness=2)
    
    return frame_rgb

def convert_frame_to_bytes(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.markdown("### AIVISION")
    st.markdown("contact us: contact@aivgroups.com")
    
    st.markdown("### Registered Plates")
    new_plate = st.text_input("Add new plate number")
    if st.button("Add Plate"):
        if add_registered_plate(new_plate):
            st.success(f"Added plate: {new_plate}")
        else:
            st.error("Plate already exists or invalid")
    
    registered_plates = get_registered_plates()
    if not registered_plates.empty:
        for _, row in registered_plates.iterrows():
            col1_1, col1_2 = st.columns([3, 1])
            with col1_1:
                st.write(row['plate_number'])
            with col1_2:
                if st.button("Delete", key=f"del_{row['plate_number']}"):
                    delete_registered_plate(row['plate_number'])
                    st.rerun()


with col2:
    st.markdown("### AI License Plate Detection")
    camera_placeholder = st.empty()
    

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        if st.button("Start Camera"):
            st.session_state.camera_active = True

            stop_event.clear()
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture('./test_image/test3.mp4')

            if 'reader_thread' not in st.session_state or st.session_state.get('reader_thread') is None or not st.session_state.reader_thread.is_alive():
                st.session_state.reader_thread = threading.Thread(
                    target=frame_reader, 
                    args=(st.session_state.cap, st.session_state.frame_queue, stop_event)
                )
                st.session_state.reader_thread.daemon = True
                st.session_state.reader_thread.start()
    with col2_2:
        if st.button("Stop Camera"):
            st.session_state.camera_active = False
            stop_event.set()
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            camera_placeholder.empty()
            st.rerun()


with col3:
    st.markdown("### Car In/Out History")

    if st.button("Clear History"):
        delete_plate_history()
        st.success("License history has been cleared!")
        st.rerun()
        
    table_placeholder = st.empty()

global_frame_count = 0

if st.session_state.camera_active:
    while st.session_state.camera_active:
        if not st.session_state.frame_queue.empty():
            frame = st.session_state.frame_queue.get()
            processed_frame = process_frame(frame)
            st.session_state.current_frame = processed_frame
            camera_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
        else:
            if st.session_state.current_frame is not None:
                camera_placeholder.image(st.session_state.current_frame, channels="RGB", use_container_width=True)
        

        global_frame_count += 1

        if global_frame_count % 10 == 0:
            process_temp_table()
            
        current_time = time.time()
        if current_time - st.session_state.last_update_time >= 0.5:
            table_data = get_plate_history()
            if table_data is not None:
                table_placeholder.dataframe(
                    table_data,
                    hide_index=True,
                    column_config={
                        "no": st.column_config.NumberColumn("No.", width="small"),
                        "License Plate": st.column_config.TextColumn("License Plate", width=None),
                        "Time": st.column_config.TextColumn("Time", width="medium"),
                        "Type": st.column_config.TextColumn("Type", width="medium"),
                    },
                    use_container_width=True
                )
            st.session_state.last_update_time = current_time
        
        time.sleep(0.01) 

if st.session_state.cap is not None and not st.session_state.camera_active:
    st.session_state.cap.release()
    st.session_state.cap = None
