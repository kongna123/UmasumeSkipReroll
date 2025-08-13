# coding=utf-8
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import os
import sys
import shutil
import threading
import time
import ast

# --- ทดสอบนำเข้าไลบรารีที่จำเป็นและแจ้งผู้ใช้หากขาดหาย ---
def show_missing_dependencies_message(missing_libs):
    """แสดงข้อความข้อผิดพลาดเมื่อพบไลบรารีที่ขาดหาย"""
    message = "พบว่าไลบรารีต่อไปนี้ยังไม่ได้ติดตั้ง:\n"
    message += "\n".join(missing_libs)
    message += "\n\nกรุณาติดตั้งโดยใช้คำสั่งใน Command Prompt หรือ Terminal:\n"
    message += f"pip install {' '.join(missing_libs)}\n"
    message += "หรือใช้ไฟล์ requirements.txt ด้วยคำสั่ง:\n"
    message += "pip install -r requirements.txt"
    messagebox.showerror("ข้อผิดพลาดในการโหลดไลบรารี", message)
    sys.exit(1)

# ตรวจสอบไลบรารี
missing_libs = []
try:
    from PIL import Image, ImageTk
except ImportError:
    missing_libs.append("Pillow")
try:
    import cv2
except ImportError:
    missing_libs.append("opencv-python")
try:
    import mss
except ImportError:
    missing_libs.append("mss")
try:
    import numpy as np
except ImportError:
    missing_libs.append("numpy")
try:
    import pyautogui
except ImportError:
    missing_libs.append("pyautogui")
try:
    from pynput import mouse, keyboard
except ImportError:
    missing_libs.append("pynput")

if missing_libs:
    show_missing_dependencies_message(missing_libs)


# --- ฟังก์ชันแก้ไขเส้นทางสำหรับ PyInstaller ---
def resource_path(relative_path):
    """
    รับเส้นทางสัมพัทธ์ของไฟล์และคืนค่าเป็นเส้นทางที่ถูกต้อง
    ทั้งในโหมดพัฒนาและเมื่อถูกแปลงเป็น .exe ด้วย PyInstaller
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- โฟลเดอร์สำหรับจัดเก็บรูปภาพและไฟล์บันทึก ---
IMAGE_FOLDER = resource_path("image")
RECORDING_FOLDER = resource_path("recordings")
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
if not os.path.exists(RECORDING_FOLDER):
    os.makedirs(RECORDING_FOLDER)


# ====================================================================
# --- ตัวแปรและฟังก์ชันสำหรับส่วน Image Counter (ตรวจจับการ์ด) ---
# ====================================================================

umasu_stop_event = threading.Event()
umasu_thread = None
images_config = []

try:
    with open(resource_path("config.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # เพิ่มการตรวจสอบบรรทัดว่าง
                name, required = line.strip().split(',', 1)
                images_config.append({
                    'name': name.strip(), 
                    'required': int(required.strip()), 
                    'found': 0, 
                    'frame': None, 
                    'photo': None, 
                    'label': None, 
                    'entry': None,
                    'last_found_time': 0
                })
except FileNotFoundError:
    pass
except Exception as e:
    messagebox.showwarning("ข้อผิดพลาดในการอ่านไฟล์", f"ไม่สามารถอ่าน config.txt ได้: {e}")

reset_image_name = 'restart.png'
# จำนวนการจับคู่จุดเด่นขั้นต่ำที่ต้องพบเพื่อยืนยันว่าภาพถูกต้อง
MIN_MATCH_COUNT = 15
# สร้าง ORB object ที่นี่เพื่อให้แน่ใจว่ามันถูกสร้างขึ้นเพียงครั้งเดียว
orb = cv2.ORB_create()
templates_features = {} # เก็บ Keypoints และ Descriptors ของภาพต้นแบบ
reset_template = None

def save_config():
    """บันทึกการตั้งค่าการ์ดลงในไฟล์"""
    with open(resource_path("config.txt"), "w", encoding="utf-8") as f:
        for img_config in images_config:
            f.write(f"{img_config['name']},{img_config['required']}\n")

def load_templates():
    """โหลดและเตรียมภาพต้นแบบทั้งหมด พร้อมคำนวณ Keypoints/Descriptors"""
    global templates_features, reset_template
    templates_features.clear()
    
    for img_config in images_config:
        full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
        if os.path.exists(full_path):
            template_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                # คำนวณ Keypoints และ Descriptors ของภาพต้นแบบ
                kp, des = orb.detectAndCompute(template_img, None)
                if kp is not None and des is not None:
                    templates_features[img_config['name']] = {'kp': kp, 'des': des}
                else:
                    print(f"คำเตือน: ไม่พบจุดเด่นในภาพ '{full_path}'")
            else:
                print(f"ข้อผิดพลาด: ไม่สามารถอ่านไฟล์รูปภาพ '{full_path}'")
        else:
            print(f"ข้อผิดพลาด: ไม่พบไฟล์รูปภาพ '{full_path}'")

    reset_path = os.path.join(IMAGE_FOLDER, reset_image_name)
    if os.path.exists(reset_path):
        reset_template = cv2.imread(reset_path, cv2.IMREAD_GRAYSCALE)
    else:
        # ไม่จำเป็นต้องมีไฟล์นี้ แต่จะแสดงข้อผิดพลาดถ้าไม่มี
        pass

def find_image_on_screen_robust(template_keypoints, template_descriptors):
    """
    ใช้ ORB feature matching เพื่อหาภาพต้นแบบบนหน้าจออย่างแม่นยำ
    โดยใช้เทคนิค findHomography และ RANSAC เพื่อกรองผลลัพธ์
    """
    sct = mss.mss()
    screen_shot = sct.grab(sct.monitors[0])
    screen_np = np.array(screen_shot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)

    # คำนวณ Keypoints และ Descriptors ของหน้าจอ
    screen_keypoints, screen_descriptors = orb.detectAndCompute(screen_gray, None)

    if screen_descriptors is None or len(screen_descriptors) < MIN_MATCH_COUNT:
        return 0
    
    # ใช้ Brute-Force Matcher เพื่อหาคู่ของ Descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(template_descriptors, screen_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # กรองการจับคู่ที่ดีที่สุด
    good_matches = matches[:50]

    if len(good_matches) >= MIN_MATCH_COUNT:
        # สร้างจุดอ้างอิงจากภาพต้นแบบและหน้าจอ
        src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([screen_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # ใช้ Homography เพื่อหาว่าภาพต้นแบบปรากฏบนหน้าจอหรือไม่
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            matches_mask = mask.ravel().tolist()
            inliers = sum(matches_mask) # นับจำนวนการจับคู่ที่ถูกต้อง
            
            if inliers >= MIN_MATCH_COUNT:
                return 1 # พบการ์ด
    
    return 0 # ไม่พบการ์ด

def run_umasu_main_loop():
    """ฟังก์ชันหลักที่ทำงานใน Background Thread"""
    while not umasu_stop_event.is_set():
        if reset_template is not None:
            # ใช้ Template Matching แบบธรรมดาสำหรับปุ่มรีเซ็ต (ซึ่งไม่ควรมีการเปลี่ยนแปลง)
            res = cv2.matchTemplate(cv2.cvtColor(np.array(mss.mss().grab(mss.mss().monitors[0])), cv2.COLOR_BGRA2GRAY), reset_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.8: # ใช้เกณฑ์ความแม่นยำสูงสำหรับปุ่มรีเซ็ต
                root.after(0, lambda: umasu_status_label.config(text="สถานะ: กำลังรีเซ็ต...", style="Warning.TLabel"))
                for img_config in images_config:
                    img_config['found'] = 0
                root.after(0, update_umasu_gui)
                time.sleep(1)
                continue
        
        all_conditions_met = True
        
        for img_config in images_config:
            try:
                img_config['required'] = int(img_config['entry'].get())
            except ValueError:
                img_config['required'] = 0

            if img_config['found'] < img_config['required']:
                template_features = templates_features.get(img_config['name'])
                if template_features is not None:
                    # ใช้ฟังก์ชันการตรวจจับภาพแบบใหม่ที่แม่นยำกว่า
                    current_found = find_image_on_screen_robust(template_features['kp'], template_features['des'])
                    
                    if current_found > 0:
                        current_time = time.time()
                        if current_time - img_config['last_found_time'] > 2.0:
                            img_config['found'] += current_found
                            img_config['last_found_time'] = current_time
            
            if img_config['found'] < img_config['required']:
                all_conditions_met = False
        
        root.after(100, update_umasu_gui)
        
        if all_conditions_met and images_config:
            root.after(0, lambda: umasu_status_label.config(text="สถานะ: พบครบแล้ว!", style="Success.TLabel"))
            umasu_stop_event.set()  
            root.after(0, update_umasu_status_after_stop)
            break
        
        time.sleep(0.5)
    
    root.after(100, update_umasu_gui)
    root.after(100, update_umasu_status_after_stop)

def update_umasu_gui():
    """อัปเดตค่าในหน้าต่าง GUI"""
    for img_config in images_config:
        if img_config.get('label'):
            img_config['label'].config(text=f"พบแล้ว: {img_config['found']}/{img_config['required']}")
            if img_config['found'] >= img_config['required']:
                img_config['label'].config(style="CardText.Success.TLabel")
            else:
                img_config['label'].config(style="CardText.TLabel")

def update_umasu_status_after_stop():
    """อัปเดตสถานะเมื่อโปรแกรมหยุดการทำงาน"""
    umasu_status_label.config(text="สถานะ: หยุดการทำงาน", style="Default.TLabel")
    umasu_start_button.config(state=tk.NORMAL)
    umasu_stop_button.config(state=tk.DISABLED)
    save_config()

def start_umasu_program():
    """เริ่มการทำงานของโปรแกรม Image Counter"""
    global umasu_thread
    if umasu_thread is None or not umasu_thread.is_alive():
        for img_config in images_config:
            img_config['found'] = 0
            img_config['last_found_time'] = 0
        
        if not images_config:
            messagebox.showwarning("คำเตือน", "กรุณาเพิ่มการ์ดที่ต้องการค้นหาก่อน")
            return

        umasu_stop_event.clear()
        umasu_thread = threading.Thread(target=run_umasu_main_loop)
        umasu_thread.daemon = True
        umasu_thread.start()
        umasu_status_label.config(text="สถานะ: กำลังค้นหา...", style="Info.TLabel")
        umasu_start_button.config(state=tk.DISABLED)
        umasu_stop_button.config(state=tk.NORMAL)

def stop_umasu_program():
    """หยุดการทำงานของโปรแกรม Image Counter"""
    umasu_stop_event.set()
    update_umasu_status_after_stop()

def upload_card():
    """เปิดหน้าต่างสำหรับอัปโหลดรูปภาพใหม่"""
    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์รูปภาพการ์ด",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not file_path:
        return

    file_name = os.path.basename(file_path)
    destination_path = os.path.join(IMAGE_FOLDER, file_name)

    if os.path.exists(destination_path):
        response = messagebox.askyesno("ไฟล์ซ้ำ", f"ไฟล์ '{file_name}' มีอยู่แล้ว ต้องการเขียนทับหรือไม่?")
        if not response:
            return
            
    try:
        shutil.copy(file_path, destination_path)
        add_card_to_gui(file_name, 1)
        messagebox.showinfo("สำเร็จ", f"อัปโหลด '{file_name}' เรียบร้อยแล้ว")
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถอัปโหลดไฟล์ได้: {e}")

def add_card_to_gui(file_name, required_count):
    """เพิ่มการ์ดใหม่ลงในโครงสร้างข้อมูลและสร้าง widget ใน GUI"""
    if any(config['name'] == file_name for config in images_config):
        return
        
    new_card_config = {
        'name': file_name,
        'required': required_count,
        'found': 0,
        'frame': None,
        'photo': None,
        'label': None,
        'entry': None,
        'last_found_time': 0
    }
    images_config.append(new_card_config)
    render_images_frame()
    load_templates()
    save_config()

def remove_card(card_config_to_remove):
    """ลบการ์ดที่เลือกออกจากรายการและ GUI"""
    response = messagebox.askyesno("ยืนยันการลบ", f"คุณต้องการลบการ์ด '{card_config_to_remove['name']}' ใช่หรือไม่?\nไฟล์รูปภาพจะถูกลบไปด้วย")
    if not response:
        return

    if card_config_to_remove in images_config:
        file_to_remove = os.path.join(IMAGE_FOLDER, card_config_to_remove['name'])
        if os.path.exists(file_to_remove):
            try:
                os.remove(file_to_remove)
            except Exception as e:
                messagebox.showwarning("คำเตือน", f"ไม่สามารถลบไฟล์ '{card_config_to_remove['name']}' ได้: {e}")

        images_config.remove(card_config_to_remove)
        card_config_to_remove['frame'].destroy()
        render_images_frame()
        load_templates()
        save_config()

def render_images_frame():
    """ล้างและสร้าง Frame การ์ดใหม่ทั้งหมด"""
    for widget in umasu_images_frame.winfo_children():
        widget.destroy()

    columns = 4
    for i, img_config in enumerate(images_config):
        frame = ttk.Frame(umasu_images_frame, style="Card.TFrame")
        frame.grid(row=i//columns, column=i%columns, padx=10, pady=10, sticky="nsew")
        img_config['frame'] = frame
        
        full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
        try:
            with Image.open(full_path) as pil_image:
                pil_image = pil_image.resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                img_config['photo'] = photo
            
            img_label = ttk.Label(frame, image=photo, style="Card.TLabel")
            img_label.pack(pady=(10, 5))
        except (FileNotFoundError, IOError):
            img_label = ttk.Label(frame, text=f"ไม่พบไฟล์\n{img_config['name']}", style="Error.TLabel", anchor="center", justify="center")
            img_label.pack(pady=(10, 5), padx=10, fill="both", expand=True)
            
        count_frame = ttk.Frame(frame, style="Card.Inner.TFrame")
        count_frame.pack(pady=5, padx=10)
        
        ttk.Label(count_frame, text="ต้องการ:", style="CardText.TLabel").pack(side=tk.LEFT, padx=(5, 2))
        
        required_var = tk.StringVar(value=str(img_config['required']))
        required_entry = ttk.Entry(count_frame, width=5, textvariable=required_var, justify="center")
        required_entry.pack(side=tk.LEFT, padx=(0, 5))
        img_config['entry'] = required_entry
        
        text_label = ttk.Label(frame, text=f"พบแล้ว: {img_config['found']}/{img_config['required']}", style="CardText.TLabel")
        text_label.pack(pady=5)
        img_config['label'] = text_label
        
        remove_button = ttk.Button(frame, text="ลบ", command=lambda config=img_config: remove_card(config), style="Danger.TButton")
        remove_button.pack(pady=(5,10), padx=10, fill=tk.X)
    
    if not images_config:
        empty_label = ttk.Label(umasu_images_frame, text="ยังไม่มีการ์ด\nคลิก 'อัปโหลดการ์ด' เพื่อเริ่มต้น", font=FONT_TITLE, style="Placeholder.TLabel", justify="center")
        empty_label.pack(expand=True, padx=20, pady=20)
    
    for i in range((len(images_config) + columns - 1) // columns):
        umasu_images_frame.grid_rowconfigure(i, weight=1)
    for i in range(columns):
        umasu_images_frame.grid_columnconfigure(i, weight=1)

# ====================================================================
# --- ตัวแปรและฟังก์ชันสำหรับส่วน Macro Recorder (บันทึกการกระทำ) ---
# ====================================================================

actions = []
is_recording = False
is_replaying = False
mouse_listener = None
keyboard_listener = None
replay_thread = None
recording_file = None
countdown_id = None
replay_stop_event = threading.Event()

MOUSE_MOVE_DURATION = 0.05
REPLAY_SPEED_FACTOR = 1.0

def on_press(key):
    """Callback เมื่อมีการกดปุ่มบนคีย์บอร์ด (ใช้สำหรับการบันทึก)"""
    global is_recording, actions, start_time
    if is_recording:
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).split('.')[-1]
        
        elapsed_time = time.time() - start_time
        actions.append({'type': 'key', 'key': key_name, 'time': elapsed_time})

def on_click(x, y, button, pressed):
    """Callback เมื่อมีการคลิกเมาส์ (ใช้สำหรับการบันทึก)"""
    global is_recording, actions, start_time
    if is_recording and pressed:
        elapsed_time = time.time() - start_time
        actions.append({'type': 'click', 'x': x, 'y': y, 'button': str(button).split('.')[-1], 'time': elapsed_time})

def on_move(x, y):
    """Callback เมื่อมีการเคลื่อนเมาส์ (ใช้สำหรับการบันทึก)"""
    global is_recording, actions, start_time
    if is_recording:
        elapsed_time = time.time() - start_time
        if not actions or actions[-1].get('x') != x or actions[-1].get('y') != y:
            actions.append({'type': 'move', 'x': x, 'y': y, 'time': elapsed_time})

def recording_countdown_tick(seconds_left):
    """ฟังก์ชันนับถอยหลังสำหรับการบันทึก"""
    global countdown_id, is_recording, actions, mouse_listener, keyboard_listener, start_time
    if not is_recording:
        return
        
    if seconds_left > 0:
        macro_countdown_label.config(text=f"เริ่มบันทึกใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, recording_countdown_tick, seconds_left - 1)
    else:
        macro_countdown_label.config(text=" ")
        macro_status_label.config(text="🔴 กำลังบันทึก", style="Status.Recording.TLabel")
        
        actions = []
        start_time = time.time()
        
        mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        mouse_listener.start()

        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        print(">>> เริ่มการบันทึกเมาส์และคีย์บอร์ด...")

def start_recording():
    """ฟังก์ชันสำหรับการเริ่มบันทึกจากปุ่ม GUI และคีย์ลัด"""
    global is_recording, countdown_id
    if is_recording:
        print(">>> การบันทึกกำลังดำเนินอยู่...")
        return
    is_recording = True
    update_ui_for_recording()
    recording_countdown_tick(5)
    print(">>> เตรียมบันทึกเมาส์และคีย์บอร์ด...")

def stop_recording():
    """ฟังก์ชันสำหรับการหยุดบันทึกจากปุ่ม GUI และคีย์ลัด"""
    global is_recording, mouse_listener, keyboard_listener, recording_file, countdown_id
    if not is_recording:
        print(">>> การบันทึกไม่ได้เริ่มทำงาน...")
        return

    is_recording = False
    if countdown_id:
        root.after_cancel(countdown_id)
        countdown_id = None
        
    if mouse_listener:
        mouse_listener.stop()
        mouse_listener = None
    if keyboard_listener:
        keyboard_listener.stop()
        keyboard_listener = None
    
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialdir=RECORDING_FOLDER,
        title="เลือกที่สำหรับบันทึกการกระทำ"
    )

    if filename:
        recording_file = filename
        with open(recording_file, 'w') as f:
            for action in actions:
                f.write(str(action) + '\n')
        
        root.after(0, lambda: messagebox.showinfo("บันทึกเสร็จสิ้น", f"บันทึกข้อมูลการกระทำ {len(actions)} ครั้ง ลงในไฟล์\n{recording_file}"))
    else:
        root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การบันทึกถูกยกเลิก ไม่มีการบันทึกไฟล์"))
    
    update_ui_for_idle()
    print(">>> หยุดการบันทึก")

def replay_countdown_tick(seconds_left, filename):
    """ฟังก์ชันนับถอยหลังสำหรับการเล่นซ้ำ"""
    global countdown_id, is_replaying
    if not is_replaying:
        return
        
    if seconds_left > 0:
        macro_countdown_label.config(text=f"เริ่มเล่นซ้ำใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, replay_countdown_tick, seconds_left - 1, filename)
    else:
        macro_countdown_label.config(text=" ")
        replay_thread = threading.Thread(target=run_replay, args=(filename,))
        replay_thread.start()

def run_replay(filename):
    """ฟังก์ชันสำหรับเล่นซ้ำการกระทำทั้งหมด (แบบวนลูป)"""
    global is_replaying, recording_file, replay_stop_event
    if not is_replaying:
        return

    recording_file = filename
    try:
        with open(recording_file, 'r') as f:
            saved_actions = [ast.literal_eval(line.strip()) for line in f]
    except FileNotFoundError:
        root.after(0, lambda: messagebox.showerror("ไฟล์ไม่พบ", f"ไม่พบไฟล์ {recording_file}"))
        is_replaying = False
        root.after(0, update_ui_for_idle)
        return
        
    pyautogui.FAILSAFE = False
    original_pause = pyautogui.PAUSE
    pyautogui.PAUSE = 0
    
    root.after(0, lambda: macro_status_label.config(text="▶️ กำลังเล่นซ้ำ", style="Status.Replaying.TLabel"))
    
    while is_replaying and not replay_stop_event.is_set():
        last_time = 0
        for action in saved_actions:
            if replay_stop_event.is_set():
                break
            
            time_to_sleep = action.get('time', 0) - last_time
            if time_to_sleep > 0:
                time.sleep(time_to_sleep / REPLAY_SPEED_FACTOR)
            
            if action['type'] == 'move':
                pyautogui.moveTo(action['x'], action['y'], duration=MOUSE_MOVE_DURATION)
            elif action['type'] == 'click':
                if action['button'] == 'left':
                    pyautogui.click(action['x'], action['y'])
                elif action['button'] == 'right':
                    pyautogui.rightClick(action['x'], action['y'])
            elif action['type'] == 'key':
                try:
                    pyautogui.press(action['key'])
                except Exception as e:
                    print(f"Failed to press key {action['key']}: {e}")
            
            last_time = action.get('time', 0)
    
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = original_pause
    is_replaying = False
    replay_stop_event.clear()
    root.after(0, update_ui_for_idle)
    print(">>> เล่นซ้ำเสร็จสมบูรณ์แล้ว")

def start_replay_with_dialog():
    """ฟังก์ชันที่ถูกเรียกจากปุ่มหรือคีย์ลัดเพื่อจัดการการเลือกไฟล์"""
    global is_replaying
    if is_replaying:
        return

    filename = filedialog.askopenfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialdir=RECORDING_FOLDER,
        title="เลือกไฟล์การกระทำที่ต้องการเล่นซ้ำ"
    )

    if not filename:
        root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การเล่นซ้ำถูกยกเลิก ไม่มีการเลือกไฟล์"))
        return
    
    is_replaying = True
    update_ui_for_replaying()
    
    replay_countdown_tick(5, filename)
    print(f">>> เตรียมเล่นซ้ำเมาส์จากไฟล์: {filename} ...")

def stop_replay_action():
    """หยุดการเล่นซ้ำ"""
    global is_replaying, replay_stop_event
    if is_replaying:
        is_replaying = False
        replay_stop_event.set()
        update_ui_for_idle()
        print(">>> หยุดการเล่นซ้ำ")

def cancel_action():
    """ยกเลิกการทำงานปัจจุบันและปิดโปรแกรม"""
    global is_recording, is_replaying, hotkey_listener, countdown_id
    if is_recording:
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
        stop_recording()
    if is_replaying:
        stop_replay_action()
    
    if hotkey_listener:
        hotkey_listener.stop()
        
    root.destroy()
    print(">>> ปิดโปรแกรม")

# ฟังก์ชันจัดการคีย์ลัดให้ Thread-safe
def hotkey_start_recording(): root.after(0, start_recording)
def hotkey_stop_recording(): root.after(0, stop_recording)
def hotkey_start_replay(): root.after(0, start_replay_with_dialog)
def hotkey_stop_replay(): root.after(0, stop_replay_action)
def hotkey_cancel_action(): root.after(0, cancel_action)

def update_ui_for_recording():
    macro_record_button.config(state=tk.DISABLED)
    macro_replay_button.config(state=tk.DISABLED)
    macro_status_label.config(text=" ", style="Status.Recording.TLabel")
    macro_countdown_label.config(text="เตรียมบันทึก...", style="Countdown.TLabel")

def update_ui_for_replaying():
    macro_record_button.config(state=tk.DISABLED)
    macro_replay_button.config(state=tk.DISABLED)
    macro_status_label.config(text=" ", style="Status.Replaying.TLabel")
    macro_countdown_label.config(text="เตรียมเล่นซ้ำ...", style="Countdown.TLabel")

def update_ui_for_idle():
    macro_record_button.config(state=tk.NORMAL)
    macro_replay_button.config(state=tk.NORMAL)
    macro_status_label.config(text="✅ พร้อมใช้งาน", style="Status.Idle.TLabel")
    macro_countdown_label.config(text="", style="Countdown.TLabel")


# ====================================================================
# --- การสร้างหน้าต่างหลักและ UI ---
# ====================================================================

# --- Color Palette & Fonts ---
COLOR_PRIMARY = "#0078D4"
COLOR_PRIMARY_HOVER = "#005A9E"
COLOR_DANGER = "#D32F2F"
COLOR_DANGER_HOVER = "#C62828"
COLOR_SUCCESS = "#2E7D32"
COLOR_WARNING = "#FF8F00"
COLOR_INFO = "#0288D1"
COLOR_BG = "#F0F0F0" # Background
COLOR_FRAME_BG = "#FFFFFF"
COLOR_TEXT = "#212121"
COLOR_TEXT_SECONDARY = "#757575"

FONT_FAMILY = "Tahoma"
FONT_NORMAL = (FONT_FAMILY, 10)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_STATUS = (FONT_FAMILY, 12, "bold")
FONT_NOTE = (FONT_FAMILY, 9)

def setup_styles(root):
    """ตั้งค่า Theme และ Style ทั้งหมดของ ttk"""
    style = ttk.Style(root)
    style.theme_use('clam')

    # General Styles
    style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_NORMAL)
    style.configure("TFrame", background=COLOR_BG)
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_NORMAL)
    style.configure("TEntry", fieldbackground=COLOR_FRAME_BG, font=FONT_NORMAL)

    # Notebook Style
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=FONT_BOLD, padding=[10, 5], background=COLOR_BG, borderwidth=0)
    style.map("TNotebook.Tab", 
              background=[("selected", COLOR_FRAME_BG), ("!selected", COLOR_BG)],
              foreground=[("selected", COLOR_PRIMARY), ("!selected", COLOR_TEXT_SECONDARY)])

    # Button Styles
    style.configure("TButton", font=FONT_BOLD, padding=[10, 5], borderwidth=1, relief="flat", background=COLOR_PRIMARY, foreground=COLOR_FRAME_BG)
    style.map("TButton",
        background=[('pressed', '!disabled', COLOR_PRIMARY_HOVER), ('active', COLOR_PRIMARY_HOVER)],
        relief=[('pressed', 'sunken')]
    )
    style.configure("Danger.TButton", background=COLOR_DANGER, foreground=COLOR_FRAME_BG)
    style.map("Danger.TButton",
        background=[('pressed', '!disabled', COLOR_DANGER_HOVER), ('active', COLOR_DANGER_HOVER)],
    )
    style.configure("Secondary.TButton", background=COLOR_TEXT_SECONDARY, foreground=COLOR_FRAME_BG)
    style.map("Secondary.TButton",
        background=[('pressed', '!disabled', "#616161"), ('active', "#616161")],
    )

    # Status Label Styles
    style.configure("Default.TLabel", foreground=COLOR_TEXT_SECONDARY, font=FONT_BOLD)
    style.configure("Info.TLabel", foreground=COLOR_INFO, font=FONT_BOLD)
    style.configure("Success.TLabel", foreground=COLOR_SUCCESS, font=FONT_BOLD)
    style.configure("Warning.TLabel", foreground=COLOR_WARNING, font=FONT_BOLD)
    style.configure("Error.TLabel", foreground=COLOR_DANGER, font=FONT_NORMAL)
    style.configure("Placeholder.TLabel", foreground=COLOR_TEXT_SECONDARY, font=FONT_BOLD)

    # Card Styles
    style.configure("Card.TFrame", background=COLOR_FRAME_BG, relief="solid", borderwidth=1, bordercolor="#E0E0E0")
    style.configure("Card.TLabel", background=COLOR_FRAME_BG)
    style.configure("Card.Inner.TFrame", background=COLOR_FRAME_BG)
    style.configure("CardText.TLabel", background=COLOR_FRAME_BG, foreground=COLOR_TEXT, font=FONT_BOLD)
    style.configure("CardText.Success.TLabel", background=COLOR_FRAME_BG, foreground=COLOR_SUCCESS, font=FONT_BOLD)
    
    # Macro Status Styles
    style.configure("Status.Idle.TLabel", foreground=COLOR_SUCCESS, font=FONT_STATUS)
    style.configure("Status.Recording.TLabel", foreground=COLOR_DANGER, font=FONT_STATUS)
    style.configure("Status.Replaying.TLabel", foreground=COLOR_INFO, font=FONT_STATUS)
    style.configure("Countdown.TLabel", foreground=COLOR_TEXT_SECONDARY, font=FONT_STATUS)
    style.configure("Hotkey.TLabel", foreground=COLOR_TEXT_SECONDARY, font=FONT_NOTE)


def create_image_counter_tab(notebook):
    """สร้าง UI สำหรับแท็บ Image Counter"""
    global umasu_frame, umasu_header_frame, umasu_status_label, umasu_container_frame
    global umasu_canvas, umasu_scrollbar, umasu_images_frame, umasu_controls_frame
    global umasu_start_button, umasu_stop_button, umasu_upload_button

    umasu_frame = ttk.Frame(notebook, padding="10")
    umasu_frame.pack(fill="both", expand=True)
    notebook.add(umasu_frame, text="  ตรวจจับการ์ด  ")

    # Header Frame
    umasu_header_frame = ttk.Frame(umasu_frame, padding="10")
    umasu_header_frame.pack(fill=tk.X)
    ttk.Label(umasu_header_frame, text="Image Counter", font=FONT_TITLE).pack(side=tk.LEFT)
    umasu_status_label = ttk.Label(umasu_header_frame, text="สถานะ: หยุดการทำงาน", style="Default.TLabel")
    umasu_status_label.pack(side=tk.RIGHT, pady=5)

    # Scrollable Frame for cards
    umasu_container_frame = ttk.Frame(umasu_frame)
    umasu_container_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    umasu_canvas = tk.Canvas(umasu_container_frame, highlightthickness=0, bg=COLOR_BG)
    umasu_scrollbar = ttk.Scrollbar(umasu_container_frame, orient="vertical", command=umasu_canvas.yview)
    umasu_images_frame = ttk.Frame(umasu_canvas, style="Card.TFrame")
    
     Dumasu_images_frame.bind(
        "<Configure>",
        lambda e: umasu_canvas.configure(scrollregion=umasu_canvas.bbox("all"))
    )

    umasu_canvas.create_window((0, 0), window=umasu_images_frame, anchor="nw")
    umasu_canvas.configure(yscrollcommand=umasu_scrollbar.set)
    umasu_canvas.pack(side="left", fill="both", expand=True)
    umasu_scrollbar.pack(side="right", fill="y")

    # Footer/Control Frame
    umasu_controls_frame = ttk.Frame(umasu_frame, padding="10")
    umasu_controls_frame.pack(fill=tk.X, side=tk.BOTTOM)

    umasu_start_button = ttk.Button(umasu_controls_frame, text="START", command=start_umasu_program)
    umasu_start_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    umasu_stop_button = ttk.Button(umasu_controls_frame, text="STOP", command=stop_umasu_program, state=tk.DISABLED, style="Secondary.TButton")
    umasu_stop_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    umasu_upload_button = ttk.Button(umasu_controls_frame, text="อัปโหลดการ์ด", command=upload_card, style="Secondary.TButton")
    umasu_upload_button.pack(side=tk.RIGHT, padx=5, pady=5)


def create_macro_recorder_tab(notebook):
    """สร้าง UI สำหรับแท็บ Macro Recorder"""
    global macro_frame, macro_record_button, macro_replay_button
    global macro_status_label, macro_countdown_label, macro_hotkey_frame, hotkey_listener

    macro_frame = ttk.Frame(notebook, padding="10")
    macro_frame.pack(fill="both", expand=True)
    notebook.add(macro_frame, text="  บันทึกมาโคร  ")

    # Header
    macro_header_frame = ttk.Frame(macro_frame, padding="10")
    macro_header_frame.pack(fill=tk.X)
    ttk.Label(macro_header_frame, text="Macro Recorder", font=FONT_TITLE).pack(side=tk.LEFT)
    macro_status_label = ttk.Label(macro_header_frame, text="✅ พร้อมใช้งาน", style="Status.Idle.TLabel")
    macro_status_label.pack(side=tk.RIGHT, pady=5)

    # Main Content
    macro_content_frame = ttk.Frame(macro_frame, padding="10", style="Card.TFrame")
    macro_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Countdown and status labels
    macro_countdown_label = ttk.Label(macro_content_frame, text="", style="Countdown.TLabel", anchor="center")
    macro_countdown_label.pack(pady=(20, 10))

    # Buttons
    macro_buttons_frame = ttk.Frame(macro_content_frame)
    macro_buttons_frame.pack(pady=20)
    
    macro_record_button = ttk.Button(macro_buttons_frame, text="RECORD", command=start_recording, style="Danger.TButton")
    macro_record_button.pack(side=tk.LEFT, padx=10)
    
    macro_replay_button = ttk.Button(macro_buttons_frame, text="REPLAY", command=start_replay_with_dialog, style="Info.TButton")
    macro_replay_button.pack(side=tk.LEFT, padx=10)

    # Hotkey Instructions
    macro_hotkey_frame = ttk.Frame(macro_content_frame)
    macro_hotkey_frame.pack(pady=20, fill=tk.X)

    ttk.Label(macro_hotkey_frame, text="คีย์ลัดสำหรับควบคุม:", font=FONT_BOLD).pack()
    ttk.Label(macro_hotkey_frame, text="Ctrl + Alt + R: เริ่มบันทึก", style="Hotkey.TLabel").pack()
    ttk.Label(macro_hotkey_frame, text="Ctrl + Alt + S: หยุดบันทึก/หยุดเล่นซ้ำ", style="Hotkey.TLabel").pack()
    ttk.Label(macro_hotkey_frame, text="Ctrl + Alt + P: เริ่มเล่นซ้ำ", style="Hotkey.TLabel").pack()
    ttk.Label(macro_hotkey_frame, text="Ctrl + Alt + Q: ยกเลิกและปิดโปรแกรม", style="Hotkey.TLabel").pack()
    
    # Setup hotkeys with pynput
    def on_activate_rec(): hotkey_start_recording()
    def on_activate_stop(): hotkey_stop_recording()
    def on_activate_play(): hotkey_start_replay()
    def on_activate_quit(): hotkey_cancel_action()

    from pynput import keyboard
    hotkey_listener = keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+r': on_activate_rec,
        '<ctrl>+<alt>+s': on_activate_stop,
        '<ctrl>+<alt>+p': on_activate_play,
        '<ctrl>+<alt>+q': on_activate_quit,
    })
    hotkey_listener.daemon = True
    hotkey_listener.start()


# --- Main Application Window ---
root = tk.Tk()
root.title("Multi-Tool Application")
root.geometry("600x700")
root.minsize(600, 700)
root.protocol("WM_DELETE_WINDOW", cancel_action) # Handle window close event

setup_styles(root)

# Create a notebook widget
notebook = ttk.Notebook(root)
notebook.pack(pady=10, padx=10, fill="both", expand=True)

# Create tabs
create_image_counter_tab(notebook)
create_macro_recorder_tab(notebook)

# Initial setup
load_templates()
render_images_frame()

root.mainloop()