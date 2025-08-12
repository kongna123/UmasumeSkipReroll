import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
from PIL import Image, ImageTk
import cv2
import mss
import numpy as np
import pyautogui
import time
import threading
import os
import sys
import shutil
from pynput import mouse, keyboard
import ast

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
            name, required = line.strip().split(',')
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

reset_image_name = 'restart.png'
MIN_MATCH_COUNT = 15
orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
templates = {}
reset_template = None

def save_config():
    """บันทึกการตั้งค่าการ์ดลงในไฟล์"""
    with open(resource_path("config.txt"), "w", encoding="utf-8") as f:
        for img_config in images_config:
            f.write(f"{img_config['name']},{img_config['required']}\n")

def load_templates():
    """โหลดและเตรียมภาพต้นแบบทั้งหมด"""
    global templates, reset_template
    templates.clear()
    
    for img_config in images_config:
        full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
        if os.path.exists(full_path):
            template_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[img_config['name']] = template_img
            else:
                print(f"ข้อผิดพลาด: ไม่สามารถอ่านไฟล์รูปภาพ '{full_path}'")
        else:
            print(f"ข้อผิดพลาด: ไม่พบไฟล์รูปภาพ '{full_path}'")

    reset_path = os.path.join(IMAGE_FOLDER, reset_image_name)
    if os.path.exists(reset_path):
        reset_template = cv2.imread(reset_path, cv2.IMREAD_GRAYSCALE)
    else:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่พบไฟล์รูปภาพ '{reset_image_name}'")

def count_image_on_screen_orb(template_img):
    """จับภาพหน้าจอและนับจำนวนครั้งที่พบภาพต้นแบบด้วย ORB และ Homography"""
    sct = mss.mss()
    screen_shot = sct.grab(sct.monitors[0])
    screen_np = np.array(screen_shot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)
    
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(screen_gray, None)

    if des1 is None or des2 is None or len(des1) < MIN_MATCH_COUNT or len(des2) < MIN_MATCH_COUNT:
        return 0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches[:50]
    
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            matches_mask = mask.ravel().tolist()
            inliers = sum(matches_mask)
            
            if inliers >= MIN_MATCH_COUNT:
                return 1, good_matches
    
    return 0, None

def run_umasu_main_loop():
    """ฟังก์ชันหลักที่ทำงานใน Background Thread"""
    while not umasu_stop_event.is_set():
        if reset_template is not None:
            current_reset_found, _ = count_image_on_screen_orb(reset_template)
            if current_reset_found > 0:
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
                template_img = templates.get(img_config['name'])
                if template_img is not None:
                    current_found, _ = count_image_on_screen_orb(template_img)
                    
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
            # เปลี่ยนสีข้อความเมื่อพบครบ
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

    columns = 4 # เพิ่มจำนวนคอลัมน์เพื่อให้กระทัดรัดขึ้น
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
    recording_countdown_tick(5) # ลดเวลานับถอยหลัง
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
    
    replay_countdown_tick(5, filename) # ลดเวลานับถอยหลัง
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
    
    umasu_images_frame.bind(
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
    global macro_status_label, macro_countdown_label

    macro_frame = ttk.Frame(notebook, padding="20")
    macro_frame.pack(fill="both", expand=True)
    notebook.add(macro_frame, text="  บันทึก/เล่นซ้ำ  ")

    macro_frame.grid_rowconfigure(0, weight=1)
    macro_frame.grid_rowconfigure(3, weight=1)
    macro_frame.grid_columnconfigure(0, weight=1)
    
    # Control Buttons Frame
    control_frame = ttk.Frame(macro_frame)
    control_frame.grid(row=1, column=0, pady=20)
    
    macro_record_button = ttk.Button(control_frame, text="บันทึก (F5)", command=start_recording)
    macro_record_button.pack(side=tk.LEFT, padx=10, ipady=10, ipadx=20)

    macro_replay_button = ttk.Button(control_frame, text="เล่นซ้ำ (F7)", command=start_replay_with_dialog)
    macro_replay_button.pack(side=tk.LEFT, padx=10, ipady=10, ipadx=20)

    # Status Display Frame
    status_frame = ttk.Frame(macro_frame, style="Card.TFrame", padding=20)
    status_frame.grid(row=2, column=0, pady=10, sticky="ew")
    status_frame.columnconfigure(0, weight=1)

    macro_status_label = ttk.Label(status_frame, text=" ", style="Status.Idle.TLabel", anchor="center")
    macro_status_label.grid(row=0, column=0, pady=5)

    macro_countdown_label = ttk.Label(status_frame, text="", style="Countdown.TLabel", anchor="center")
    macro_countdown_label.grid(row=1, column=0, pady=5)

    hotkey_info = "F5: เริ่มบันทึก  |  F6: หยุดบันทึก  |  F7: เล่นซ้ำ  |  F8: หยุดเล่นซ้ำ  |  F9: ปิดโปรแกรม"
    hotkey_label = ttk.Label(macro_frame, text=hotkey_info, style="Hotkey.TLabel", anchor="center")
    hotkey_label.grid(row=4, column=0, pady=(20, 0), sticky="s")


# --- Main Application Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("UMA Tool - AIO")
    root.geometry("1000x750")
    root.minsize(800, 600)
    root.configure(bg=COLOR_BG)

    # Setup styles
    setup_styles(root)
    
    # Create Notebook for tabs
    notebook = ttk.Notebook(root, style="TNotebook")
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # Create Tabs
    create_image_counter_tab(notebook)
    create_macro_recorder_tab(notebook)
    
    # Initial load for Image Counter
    load_templates()
    render_images_frame()

    # Initial UI state for Macro
    update_ui_for_idle()

    # Create Hotkeys Listener
    hotkeys = {
        '<f5>': hotkey_start_recording,
        '<f6>': hotkey_stop_recording,
        '<f7>': hotkey_start_replay,
        '<f8>': hotkey_stop_replay,
        '<f9>': hotkey_cancel_action
    }
    hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
    hotkey_listener.start()

    # Graceful shutdown
    root.protocol("WM_DELETE_WINDOW", cancel_action)
    root.mainloop()
