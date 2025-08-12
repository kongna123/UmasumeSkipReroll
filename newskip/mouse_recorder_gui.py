# coding: utf-8
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

# ตัวแปรสำหรับควบคุมการทำงานของ Image Counter
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

# กำหนดชื่อไฟล์ภาพที่จะใช้สำหรับเงื่อนไขรีเซ็ต
reset_image_name = 'restart.png'

# กำหนดค่าคงที่สำหรับ Feature Matching
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
                root.after(0, lambda: umasu_status_label.config(text=f"รีเซ็ต!", style="Error.TLabel"))
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
            root.after(0, lambda: umasu_status_label.config(text="พบครบแล้ว!", style="Success.TLabel"))
            # แทนที่ pyautogui.keyDown('f8') ด้วยการหยุดการทำงาน
            umasu_stop_event.set() 
            root.after(0, update_umasu_status_after_stop)
            break
        
        time.sleep(0.5)
    
    root.after(100, update_umasu_gui)
    root.after(100, update_umasu_status_after_stop)

def update_umasu_gui():
    """อัปเดตค่าในหน้าต่าง GUI"""
    for img_config in images_config:
        img_config['label'].config(text=f"มี: {img_config['found']}/{img_config['required']} ใบ")

def update_umasu_status_after_stop():
    """อัปเดตสถานะเมื่อโปรแกรมหยุดการทำงาน"""
    umasu_status_label.config(text="หยุดการทำงาน", style="Default.TLabel")
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
        umasu_status_label.config(text="กำลังค้นหา...", style="Info.TLabel")
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
    if any(config['name'] == file_name for config in images_config):
        return
        
    images_config.append(new_card_config)
    render_images_frame()
    load_templates()
    save_config()

def remove_card(card_config_to_remove):
    """ลบการ์ดที่เลือกออกจากรายการและ GUI"""
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

    columns = 3
    for i, img_config in enumerate(images_config):
        frame = ttk.Frame(umasu_images_frame, style="Card.TFrame")
        frame.grid(row=i//columns, column=i%columns, padx=10, pady=10, sticky="nsew")
        img_config['frame'] = frame
        
        full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
        try:
            pil_image = Image.open(full_path)
            pil_image = pil_image.resize((100, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            img_config['photo'] = photo
        
            img_label = ttk.Label(frame, image=photo, style="CardImage.TLabel")
            img_label.pack(pady=(10, 5))
        except FileNotFoundError:
            img_label = ttk.Label(frame, text=f"ไม่พบไฟล์\n{img_config['name']}", font=thai_font, style="Error.TLabel", anchor="center")
            img_label.pack(pady=(10, 5))
            
        count_frame = ttk.Frame(frame, style="Card.TFrame")
        count_frame.pack()
        
        ttk.Label(count_frame, text="ต้องการ:", font=thai_font, style="CardText.TLabel").pack(side=tk.LEFT, padx=(5, 2))
        
        required_var = tk.StringVar(value=str(img_config['required']))
        required_entry = ttk.Entry(count_frame, width=5, textvariable=required_var, font=thai_font, justify="center")
        required_entry.pack(side=tk.LEFT, padx=(0, 5))
        img_config['entry'] = required_entry
        
        text_label = ttk.Label(frame, text=f"มี: {img_config['found']}/{img_config['required']} ใบ", font=thai_font_bold, style="CardText.TLabel")
        text_label.pack(pady=5)
        img_config['label'] = text_label
        
        remove_button = ttk.Button(frame, text="ลบ", command=lambda config=img_config: remove_card(config), style="Danger.TButton")
        remove_button.pack(pady=5, padx=10, fill=tk.X)
    
    if not images_config:
        empty_label = ttk.Label(umasu_images_frame, text="ยังไม่มีการ์ด\nโปรดอัปโหลดหรือเพิ่มการ์ดใหม่", font=thai_font_title, style="Info.TLabel", justify="center")
        empty_label.pack(expand=True)
    
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
        macro_countdown_label.config(text=f"กำลังจะเริ่มบันทึกใน {seconds_left} วินาที...", style="MacroCountdown.TLabel")
        countdown_id = root.after(1000, recording_countdown_tick, seconds_left - 1)
    else:
        macro_countdown_label.config(text="กำลังบันทึก...", style="MacroCountdown.TLabel")
        macro_status_label.config(text="สถานะ: กำลังบันทึก", style="MacroStatus.Recording.TLabel")
        
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
    recording_countdown_tick(10)
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
        macro_countdown_label.config(text=f"กำลังจะเริ่มเล่นซ้ำใน {seconds_left} วินาที...", style="MacroCountdown.TLabel")
        countdown_id = root.after(1000, replay_countdown_tick, seconds_left - 1, filename)
    else:
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
    
    root.after(0, lambda: macro_countdown_label.config(text="กำลังเล่นซ้ำ...", style="MacroCountdown.TLabel"))
    
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
    
    replay_countdown_tick(10, filename)
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
def hotkey_start_recording():
    root.after(0, start_recording)

def hotkey_stop_recording():
    root.after(0, stop_recording)

def hotkey_start_replay():
    root.after(0, start_replay_with_dialog)

def hotkey_stop_replay():
    root.after(0, stop_replay_action)

def hotkey_cancel_action():
    root.after(0, cancel_action)

def update_ui_for_recording():
    macro_record_button.config(state=tk.DISABLED)
    macro_replay_button.config(state=tk.DISABLED)
    macro_status_label.config(text="สถานะ: กำลังเตรียมการ...", style="MacroStatus.Recording.TLabel")
    macro_countdown_label.config(text="", style="MacroCountdown.Recording.TLabel")

def update_ui_for_replaying():
    macro_record_button.config(state=tk.DISABLED)
    macro_replay_button.config(state=tk.DISABLED)
    macro_status_label.config(text="สถานะ: กำลังเตรียมการ...", style="MacroStatus.Replaying.TLabel")
    macro_countdown_label.config(text="", style="MacroCountdown.Replaying.TLabel")

def update_ui_for_idle():
    macro_record_button.config(state=tk.NORMAL)
    macro_replay_button.config(state=tk.NORMAL)
    macro_status_label.config(text="สถานะ: พร้อมใช้งาน", style="MacroStatus.Idle.TLabel")
    macro_countdown_label.config(text="", style="MacroCountdown.Default.TLabel")


# ====================================================================
# --- การสร้างหน้าต่างหลักและ UI ---
# ====================================================================

root = tk.Tk()
root.title("UMA Tool - AIO")
root.geometry("1000x700")

# กำหนด Style และ Font
style = ttk.Style(root)
style.theme_use('vista') # หรือ 'clam', 'alt', 'default'
thai_font = font.Font(family="Tahoma", size=10)
thai_font_bold = font.Font(family="Tahoma", size=12, weight="bold")
thai_font_title = font.Font(family="Tahoma", size=14, weight="bold")

style.configure("TLabel", font=thai_font)
style.configure("TButton", font=thai_font_bold)
style.configure("TEntry", font=thai_font)
style.configure("TNotebook.Tab", font=thai_font_bold)

# Style สำหรับสถานะ
style.configure("Default.TLabel", foreground="#000")
style.configure("Info.TLabel", foreground="#3498db")
style.configure("Success.TLabel", foreground="#2ecc71")
style.configure("Error.TLabel", foreground="#e74c3c")
# Style สำหรับ Macro Status
style.configure("MacroStatus.Idle.TLabel", foreground="#27ae60", font=("Arial", 12))
style.configure("MacroStatus.Recording.TLabel", foreground="#e74c3c", font=("Arial", 12))
style.configure("MacroStatus.Replaying.TLabel", foreground="#2980b9", font=("Arial", 12))
# Style สำหรับ Macro Countdown
style.configure("MacroCountdown.Recording.TLabel", foreground="blue", font=("Arial", 12, "bold"))
style.configure("MacroCountdown.Replaying.TLabel", foreground="red", font=("Arial", 12, "bold"))
style.configure("MacroCountdown.Default.TLabel", foreground="black", font=("Arial", 12, "bold"))

# Style สำหรับการ์ด
style.configure("Card.TFrame", background="#FFFFFF", relief="solid", borderwidth=1)
style.configure("CardImage.TLabel", background="#FFFFFF")
style.configure("CardText.TLabel", background="#ecf0f1", foreground="#2c3e50")
style.configure("Danger.TButton", foreground="black", background="#e74c3c")
style.map("Danger.TButton",
          background=[('pressed', '!disabled', '#c0392b'), ('active', '#e74c3c')],
          foreground=[('pressed', '!disabled', 'black'), ('active', 'black')])

# สร้าง Notebook สำหรับจัดการแท็บ
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

# --- แท็บที่ 1: Image Counter ---
umasu_frame = ttk.Frame(notebook)
notebook.add(umasu_frame, text="ตรวจจับการ์ด (Image Counter)")

# Header Frame
umasu_header_frame = ttk.Frame(umasu_frame, padding="10")
umasu_header_frame.pack(fill=tk.X)
ttk.Label(umasu_header_frame, text="Image Counter & Automator Pro", font=thai_font_title).pack(side=tk.LEFT)
umasu_status_label = ttk.Label(umasu_header_frame, text="หยุดการทำงาน", font=thai_font_bold, style="Default.TLabel")
umasu_status_label.pack(side=tk.RIGHT)

# Scrollable Frame สำหรับการ์ด
umasu_container_frame = ttk.Frame(umasu_frame)
umasu_container_frame.pack(fill="both", expand=True, padx=10, pady=10)
umasu_canvas = tk.Canvas(umasu_container_frame, highlightthickness=0)
umasu_scrollbar = ttk.Scrollbar(umasu_container_frame, orient="vertical", command=umasu_canvas.yview)
umasu_images_frame = ttk.Frame(umasu_canvas, style="Card.TFrame")

umasu_images_frame.bind(
    "<Configure>",
    lambda e: umasu_canvas.configure(
        scrollregion=umasu_canvas.bbox("all")
    )
)

umasu_canvas.create_window((0, 0), window=umasu_images_frame, anchor="nw")
umasu_canvas.configure(yscrollcommand=umasu_scrollbar.set)
umasu_canvas.pack(side="left", fill="both", expand=True)
umasu_scrollbar.pack(side="right", fill="y")

# Footer/Control Frame
umasu_controls_frame = ttk.Frame(umasu_frame, padding="10")
umasu_controls_frame.pack(fill=tk.X, side=tk.BOTTOM)

umasu_start_button = ttk.Button(umasu_controls_frame, text="Start", command=start_umasu_program, style="Accent.TButton")
umasu_start_button.pack(side=tk.LEFT, padx=5, pady=5)

umasu_stop_button = ttk.Button(umasu_controls_frame, text="Stop", command=stop_umasu_program, state=tk.DISABLED, style="Accent.TButton")
umasu_stop_button.pack(side=tk.LEFT, padx=5, pady=5)

umasu_upload_button = ttk.Button(umasu_controls_frame, text="อัปโหลดการ์ด", command=upload_card)
umasu_upload_button.pack(side=tk.RIGHT, padx=5, pady=5)

# โหลดภาพและสร้าง GUI เริ่มต้นสำหรับ Image Counter
load_templates()
render_images_frame()


# --- แท็บที่ 2: Macro Recorder ---
macro_frame = ttk.Frame(notebook)
notebook.add(macro_frame, text="บันทึก/เล่นซ้ำ (Macro)")

macro_inner_frame = ttk.Frame(macro_frame, padding="20")
macro_inner_frame.pack(expand=True)

macro_button_frame = ttk.Frame(macro_inner_frame)
macro_button_frame.pack(pady=20)

macro_record_button = ttk.Button(macro_button_frame, text="บันทึก (F5)", command=start_recording)
macro_record_button.grid(row=0, column=0, padx=10, pady=10)

macro_replay_button = ttk.Button(macro_button_frame, text="เล่นซ้ำ (F7)", command=start_replay_with_dialog)
macro_replay_button.grid(row=0, column=1, padx=10, pady=10)

macro_status_label = ttk.Label(macro_inner_frame, text="สถานะ: พร้อมใช้งาน", style="MacroStatus.Idle.TLabel")
macro_status_label.pack(pady=10)

macro_countdown_label = ttk.Label(macro_inner_frame, text="", style="MacroCountdown.Default.TLabel")
macro_countdown_label.pack(pady=5)

hotkey_label = ttk.Label(macro_inner_frame, text="คีย์ลัด: บันทึก (F5), หยุดบันทึก (F6), เล่นซ้ำ (F7), หยุดเล่นซ้ำ (F8), ยกเลิก/ปิด (F9)", font=("Arial", 10), foreground="#555")
hotkey_label.pack(pady=5)

# --- สร้าง Hotkeys Listener ---
hotkeys = {
    '<f5>': hotkey_start_recording,
    '<f6>': hotkey_stop_recording,
    '<f7>': hotkey_start_replay,
    '<f8>': hotkey_stop_replay,
    '<f9>': hotkey_cancel_action
}
hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
hotkey_listener.start()

# ตรวจสอบว่า hotkey listener ทำงานอยู่หรือไม่
try:
    if hotkey_listener.is_alive():
        print("Hotkey listener thread is running.")
    else:
        print("Hotkey listener thread is not running.")
except Exception as e:
    print(f"Failed to check hotkey listener thread status: {e}")

update_ui_for_idle()

root.mainloop()
