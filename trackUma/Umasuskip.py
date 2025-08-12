# coding: utf-8
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mss
import numpy as np
import pyautogui
import time
import threading
import os
import sys

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

# -----------------------------------------------

# --- ใช้ฟังก์ชันใหม่เพื่อกำหนดเส้นทางโฟลเดอร์รูปภาพ ---
IMAGE_FOLDER = resource_path("image")
# -----------------------------------------------

# ตัวแปรสำหรับควบคุมการทำงาน
stop_event = threading.Event()
thread = None

# กำหนดชื่อไฟล์ภาพและจำนวนครั้งที่ต้องการ
# (อย่าลืมแก้ไขชื่อไฟล์ภาพและจำนวนที่ต้องการให้ตรงกับของคุณ)
images_config = [
    {'name': 'SuperCreek.png', 'required': 4, 'found': 0, 'label': None, 'photo': None},
    {'name': 'Kitasan.png', 'required': 2, 'found': 0, 'label': None, 'photo': None},
]

# กำหนดชื่อไฟล์ภาพที่จะใช้สำหรับเงื่อนไขรีเซ็ต
reset_image_name = 'restart.png'

# กำหนดค่าคงที่สำหรับ Feature Matching
MIN_MATCH_COUNT = 10  # จำนวนจุดเด่นที่ตรงกันขั้นต่ำที่ถือว่า "พบ" ภาพ
orb = cv2.ORB_create(nfeatures=500) # สร้าง Object สำหรับตรวจจับจุดเด่น

# โหลดภาพต้นแบบล่วงหน้า
templates = {}
for img_config in images_config:
    full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
    template_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if template_img is not None:
        templates[img_config['name']] = template_img
    else:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่พบไฟล์รูปภาพ '{full_path}'")
        sys.exit(1)

reset_template = cv2.imread(os.path.join(IMAGE_FOLDER, reset_image_name), cv2.IMREAD_GRAYSCALE)
if reset_template is None:
    messagebox.showerror("ข้อผิดพลาด", f"ไม่พบไฟล์รูปภาพ '{reset_image_name}'")
    sys.exit(1)

def count_image_on_screen_orb(template_img):
    """
    จับภาพหน้าจอและนับจำนวนครั้งที่พบภาพต้นแบบ
    โดยใช้วิธี Feature Matching (ORB) ซึ่งแม่นยำกว่า
    """
    sct = mss.mss()
    screen_shot = sct.grab(sct.monitors[0])
    screen_np = np.array(screen_shot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)

    # 1. ค้นหาจุดเด่นและ Descriptors ของภาพต้นแบบและภาพหน้าจอ
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(screen_gray, None)

    if des1 is None or des2 is None:
        return 0, None

    # 2. ใช้ Brute-Force Matcher เพื่อหาคู่ที่ตรงกันที่สุด
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 3. เรียงลำดับคู่ที่ตรงกันตามระยะทาง (Distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # 4. กรองเอาเฉพาะคู่ที่ตรงกันที่ดีที่สุด
    good_matches = matches[:MIN_MATCH_COUNT]
    
    # 5. ตรวจสอบว่ามีจุดที่ตรงกันมากพอหรือไม่
    if len(good_matches) > MIN_MATCH_COUNT:
        # ถ้าเจอมากพอ ให้ถือว่าภาพนั้นถูกพบ
        return 1, good_matches
    
    return 0, None


def run_main_loop():
    """
    ฟังก์ชันหลักที่ทำงานใน Background Thread
    """
    loop_count = 0
    
    while not stop_event.is_set():
        loop_count += 1
        
        print(f"----------------------------------------")
        print(f"รอบที่ {loop_count}: กำลังค้นหารูปภาพ...")

        # ตรวจสอบภาพรีเซ็ต
        current_reset_found, _ = count_image_on_screen_orb(reset_template)
        if current_reset_found > 0:
            print(f">>> พบ '{reset_image_name}' กำลังรีเซ็ตการค้นหา...")
            root.after(0, lambda: status_label.config(text=f"พบ '{reset_image_name}'... กำลังรีเซ็ต!", style="Red.TLabel"))
            for img_config in images_config:
                img_config['found'] = 0
            root.after(0, update_gui)
            time.sleep(1)
            continue
        
        all_conditions_met = True
        
        for img_config in images_config:
            if img_config['found'] < img_config['required']:
                # ใช้ฟังก์ชันใหม่ในการนับภาพ
                template_img = templates[img_config['name']]
                current_found, _ = count_image_on_screen_orb(template_img)
                if current_found > 0:
                    print(f"พบ '{img_config['name']}' เพิ่ม {current_found} ใบ (รวม {img_config['found'] + current_found} ใบ)")
                img_config['found'] += current_found
            
            if img_config['found'] < img_config['required']:
                all_conditions_met = False
        
        root.after(100, update_gui)
        
        if all_conditions_met:
            print(">>> พบภาพครบตามเงื่อนไขแล้ว! กำลังกดปุ่ม 'F8' และหยุดการทำงาน")
            root.after(0, lambda: status_label.config(text="พบภาพครบตามเงื่อนไขแล้ว!", style="Green.TLabel"))
            pyautogui.keyDown('f8')
            time.sleep(2)
            pyautogui.keyUp('f8')
            stop_event.set()
            break
        
        time.sleep(0.5)
    
    root.after(100, update_gui)
    root.after(100, update_status_after_stop)
    print("----------------------------------------")
    print(">>> โปรแกรมหยุดการทำงานแล้ว")

def update_gui():
    """
    อัปเดตค่าในหน้าต่าง GUI
    """
    for img_config in images_config:
        img_config['label'].config(text=f"มี: {img_config['found']}/{img_config['required']} ใบ")

def update_status_after_stop():
    """
    อัปเดตสถานะเมื่อโปรแกรมหยุดการทำงาน
    """
    status_label.config(text="หยุดการทำงาน", style="Gray.TLabel")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def start_program():
    """
    เริ่มการทำงานของโปรแกรม
    """
    global thread
    if thread is None or not thread.is_alive():
        for img_config in images_config:
            img_config['found'] = 0
        stop_event.clear()
        thread = threading.Thread(target=run_main_loop)
        thread.daemon = True
        thread.start()
        status_label.config(text="กำลังค้นหา...", style="Blue.TLabel")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        print(">>> เริ่มการทำงานของโปรแกรมแล้ว")

def stop_program():
    """
    หยุดการทำงานของโปรแกรม
    """
    stop_event.set()
    update_status_after_stop()

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("Image Counter & Automator Pro")
root.geometry("800x400")
root.config(bg="#f0f0f0")

style = ttk.Style()
style.configure("Blue.TLabel", foreground="blue")
style.configure("Green.TLabel", foreground="green")
style.configure("Red.TLabel", foreground="red")
style.configure("Gray.TLabel", foreground="gray")

# Frame สำหรับแสดงรูปภาพและจำนวน
images_frame = ttk.Frame(root, padding="10")
images_frame.pack(expand=True, fill=tk.BOTH)

columns = 3
for i, img_config in enumerate(images_config):
    frame = ttk.Frame(images_frame)
    frame.grid(row=i//columns, column=i%columns, padx=10, pady=10)
    
    full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
    try:
        pil_image = Image.open(full_path)
        pil_image = pil_image.resize((100, 100), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        img_config['photo'] = photo
    
        img_label = ttk.Label(frame, image=photo)
        img_label.pack()
    except FileNotFoundError:
        img_label = ttk.Label(frame, text=f"ไม่พบไฟล์: {full_path}", font=("Arial", 8))
        img_label.pack()
        
    text_label = ttk.Label(frame, text=f"มี: {img_config['found']}/{img_config['required']} ใบ", font=("Arial", 12))
    text_label.pack()
    img_config['label'] = text_label

# Frame สำหรับปุ่มควบคุม
controls_frame = ttk.Frame(root, padding="10")
controls_frame.pack(fill=tk.X)

# ปุ่ม Start/Stop
start_button = ttk.Button(controls_frame, text="Start", command=start_program)
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = ttk.Button(controls_frame, text="Stop", command=stop_program, state=tk.DISABLED)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

# Label แสดงสถานะ
status_label = ttk.Label(controls_frame, text="หยุดการทำงาน", font=("Arial", 12, "bold"), style="Gray.TLabel")
status_label.pack(side=tk.RIGHT, padx=5, pady=5)

root.mainloop()
