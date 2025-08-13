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

# ต้องติดตั้งไลบรารี playsound เพื่อให้การแจ้งเตือนด้วยเสียงทำงาน
# สามารถติดตั้งได้โดยใช้คำสั่ง: pip install playsound
from playsound import playsound

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
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# ตัวแปรสำหรับควบคุมการทำงาน
stop_event = threading.Event()
thread = None

# กำหนดชื่อไฟล์ภาพและจำนวนครั้งที่ต้องการ
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
                'last_found_time': 0 # เพิ่มตัวแปรสำหรับ Cooldown
            })
except FileNotFoundError:
    pass

# กำหนดชื่อไฟล์ภาพที่จะใช้สำหรับเงื่อนไขรีเซ็ต
reset_image_name = 'restart.png'

# กำหนดค่าคงที่สำหรับ Feature Matching
# [การปรับปรุง] ปรับค่า MIN_MATCH_COUNT และ nfeatures ตามโค้ดที่คุณส่งมา
MIN_MATCH_COUNT = 10 
orb = cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE) 

templates = {}
reset_template = None

# --- ฟังก์ชันจัดการข้อมูล ---
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

# --- ฟังก์ชันการทำงานหลัก (ปรับปรุงแล้ว) ---
def count_image_on_screen_orb(template_img):
    """
    จับภาพหน้าจอและนับจำนวนครั้งที่พบภาพต้นแบบด้วย ORB และ Homography
    คืนค่าเป็น tuple: (จำนวนที่พบ, ความแม่นยำเป็นเปอร์เซ็นต์)
    """
    sct = mss.mss()
    screen_shot = sct.grab(sct.monitors[0])
    screen_np = np.array(screen_shot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)

    found_count = 0
    best_accuracy = 0

    # [การปรับปรุง] ใช้ลูปการปรับขนาดตามโค้ดที่คุณส่งมา
    for scale in np.linspace(1.0, 0.2, 5): 
        # ข้ามการปรับขนาดที่เล็กเกินไป
        if template_img.shape[0] * scale < 20 or template_img.shape[1] * scale < 20:
            continue
            
        scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        kp1, des1 = orb.detectAndCompute(scaled_template, None)
        kp2, des2 = orb.detectAndCompute(screen_gray, None)
        
        if des1 is None or des2 is None or len(des1) < MIN_MATCH_COUNT or len(des2) < MIN_MATCH_COUNT:
            continue
            
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
                    accuracy_percent = (inliers / len(good_matches)) * 100
                    found_count = 1
                    best_accuracy = max(best_accuracy, accuracy_percent)
                    # ถ้าเจอแล้วให้ return ทันทีเพื่อประสิทธิภาพ
                    return found_count, best_accuracy

    return found_count, best_accuracy

def play_sound_and_notify(card_name, accuracy_percent):
    """ฟังก์ชันสำหรับเล่นเสียงและแสดงหน้าต่างแจ้งเตือน"""
    # [การแก้ไข] เปลี่ยนชื่อไฟล์เสียงที่ใช้ให้เป็นชื่อที่ง่ายขึ้น
    sound_path = resource_path("notification.wav")
    
    # [การปรับปรุง] เพิ่มการตรวจสอบไฟล์เสียงก่อนเล่นและจัดการข้อผิดพลาด
    if os.path.exists(sound_path):
        try:
            print(f"กำลังเล่นไฟล์เสียง: {sound_path}")
            playsound(sound_path, False) 
        except Exception as e:
            print(f"ข้อผิดพลาดในการเล่นเสียง: {e}")
            messagebox.showwarning("คำเตือน", f"ไม่สามารถเล่นไฟล์เสียง '{sound_path}' ได้\nข้อผิดพลาด: {e}")
    else:
        print("ไม่พบไฟล์เสียง notification.wav")
        messagebox.showwarning("คำเตือน", "ไม่พบไฟล์เสียง notification.wav\nโปรดตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับโปรแกรม")

    message = f"พบการ์ด '{card_name}' แล้ว!\nความแม่นยำ: {accuracy_percent:.2f}%"
    messagebox.showinfo("ตรวจพบการ์ด", message)

def run_main_loop():
    """ฟังก์ชันหลักที่ทำงานใน Background Thread"""
    while not stop_event.is_set():
        if reset_template is not None:
            current_reset_found, _ = count_image_on_screen_orb(reset_template)
            if current_reset_found > 0:
                root.after(0, lambda: status_label.config(text=f"รีเซ็ต!", style="Error.TLabel"))
                for img_config in images_config:
                    img_config['found'] = 0
                root.after(0, update_gui)
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
                    # รับค่าความแม่นยำกลับมาด้วย
                    current_found, accuracy = count_image_on_screen_orb(template_img)
                    
                    # --- แก้ไขส่วนนี้เพื่อป้องกันการนับซ้ำ ---
                    if current_found > 0:
                        current_time = time.time()
                        # กำหนดช่วงเวลา Cooldown 2 วินาที
                        if current_time - img_config['last_found_time'] > 2.0:
                            img_config['found'] += current_found
                            img_config['last_found_time'] = current_time
                            
                            # เรียกฟังก์ชันแจ้งเตือนด้วยเสียงและแสดงความแม่นยำ
                            root.after(0, lambda name=img_config['name'], acc=accuracy: play_sound_and_notify(name, acc))
                    # --------------------------------------
            
            if img_config['found'] < img_config['required']:
                all_conditions_met = False
        
        root.after(100, update_gui)
        
        if all_conditions_met and images_config:
            root.after(0, lambda: status_label.config(text="พบครบแล้ว!", style="Success.TLabel"))
            pyautogui.keyDown('f8')
            time.sleep(2)
            pyautogui.keyUp('f8')
            stop_event.set()
            break
        
        time.sleep(0.5)
    
    root.after(100, update_gui)
    root.after(100, update_status_after_stop)

# --- ฟังก์ชันควบคุม GUI ---
def update_gui():
    """อัปเดตค่าในหน้าต่าง GUI"""
    for img_config in images_config:
        img_config['label'].config(text=f"มี: {img_config['found']}/{img_config['required']} ใบ")

def update_status_after_stop():
    """อัปเดตสถานะเมื่อโปรแกรมหยุดการทำงาน"""
    status_label.config(text="หยุดการทำงาน", style="Default.TLabel")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    save_config()

def start_program():
    """เริ่มการทำงานของโปรแกรม"""
    global thread
    if thread is None or not thread.is_alive():
        for img_config in images_config:
            img_config['found'] = 0
            img_config['last_found_time'] = 0 # รีเซ็ต Cooldown
        
        if not images_config:
            messagebox.showwarning("คำเตือน", "กรุณาเพิ่มการ์ดที่ต้องการค้นหาก่อน")
            return

        stop_event.clear()
        thread = threading.Thread(target=run_main_loop)
        thread.daemon = True
        thread.start()
        status_label.config(text="กำลังค้นหา...", style="Info.TLabel")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

def stop_program():
    """หยุดการทำงานของโปรแกรม"""
    stop_event.set()
    update_status_after_stop()

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
        'last_found_time': 0 # เพิ่มตัวแปรสำหรับ Cooldown
    }
    # ป้องกันการเพิ่มการ์ดซ้ำ
    if any(config['name'] == file_name for config in images_config):
        return
        
    images_config.append(new_card_config)
    render_images_frame()
    load_templates()
    save_config()

def remove_card(card_config_to_remove):
    """ลบการ์ดที่เลือกออกจากรายการและ GUI"""
    if card_config_to_remove in images_config:
        # ลบไฟล์รูปภาพออกจากโฟลเดอร์ image ด้วย
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
    for widget in images_frame.winfo_children():
        widget.destroy()

    columns = 3
    for i, img_config in enumerate(images_config):
        frame = ttk.Frame(images_frame, style="Card.TFrame")
        frame.grid(row=i//columns, column=i%columns, padx=10, pady=10, sticky="nsew")
        img_config['frame'] = frame
        
        # ส่วนรูปภาพ
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
            
        # ส่วนแสดงผล/แก้ไขจำนวน
        count_frame = ttk.Frame(frame, style="Card.TFrame")
        count_frame.pack()
        
        ttk.Label(count_frame, text="ต้องการ:", font=thai_font, style="CardText.TLabel").pack(side=tk.LEFT, padx=(5, 2))
        
        required_var = tk.StringVar(value=str(img_config['required']))
        required_entry = ttk.Entry(count_frame, width=5, textvariable=required_var, font=thai_font, justify="center")
        required_entry.pack(side=tk.LEFT, padx=(0, 5))
        img_config['entry'] = required_entry
        
        # ส่วนแสดงจำนวนที่พบ
        text_label = ttk.Label(frame, text=f"มี: {img_config['found']}/{img_config['required']} ใบ", font=thai_font_bold, style="CardText.TLabel")
        text_label.pack(pady=5)
        img_config['label'] = text_label
        
        # ปุ่มลบการ์ด
        remove_button = ttk.Button(frame, text="ลบ", command=lambda config=img_config: remove_card(config), style="Danger.TButton")
        remove_button.pack(pady=5, padx=10, fill=tk.X)
    
    # ถ้าไม่มีการ์ด ให้แสดงข้อความ
    if not images_config:
        empty_label = ttk.Label(images_frame, text="ยังไม่มีการ์ด\nโปรดอัปโหลดหรือเพิ่มการ์ดใหม่", font=thai_font_title, style="Info.TLabel", justify="center")
        empty_label.pack(expand=True)
    
    # ทำให้ grid ใน images_frame ขยายได้
    for i in range((len(images_config) + columns - 1) // columns):
        images_frame.grid_rowconfigure(i, weight=1)
    for i in range(columns):
        images_frame.grid_columnconfigure(i, weight=1)


# --- สร้างหน้าต่างหลัก ---
root = tk.Tk()
root.title("Image Counter & Automator Pro")
root.geometry("1000x700")
style = ttk.Style(root)

# กำหนด Font สำหรับภาษาไทย
thai_font = font.Font(family="Tahoma", size=10)
thai_font_bold = font.Font(family="Tahoma", size=12, weight="bold")
thai_font_title = font.Font(family="Tahoma", size=14, weight="bold")
# กำหนด Style ใหม่สำหรับ Widget ต่างๆ
style.configure("TLabel", font=thai_font)
style.configure("TButton", font=thai_font_bold)
style.configure("TEntry", font=thai_font)

# Style สำหรับสถานะ
style.configure("Default.TLabel")
style.configure("Info.TLabel")
style.configure("Success.TLabel")
style.configure("Error.TLabel")

# Style สำหรับการ์ด
style.configure("Card.TFrame", relief="flat", borderwidth=0)
style.configure("CardImage.TLabel")
style.configure("CardText.TLabel")
style.configure("Danger.TButton")


# --- สร้าง UI Layout ---
# Header Frame
header_frame = ttk.Frame(root, padding="10")
header_frame.pack(fill=tk.X)
ttk.Label(header_frame, text="Image Counter & Automator Pro", font=thai_font_title).pack(side=tk.LEFT)
status_label = ttk.Label(header_frame, text="หยุดการทำงาน", font=thai_font_bold, style="Default.TLabel")
status_label.pack(side=tk.RIGHT)

# Scrollable Frame สำหรับการ์ด
container_frame = ttk.Frame(root)
container_frame.pack(fill="both", expand=True, padx=10, pady=10)
canvas = tk.Canvas(container_frame, highlightthickness=0)
scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=canvas.yview)
images_frame = ttk.Frame(canvas, style="Card.TFrame")

images_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=images_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Footer/Control Frame
controls_frame = ttk.Frame(root, padding="10")
controls_frame.pack(fill=tk.X, side=tk.BOTTOM)

start_button = ttk.Button(controls_frame, text="Start", command=start_program, style="Accent.TButton")
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = ttk.Button(controls_frame, text="Stop", command=stop_program, state=tk.DISABLED, style="Accent.TButton")
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

upload_button = ttk.Button(controls_frame, text="อัปโหลดการ์ด", command=upload_card)
upload_button.pack(side=tk.LEFT, padx=5, pady=5)

# โหลดภาพและสร้าง GUI เริ่มต้น
load_templates()
render_images_frame()

root.mainloop()
