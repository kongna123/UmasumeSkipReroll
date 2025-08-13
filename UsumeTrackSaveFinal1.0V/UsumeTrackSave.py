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
import winsound
from pynput import mouse, keyboard
import ast

# --- ฟังก์ชันแก้ไขเส้นทางสำหรับ PyInstaller ---
def resource_path(relative_path):
    """
    Accepts a relative path and returns a correct path
    in both development mode and when converted to .exe with PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Sound file path ---
SOUND_FILE = resource_path("notification.wav")
if not os.path.exists(SOUND_FILE):
    print(f"Error: Sound file not found at '{SOUND_FILE}'. Sound notifications will be disabled.")
    SOUND_FILE = None

# ----------------------------------------------------------------------
# GLOBAL VARIABLES AND CONSTANTS
# ----------------------------------------------------------------------

# --- Card Counter Global variables ---
is_sound_muted = False
is_accuracy_hidden = False
card_stop_event = threading.Event()
card_thread = None
IMAGE_FOLDER = resource_path("image")
reset_image_name = 'restart.png'
MIN_MATCH_COUNT = 10
orb = cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE)
templates = {}
reset_template = None
images_config = []

# --- Macro Automation Global variables ---
actions = []
is_recording = False
is_replaying = False
mouse_listener = None
keyboard_listener = None
hotkey_listener = None
start_time = None
replay_thread = None
recording_file = None
countdown_id = None

# Siganls for hotkey handling (Refactored for better signal management)
stop_recording_signal = threading.Event()
start_infinite_replay_signal = threading.Event()
start_limited_replay_signal = threading.Event()
stop_replay_signal = threading.Event()
cancel_all_signal = threading.Event()

# Replay speed settings
MOUSE_MOVE_DURATION = 0.05
REPLAY_SPEED_FACTOR = 1.0


# ----------------------------------------------------------------------
# COMMON UTILITY FUNCTIONS
# ----------------------------------------------------------------------
def show_custom_notification(title, message, parent_root):
    """
    Creates a simple custom Toplevel window to display a notification
    without triggering the default system sound from messagebox.
    """
    popup = tk.Toplevel(parent_root)
    popup.title(title)
    popup.config(padx=20, pady=20, bg="#2c3e50") # Dark background
    popup.grab_set()  # Make the popup modal

    label = ttk.Label(popup, text=message, font=thai_font_bold, justify=tk.CENTER, foreground="white", background="#2c3e50")
    label.pack(pady=10)

    ok_button = ttk.Button(popup, text="ตกลง", command=popup.destroy, style="Accent.TButton")
    ok_button.pack(pady=10)

    # Center the popup window on the screen
    parent_root.update_idletasks()
    window_width = popup.winfo_width()
    window_height = popup.winfo_height()
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    popup.geometry(f"+{x}+{y}")


# ----------------------------------------------------------------------
# IMAGE COUNTER FUNCTIONS
# ----------------------------------------------------------------------
def play_notification_sound():
    """
    Plays the notification sound if the file is loaded successfully and sound is not muted.
    """
    global is_sound_muted
    if not is_sound_muted and SOUND_FILE:
        try:
            winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME)
        except Exception as e:
            print(f"Failed to play sound: {e}")

def save_config():
    """
    [FIXED] Saves card settings to a file by reading the latest values from the entry widgets.
    """
    with open(resource_path("config.txt"), "w", encoding="utf-8") as f:
        for img_config in images_config:
            try:
                # Read the latest value directly from the Entry widget
                required_value = int(img_config['entry'].get())
            except (ValueError, KeyError):
                required_value = img_config.get('required', 0)
            
            f.write(f"{img_config['name']},{required_value}\n")

def load_templates():
    """Loads and prepares all template images."""
    global templates, reset_template
    templates.clear()
    
    for img_config in images_config:
        full_path = os.path.join(IMAGE_FOLDER, img_config['name'])
        if os.path.exists(full_path):
            template_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[img_config['name']] = template_img
            else:
                print(f"Error: Unable to read image file '{full_path}'")
        else:
            print(f"Error: Image file not found '{full_path}'")

    reset_path = os.path.join(IMAGE_FOLDER, reset_image_name)
    if os.path.exists(reset_path):
        reset_template = cv2.imread(reset_path, cv2.IMREAD_GRAYSCALE)
    else:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่พบไฟล์รูปภาพสำหรับรีเซ็ต '{reset_image_name}'")

def count_image_on_screen_orb(template_img):
    """
    Captures a screenshot and counts how many times the template image is found
    using ORB and Homography.
    Returns a tuple: (count found, percentage accuracy).
    """
    sct = mss.mss()
    screen_shot = sct.grab(sct.monitors[0])
    screen_np = np.array(screen_shot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2GRAY)

    found_count = 0
    best_accuracy = 0

    for scale in np.linspace(1.0, 0.2, 5): 
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
                    return found_count, best_accuracy
    return found_count, best_accuracy

def show_notification_message_card(card_name, accuracy_percent):
    """Function to display a notification window for the card counter."""
    global is_accuracy_hidden
    if is_accuracy_hidden:
        message = f"พบการ์ด '{card_name}' แล้ว!"
    else:
        message = f"พบการ์ด '{card_name}' แล้ว!\nความแม่นยำ: {accuracy_percent:.2f}%"
    show_custom_notification("พบการ์ด", message, root)

def stop_replay_macro():
    """Stops the macro replay and updates the UI."""
    global is_replaying, countdown_id
    is_replaying = False
    if countdown_id:
        root.after_cancel(countdown_id)
        countdown_id = None
    update_macro_ui_for_idle()

def run_card_main_loop():
    """Main function for the card counter that runs in a background thread."""
    while not card_stop_event.is_set():
        if reset_template is not None:
            current_reset_found, _ = count_image_on_screen_orb(reset_template)
            if current_reset_found > 0:
                root.after(0, lambda: card_status_label.config(text=f"กำลังรีเซ็ต!", style="Error.TLabel"))
                for img_config in images_config:
                    img_config['found'] = 0
                root.after(0, update_card_gui)
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
                    current_found, accuracy = count_image_on_screen_orb(template_img)
                    
                    if current_found > 0:
                        current_time = time.time()
                        if current_time - img_config['last_found_time'] > 2.0:
                            img_config['found'] += current_found
                            img_config['last_found_time'] = current_time
                            
                            root.after(0, lambda name=img_config['name'], acc=accuracy: show_notification_message_card(name, acc))
                            play_notification_sound()
                
            if img_config['found'] < img_config['required']:
                all_conditions_met = False
        
        root.after(100, update_card_gui)
        
        if all_conditions_met and images_config:
            root.after(0, lambda: card_status_label.config(text="พบการ์ดครบแล้ว!", style="Success.TLabel"))
            play_notification_sound()
            
            # --- START OF USER CHANGE ---
            # Now, stop the macro loop instead of pressing F8
            if is_replaying:
                root.after(0, stop_replay_macro)
            # --- END OF USER CHANGE ---
            
            card_stop_event.set()
            break
        
        time.sleep(0.5)
    
    root.after(100, update_card_gui)
    root.after(100, update_card_status_after_stop)

def update_card_gui():
    """Updates the values in the card counter GUI window."""
    for img_config in images_config:
        img_config['label'].config(text=f"พบ: {img_config['found']}/{img_config['required']} การ์ด")

def update_card_status_after_stop():
    """Updates the status when the program stops."""
    card_status_label.config(text="หยุด", style="Default.TLabel")
    card_start_button.config(state=tk.NORMAL)
    card_stop_button.config(state=tk.DISABLED)
    save_config()

def start_card_program():
    """Starts the card counter program."""
    global card_thread
    if card_thread is None or not card_thread.is_alive():
        for img_config in images_config:
            img_config['found'] = 0
            img_config['last_found_time'] = 0
        
        if not images_config:
            messagebox.showwarning("คำเตือน", "โปรดเพิ่มการ์ดที่ต้องการค้นหาก่อน")
            return

        card_stop_event.clear()
        card_thread = threading.Thread(target=run_card_main_loop)
        card_thread.daemon = True
        card_thread.start()
        card_status_label.config(text="กำลังค้นหา...", style="Info.TLabel")
        card_start_button.config(state=tk.DISABLED)
        card_stop_button.config(state=tk.NORMAL)

def stop_card_program():
    """Stops the card counter program."""
    card_stop_event.set()
    update_card_status_after_stop()

def toggle_mute():
    """Toggles the global sound muted state."""
    global is_sound_muted
    is_sound_muted = not is_sound_muted
    if is_sound_muted:
        card_mute_button.config(text="เปิดเสียง")
    else:
        card_mute_button.config(text="ปิดเสียง")

def toggle_accuracy_display():
    """Toggles the global accuracy hidden state."""
    global is_accuracy_hidden
    is_accuracy_hidden = not is_accuracy_hidden
    if is_accuracy_hidden:
        card_accuracy_button.config(text="เปิดเปอร์เซ็นต์")
    else:
        card_accuracy_button.config(text="ปิดเปอร์เซ็นต์")

def upload_card():
    """Opens a window to upload a new image."""
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
        messagebox.showinfo("สำเร็จ", f"อัปโหลดไฟล์ '{file_name}' เรียบร้อยแล้ว")
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"อัปโหลดไฟล์ล้มเหลว: {e}")

def add_card_to_gui(file_name, required_count):
    """Adds a new card to the data structure and creates a widget in the GUI."""
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
    """Removes the selected card from the list and GUI."""
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
    """Clears and creates a new card Frame."""
    for widget in images_frame.winfo_children():
        widget.destroy()

    columns = 3
    for i, img_config in enumerate(images_config):
        frame = ttk.Frame(images_frame, style="Card.TFrame")
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
            img_label = ttk.Label(frame, text=f"ไม่พบไฟล์\n{img_config['name']}", font=thai_font_bold, style="Error.TLabel", anchor="center")
            img_label.pack(pady=(10, 5))
            
        count_frame = ttk.Frame(frame, style="Card.TFrame")
        count_frame.pack()
        
        ttk.Label(count_frame, text="จำนวนที่ต้องการ:", font=thai_font, style="CardText.TLabel").pack(side=tk.LEFT, padx=(5, 2))
        
        required_var = tk.StringVar(value=str(img_config['required']))
        required_entry = ttk.Entry(count_frame, width=5, textvariable=required_var, font=thai_font, justify="center")
        required_entry.pack(side=tk.LEFT, padx=(0, 5))
        img_config['entry'] = required_entry
        
        text_label = ttk.Label(frame, text=f"พบ: {img_config['found']}/{img_config['required']} การ์ด", font=thai_font_bold, style="CardText.TLabel")
        text_label.pack(pady=5)
        img_config['label'] = text_label
        
        remove_button = ttk.Button(frame, text="ลบ", command=lambda config=img_config: remove_card(config), style="Danger.TButton")
        remove_button.pack(pady=5, padx=10, fill=tk.X)
    
    if not images_config:
        empty_label = ttk.Label(images_frame, text="ยังไม่มีการ์ด\nโปรดอัปโหลดหรือเพิ่มการ์ดใหม่", font=thai_font_title, style="Info.TLabel", justify="center")
        empty_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    for i in range((len(images_config) + columns - 1) // columns):
        images_frame.grid_rowconfigure(i, weight=1)
    for i in range(columns):
        images_frame.grid_columnconfigure(i, weight=1)

# ----------------------------------------------------------------------
# MACRO AUTOMATION FUNCTIONS
# ----------------------------------------------------------------------
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
        macro_countdown_label.config(text=f"กำลังจะเริ่มบันทึกใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, recording_countdown_tick, seconds_left - 1)
    else:
        macro_countdown_label.config(text="กำลังบันทึก...")
        macro_status_label.config(text="สถานะ: กำลังบันทึก", foreground="#e74c3c")
        
        actions = []
        start_time = time.time()
        
        mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        mouse_listener.start()

        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        print(">>> เริ่มการบันทึกเมาส์และคีย์บอร์ด...")


def start_recording_macro():
    """ฟังก์ชันสำหรับการเริ่มบันทึกจากปุ่ม GUI และคีย์ลัด"""
    global is_recording, actions, mouse_listener, keyboard_listener, start_time, countdown_id
    if is_recording:
        print(">>> การบันทึกกำลังดำเนินอยู่...")
        return

    is_recording = True
    update_macro_ui_for_recording()
    recording_countdown_tick(10)
    print(">>> เตรียมบันทึกเมาส์และคีย์บอร์ด...")

def stop_recording_macro(auto_stop=False):
    """ฟังก์ชันสำหรับการหยุดบันทึกจากปุ่ม GUI และคีย์ลัด"""
    global is_recording, mouse_listener, keyboard_listener, recording_file, countdown_id
    global stop_recording_signal

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
    
    if not auto_stop:
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
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
    
    update_macro_ui_for_idle()
    print(">>> หยุดการบันทึก")
    stop_recording_signal.clear()

def replay_countdown_tick(seconds_left, filename, loop_count):
    """ฟังก์ชันนับถอยหลังสำหรับการเล่นซ้ำ"""
    global countdown_id, is_replaying
    if not is_replaying:
        return
        
    if seconds_left > 0:
        macro_countdown_label.config(text=f"กำลังจะเริ่มเล่นซ้ำใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, replay_countdown_tick, seconds_left - 1, filename, loop_count)
    else:
        replay_thread = threading.Thread(target=run_replay, args=(filename, loop_count,))
        replay_thread.start()

def run_replay(filename, loop_count):
    """ฟังก์ชันสำหรับเล่นซ้ำการกระทำทั้งหมด"""
    global is_replaying, recording_file
    if not is_replaying:
        return

    recording_file = filename
    try:
        with open(recording_file, 'r') as f:
            saved_actions = [ast.literal_eval(line.strip()) for line in f]
    except FileNotFoundError:
        root.after(0, lambda: messagebox.showerror("ไฟล์ไม่พบ", f"ไม่พบไฟล์ {recording_file}"))
        is_replaying = False
        root.after(0, update_macro_ui_for_idle)
        return
        
    pyautogui.FAILSAFE = False
    original_pause = pyautogui.PAUSE
    pyautogui.PAUSE = 0
    
    root.after(0, lambda: macro_countdown_label.config(text="กำลังเล่นซ้ำ..."))
    
    replay_count = 0
    
    # Loop indefinitely if loop_count is 0, or for the specified number of times.
    while is_replaying and (loop_count == 0 or replay_count < loop_count):
        last_time = 0
        for action in saved_actions:
            if not is_replaying:
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
        
        if loop_count > 0:
            replay_count += 1
            root.after(0, lambda: macro_countdown_label.config(text=f"กำลังเล่นซ้ำรอบที่ {replay_count}/{loop_count}"))
        else:
            root.after(0, lambda: macro_countdown_label.config(text=f"กำลังเล่นซ้ำรอบที่ {replay_count+1} (วนลูป)"))

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = original_pause
    is_replaying = False
    root.after(0, update_macro_ui_for_idle)
    print(">>> เล่นซ้ำเสร็จสมบูรณ์แล้ว")

def choose_replay_file(callback_function, loop_count):
    """Opens a file dialog and calls the specified callback function with the selected file."""
    if is_replaying:
        return
    
    filename = filedialog.askopenfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        title="เลือกไฟล์การกระทำที่ต้องการเล่นซ้ำ"
    )

    if not filename:
        root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การเล่นซ้ำถูกยกเลิก ไม่มีการเลือกไฟล์"))
        update_macro_ui_for_idle()
        return

    callback_function(filename, loop_count)

def start_infinite_replay_from_gui():
    """Starts an infinite replay from the GUI button."""
    choose_replay_file(start_infinite_replay, 0)

def start_infinite_replay(filename, _):
    """
    Function to start an infinite replay loop.
    This is called from the GUI or hotkey.
    """
    global is_replaying
    if is_replaying:
        return
    
    is_replaying = True
    update_macro_ui_for_replaying(loop_count="∞")
    replay_countdown_tick(10, filename, 0)
    print(f">>> เตรียมเล่นซ้ำแบบวนลูปจากไฟล์: {filename} ...")

def start_limited_replay_from_gui():
    """Starts a limited replay from the GUI button."""
    try:
        loop_count = int(replay_loop_entry.get())
        if not 1 <= loop_count <= 99:
            raise ValueError
        choose_replay_file(start_limited_replay, loop_count)
    except ValueError:
        messagebox.showerror("ข้อผิดพลาด", "โปรดระบุจำนวนรอบที่ถูกต้อง (1-99)")
        update_macro_ui_for_idle()

def start_limited_replay(filename, loop_count):
    """
    Function to start a limited replay loop.
    This is called from the GUI or hotkey.
    """
    global is_replaying
    if is_replaying:
        return
        
    is_replaying = True
    update_macro_ui_for_replaying(loop_count=loop_count)
    replay_countdown_tick(10, filename, loop_count)
    print(f">>> เตรียมเล่นซ้ำจำนวน {loop_count} รอบ จากไฟล์: {filename} ...")

def cancel_all_actions():
    """
    [FIXED] ยกเลิกการทำงานปัจจุบันและปิดโปรแกรม
    พร้อมกับเรียกฟังก์ชัน save_config() เพื่อบันทึกค่าล่าสุดก่อนปิดโปรแกรม
    """
    global is_recording, is_replaying, hotkey_listener, countdown_id
    if is_recording:
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
        stop_recording_macro()
    if is_replaying:
        is_replaying = False
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
    
    if hotkey_listener:
        hotkey_listener.stop()
    
    # Save config before closing the application
    save_config()
    root.destroy()
    print(">>> ปิดโปรแกรม")

# ฟังก์ชันจัดการคีย์ลัดให้ Thread-safe
def hotkey_start_recording():
    root.after(0, start_recording_macro)

def hotkey_stop_recording():
    stop_recording_signal.set()
    
def hotkey_start_infinite_replay():
    start_infinite_replay_signal.set()

def hotkey_start_limited_replay():
    start_limited_replay_signal.set()

def hotkey_stop_replay():
    stop_replay_signal.set()

def hotkey_cancel_all():
    cancel_all_signal.set()

def check_hotkey_signal():
    """ฟังก์ชันที่จะถูกเรียกทุกๆ 100ms เพื่อตรวจสอบสัญญาณจากคีย์ลัด"""
    if stop_recording_signal.is_set():
        root.after(0, stop_recording_macro)
        stop_recording_signal.clear()
    
    if start_infinite_replay_signal.is_set():
        root.after(0, start_infinite_replay_from_gui)
        start_infinite_replay_signal.clear()
        
    if start_limited_replay_signal.is_set():
        root.after(0, start_limited_replay_from_gui)
        start_limited_replay_signal.clear()

    if stop_replay_signal.is_set():
        root.after(0, stop_replay_macro)
        stop_replay_signal.clear()

    if cancel_all_signal.is_set():
        root.after(0, cancel_all_actions)
        cancel_all_signal.clear()

    root.after(100, check_hotkey_signal)


def update_macro_ui_for_recording():
    macro_record_button.config(state=tk.DISABLED, bg="#34495e")
    macro_replay_infinite_button.config(state=tk.DISABLED, bg="#34495e")
    macro_replay_limited_button.config(state=tk.DISABLED, bg="#34495e")
    macro_cancel_button.config(text="หยุดบันทึก (F6)", bg="#e74c3c", activebackground="#c0392b", state=tk.NORMAL)
    macro_status_label.config(text="สถานะ: กำลังเตรียมการ...", foreground="#e74c3c")
    macro_countdown_label.config(text="", foreground="#3498db")

def update_macro_ui_for_replaying(loop_count="∞"):
    macro_record_button.config(state=tk.DISABLED, bg="#34495e")
    macro_replay_infinite_button.config(state=tk.DISABLED, bg="#34495e")
    macro_replay_limited_button.config(state=tk.DISABLED, bg="#34495e")
    macro_cancel_button.config(text="หยุดเล่นซ้ำ (F8)", bg="#e74c3c", activebackground="#c0392b", state=tk.NORMAL)
    macro_status_label.config(text=f"สถานะ: กำลังเตรียมการ ({loop_count} รอบ)", foreground="#2980b9")
    macro_countdown_label.config(text="", foreground="#2ecc71")

def update_macro_ui_for_idle():
    macro_record_button.config(state=tk.NORMAL, bg="#2ecc71", activebackground="#27ae60")
    macro_replay_infinite_button.config(state=tk.NORMAL, bg="#3498db", activebackground="#2980b9")
    macro_replay_limited_button.config(state=tk.NORMAL, bg="#3498db", activebackground="#2980b9")
    macro_cancel_button.config(text="ยกเลิกการใช้งาน (F10)", bg="#f39c12", activebackground="#e67e22", state=tk.NORMAL)
    macro_status_label.config(text="สถานะ: พร้อมใช้งาน", foreground="#2ecc71")
    macro_countdown_label.config(text="")


# ----------------------------------------------------------------------
# MAIN APPLICATION SETUP
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # --- Create main window ---
    root = tk.Tk()
    root.title("ตัวนับรูปภาพและมาโครอัตโนมัติ")
    root.geometry("1000x700")
    root.config(bg="#2c3e50")
    
    # --- Define Thai fonts ---
    thai_font = font.Font(family="Tahoma", size=11)
    thai_font_bold = font.Font(family="Tahoma", size=13, weight="bold")
    thai_font_title = font.Font(family="Tahoma", size=16, weight="bold")
    
    # --- Configure styles for a dark theme ---
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure("TFrame", background="#2c3e50", foreground="white")
    style.configure("TLabel", font=thai_font, background="#2c3e50", foreground="white")
    style.configure("TButton", font=thai_font_bold, foreground="white", background="#34495e", borderwidth=0, relief="flat", padding=10)
    style.map("TButton", background=[("active", "#34495e")])
    style.configure("TNotebook", background="#2c3e50", borderwidth=0)
    style.configure("TNotebook.Tab", background="#34495e", foreground="white", font=thai_font_bold, padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", "#3498db")], foreground=[("selected", "white")])

    # Custom styles
    style.configure("Accent.TButton", font=thai_font_bold, foreground="white", background="#3498db", borderwidth=0, relief="flat")
    style.map("Accent.TButton", background=[("active", "#2980b9"), ("disabled", "#7f8c8d")])
    style.configure("Danger.TButton", font=thai_font_bold, foreground="white", background="#e74c3c", borderwidth=0, relief="flat")
    style.map("Danger.TButton", background=[("active", "#c0392b")])
    
    # Label styles for different statuses
    style.configure("Default.TLabel", foreground="white")
    style.configure("Info.TLabel", foreground="#3498db")
    style.configure("Success.TLabel", foreground="#2ecc71")
    style.configure("Error.TLabel", foreground="#e74c3c")

    # Card-specific styles
    style.configure("Card.TFrame", background="#34495e", relief="flat", borderwidth=0, corner_radius=10)
    style.configure("CardImage.TLabel", background="#34495e")
    style.configure("CardText.TLabel", background="#34495e", foreground="white")


    # --- Create Notebook (Tabbed Interface) ---
    notebook = ttk.Notebook(root, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # ------------------------------------------------------------------
    # TAB 1: CARD COUNTER
    # ------------------------------------------------------------------
    card_tab = ttk.Frame(notebook)
    notebook.add(card_tab, text="ตัวนับการ์ด")

    # Header Frame
    card_header_frame = ttk.Frame(card_tab, padding="10")
    card_header_frame.pack(fill=tk.X)
    ttk.Label(card_header_frame, text="ตัวนับรูปภาพและระบบอัตโนมัติ", font=thai_font_title).pack(side=tk.LEFT)
    card_status_label = ttk.Label(card_header_frame, text="หยุด", font=thai_font_bold, style="Default.TLabel")
    card_status_label.pack(side=tk.RIGHT)

    # Scrollable Frame for cards
    card_container_frame = ttk.Frame(card_tab)
    card_container_frame.pack(fill="both", expand=True, padx=10, pady=10)
    card_canvas = tk.Canvas(card_container_frame, highlightthickness=0, bg="#2c3e50")
    card_scrollbar = ttk.Scrollbar(card_container_frame, orient="vertical", command=card_canvas.yview)
    images_frame = ttk.Frame(card_canvas, style="Card.TFrame")
    images_frame.bind(
        "<Configure>",
        lambda e: card_canvas.configure(
            scrollregion=card_canvas.bbox("all")
        )
    )
    card_canvas.create_window((0, 0), window=images_frame, anchor="nw")
    card_canvas.configure(yscrollcommand=card_scrollbar.set)
    card_canvas.pack(side="left", fill="both", expand=True)
    card_scrollbar.pack(side="right", fill="y")

    # Footer/Control Frame
    card_controls_frame = ttk.Frame(card_tab, padding="10")
    card_controls_frame.pack(fill=tk.X, side=tk.BOTTOM)

    card_start_button = ttk.Button(card_controls_frame, text="เริ่ม", command=start_card_program, style="Accent.TButton")
    card_start_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
    card_stop_button = ttk.Button(card_controls_frame, text="หยุด", command=stop_card_program, state=tk.DISABLED, style="Accent.TButton")
    card_stop_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
    card_upload_button = ttk.Button(card_controls_frame, text="อัปโหลดการ์ด", command=upload_card, style="Accent.TButton")
    card_upload_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
    card_mute_button = ttk.Button(card_controls_frame, text="ปิดเสียง", command=toggle_mute, style="Accent.TButton")
    card_mute_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
    card_accuracy_button = ttk.Button(card_controls_frame, text="ปิดเปอร์เซ็นต์", command=toggle_accuracy_display, style="Accent.TButton")
    card_accuracy_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
    
    # Load images and create initial GUI
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
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
    load_templates()
    render_images_frame()

    # ------------------------------------------------------------------
    # TAB 2: MACRO AUTOMATION
    # ------------------------------------------------------------------
    macro_tab = ttk.Frame(notebook)
    notebook.add(macro_tab, text="มาโครอัตโนมัติ")
    
    # Use a cleaner frame for macro controls
    macro_control_frame = ttk.Frame(macro_tab, padding="20")
    macro_control_frame.pack(pady=20, fill=tk.BOTH, expand=True)

    # Status labels
    macro_status_label = ttk.Label(macro_control_frame, text="สถานะ: พร้อมใช้งาน", font=thai_font_bold, style="Success.TLabel")
    macro_status_label.pack(pady=10)
    macro_countdown_label = ttk.Label(macro_control_frame, text="", font=thai_font_bold)
    macro_countdown_label.pack(pady=5)

    # Buttons frame with grid layout
    macro_button_frame = ttk.Frame(macro_control_frame)
    macro_button_frame.pack(pady=20)
    
    macro_record_button = tk.Button(macro_button_frame, text="บันทึก (F5)", command=start_recording_macro, width=15, height=2,
                                     font=thai_font_bold, fg="white", bg="#2ecc71", bd=0, activebackground="#27ae60", relief="flat")
    macro_record_button.grid(row=0, column=0, padx=10, pady=10)
    
    macro_replay_infinite_button = tk.Button(macro_button_frame, text="เล่นซ้ำวนลูป (F7)", command=start_infinite_replay_from_gui, width=15, height=2,
                                     font=thai_font_bold, fg="white", bg="#3498db", bd=0, activebackground="#2980b9", relief="flat")
    macro_replay_infinite_button.grid(row=0, column=1, padx=10, pady=10)

    replay_loop_frame = ttk.Frame(macro_button_frame)
    replay_loop_frame.grid(row=0, column=2, padx=10, pady=10)
    
    ttk.Label(replay_loop_frame, text="จำนวนรอบ (F9):", font=thai_font_bold, foreground="white", background="#2c3e50").pack(pady=2)
    replay_loop_var = tk.StringVar(value="10")
    replay_loop_entry = ttk.Entry(replay_loop_frame, width=5, textvariable=replay_loop_var, font=thai_font_bold, justify="center")
    replay_loop_entry.pack(side=tk.LEFT)
    
    macro_replay_limited_button = tk.Button(replay_loop_frame, text="เริ่ม", command=start_limited_replay_from_gui, width=5, height=1,
                                     font=thai_font_bold, fg="white", bg="#3498db", bd=0, activebackground="#2980b9", relief="flat")
    macro_replay_limited_button.pack(side=tk.LEFT, padx=5)

    macro_cancel_button = tk.Button(macro_control_frame, text="ยกเลิกการใช้งาน (F10)", command=cancel_all_actions, width=50, height=2,
                                     font=thai_font_bold, fg="white", bg="#f39c12", bd=0, activebackground="#e67e22", relief="flat")
    macro_cancel_button.pack(pady=20)
    
    macro_hotkey_label = ttk.Label(macro_control_frame, text="คีย์ลัด: F5 บันทึก | F6 หยุดบันทึก | F7 เล่นซ้ำวนลูป | F9 เล่นซ้ำตามจำนวน | F8 หยุดเล่นซ้ำ | F10 ยกเลิกการทำงานทั้งหมด", font=thai_font)
    macro_hotkey_label.pack(pady=5)
    
    # --- Setup hotkeys for macro automation ---
    hotkeys = {
        '<f5>': hotkey_start_recording,
        '<f6>': hotkey_stop_recording,
        '<f7>': hotkey_start_infinite_replay,
        '<f9>': hotkey_start_limited_replay,
        '<f8>': hotkey_stop_replay,
        '<f10>': hotkey_cancel_all
    }
    
    try:
        hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
        hotkey_listener.start()
        print(">>> ตั้งค่าคีย์ลัดสำเร็จ (F5, F6, F7, F8, F9, F10)")
    except Exception as e:
        print(f"Failed to set up hotkeys: {e}")
        
    root.protocol("WM_DELETE_WINDOW", cancel_all_actions)
    root.after(100, check_hotkey_signal)
    root.mainloop()
