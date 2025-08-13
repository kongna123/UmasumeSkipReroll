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
import winsound # ใช้ไลบรารี winsound ที่มาพร้อมกับ Python บน Windows

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

# Global variables for control
is_sound_muted = False
is_accuracy_hidden = False
stop_event = threading.Event()
thread = None

def play_notification_sound():
    """
    Plays the notification sound if the file is loaded successfully and sound is not muted.
    """
    global is_sound_muted
    if not is_sound_muted and SOUND_FILE:
        try:
            # winsound.PlaySound plays a .wav file
            winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME)
        except Exception as e:
            print(f"Failed to play sound: {e}")

# --- Custom Notification Window to avoid system sounds ---
def show_custom_notification(title, message):
    """
    Creates a simple custom Toplevel window to display a notification
    without triggering the default system sound from messagebox.
    """
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.config(padx=20, pady=20)
    popup.grab_set()  # Make the popup modal
    
    label = ttk.Label(popup, text=message, font=thai_font, justify=tk.CENTER)
    label.pack(pady=10)
    
    ok_button = ttk.Button(popup, text="ตกลง", command=popup.destroy)
    ok_button.pack(pady=10)
    
    # Center the popup window on the screen
    root.update_idletasks()
    window_width = popup.winfo_width()
    window_height = popup.winfo_height()
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    popup.geometry(f"{window_width}x{window_height}+{x}+{y}")


# --- Set image folder path using new function ---
IMAGE_FOLDER = resource_path("image")
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Set image file names and required counts
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
                'last_found_time': 0 # Added variable for cooldown
            })
except FileNotFoundError:
    pass

# Set the image file name for the reset condition
reset_image_name = 'restart.png'

# Define constants for Feature Matching
MIN_MATCH_COUNT = 10 
orb = cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE) 

templates = {}
reset_template = None

# --- Data management functions ---
def save_config():
    """Saves card settings to a file."""
    with open(resource_path("config.txt"), "w", encoding="utf-8") as f:
        for img_config in images_config:
            f.write(f"{img_config['name']},{img_config['required']}\n")

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

# --- Main function (updated) ---
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
        # Skip scaling that is too small
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
                    # If found, return immediately for efficiency
                    return found_count, best_accuracy

    return found_count, best_accuracy

def show_notification_message(card_name, accuracy_percent):
    """Function to display a notification window."""
    global is_accuracy_hidden
    if is_accuracy_hidden:
        message = f"พบการ์ด '{card_name}' แล้ว!"
    else:
        message = f"พบการ์ด '{card_name}' แล้ว!\nความแม่นยำ: {accuracy_percent:.2f}%"
    show_custom_notification("พบการ์ด", message)

def run_main_loop():
    """Main function that runs in a background thread."""
    while not stop_event.is_set():
        if reset_template is not None:
            current_reset_found, _ = count_image_on_screen_orb(reset_template)
            if current_reset_found > 0:
                root.after(0, lambda: status_label.config(text=f"กำลังรีเซ็ต!", style="Error.TLabel"))
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
                    current_found, accuracy = count_image_on_screen_orb(template_img)
                    
                    if current_found > 0:
                        current_time = time.time()
                        if current_time - img_config['last_found_time'] > 2.0:
                            img_config['found'] += current_found
                            img_config['last_found_time'] = current_time
                            
                            # Show notification window and play sound
                            root.after(0, lambda name=img_config['name'], acc=accuracy: show_notification_message(name, acc))
                            play_notification_sound()
                
            if img_config['found'] < img_config['required']:
                all_conditions_met = False
        
        root.after(100, update_gui)
        
        if all_conditions_met and images_config:
            root.after(0, lambda: status_label.config(text="พบการ์ดครบแล้ว!", style="Success.TLabel"))
            play_notification_sound()
            pyautogui.keyDown('f8')
            time.sleep(2)
            pyautogui.keyUp('f8')
            stop_event.set()
            break
        
        time.sleep(0.5)
    
    root.after(100, update_gui)
    root.after(100, update_status_after_stop)

# --- GUI control functions ---
def update_gui():
    """Updates the values in the GUI window."""
    for img_config in images_config:
        img_config['label'].config(text=f"พบ: {img_config['found']}/{img_config['required']} การ์ด")

def update_status_after_stop():
    """Updates the status when the program stops."""
    status_label.config(text="หยุด", style="Default.TLabel")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    save_config()

def start_program():
    """Starts the program."""
    global thread
    if thread is None or not thread.is_alive():
        for img_config in images_config:
            img_config['found'] = 0
            img_config['last_found_time'] = 0 # Reset cooldown
        
        if not images_config:
            messagebox.showwarning("คำเตือน", "โปรดเพิ่มการ์ดที่ต้องการค้นหาก่อน")
            return

        stop_event.clear()
        thread = threading.Thread(target=run_main_loop)
        thread.daemon = True
        thread.start()
        status_label.config(text="กำลังค้นหา...", style="Info.TLabel")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

def stop_program():
    """Stops the program."""
    stop_event.set()
    update_status_after_stop()

def toggle_mute():
    """Toggles the global sound muted state."""
    global is_sound_muted
    is_sound_muted = not is_sound_muted
    if is_sound_muted:
        mute_button.config(text="เปิดเสียง")
    else:
        mute_button.config(text="ปิดเสียง")

def toggle_accuracy_display():
    """Toggles the global accuracy hidden state."""
    global is_accuracy_hidden
    is_accuracy_hidden = not is_accuracy_hidden
    if is_accuracy_hidden:
        accuracy_button.config(text="เปิดเปอร์เซ็นต์")
    else:
        accuracy_button.config(text="ปิดเปอร์เซ็นต์")

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
        'last_found_time': 0 # Added variable for cooldown
    }
    # Prevent adding duplicate cards
    if any(config['name'] == file_name for config in images_config):
        return
        
    images_config.append(new_card_config)
    render_images_frame()
    load_templates()
    save_config()

def remove_card(card_config_to_remove):
    """Removes the selected card from the list and GUI."""
    if card_config_to_remove in images_config:
        # Also delete the image file from the image folder
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
        
        # Image part
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
            
        # Display/edit count part
        count_frame = ttk.Frame(frame, style="Card.TFrame")
        count_frame.pack()
        
        ttk.Label(count_frame, text="จำนวนที่ต้องการ:", font=thai_font, style="CardText.TLabel").pack(side=tk.LEFT, padx=(5, 2))
        
        required_var = tk.StringVar(value=str(img_config['required']))
        required_entry = ttk.Entry(count_frame, width=5, textvariable=required_var, font=thai_font, justify="center")
        required_entry.pack(side=tk.LEFT, padx=(0, 5))
        img_config['entry'] = required_entry
        
        # Display found count
        text_label = ttk.Label(frame, text=f"พบ: {img_config['found']}/{img_config['required']} การ์ด", font=thai_font_bold, style="CardText.TLabel")
        text_label.pack(pady=5)
        img_config['label'] = text_label
        
        # Remove card button
        remove_button = ttk.Button(frame, text="ลบ", command=lambda config=img_config: remove_card(config), style="Danger.TButton")
        remove_button.pack(pady=5, padx=10, fill=tk.X)
    
    # If no cards, display a message
    if not images_config:
        empty_label = ttk.Label(images_frame, text="ยังไม่มีการ์ด\nโปรดอัปโหลดหรือเพิ่มการ์ดใหม่", font=thai_font_title, style="Info.TLabel", justify="center")
        empty_label.pack(expand=True)
    
    # Make the grid in images_frame expandable
    for i in range((len(images_config) + columns - 1) // columns):
        images_frame.grid_rowconfigure(i, weight=1)
    for i in range(columns):
        images_frame.grid_columnconfigure(i, weight=1)


# --- Create main window ---
root = tk.Tk()
root.title("ตัวนับรูปภาพและระบบอัตโนมัติ")
root.geometry("1000x700")
style = ttk.Style(root)

# Define Thai font
thai_font = font.Font(family="Tahoma", size=10)
thai_font_bold = font.Font(family="Tahoma", size=12, weight="bold")
thai_font_title = font.Font(family="Tahoma", size=14, weight="bold")
# Define new styles for widgets
style.configure("TLabel", font=thai_font)
style.configure("TButton", font=thai_font_bold)
style.configure("TEntry", font=thai_font)

# Style for status
style.configure("Default.TLabel")
style.configure("Info.TLabel")
style.configure("Success.TLabel")
style.configure("Error.TLabel")

# Style for cards
style.configure("Card.TFrame", relief="flat", borderwidth=0)
style.configure("CardImage.TLabel")
style.configure("CardText.TLabel")
style.configure("Danger.TButton")


# --- Create UI Layout ---
# Header Frame
header_frame = ttk.Frame(root, padding="10")
header_frame.pack(fill=tk.X)
ttk.Label(header_frame, text="ตัวนับรูปภาพและระบบอัตโนมัติ", font=thai_font_title).pack(side=tk.LEFT)
status_label = ttk.Label(header_frame, text="หยุด", font=thai_font_bold, style="Default.TLabel")
status_label.pack(side=tk.RIGHT)

# Scrollable Frame for cards
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

start_button = ttk.Button(controls_frame, text="เริ่ม", command=start_program, style="Accent.TButton")
start_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

stop_button = ttk.Button(controls_frame, text="หยุด", command=stop_program, state=tk.DISABLED, style="Accent.TButton")
stop_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

upload_button = ttk.Button(controls_frame, text="อัปโหลดการ์ด", command=upload_card, style="Accent.TButton")
upload_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

# New mute button
mute_button = ttk.Button(controls_frame, text="ปิดเสียง", command=toggle_mute, style="Accent.TButton")
mute_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

# New accuracy toggle button
accuracy_button = ttk.Button(controls_frame, text="ปิดเปอร์เซ็นต์", command=toggle_accuracy_display, style="Accent.TButton")
accuracy_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)


# Load images and create initial GUI
load_templates()
render_images_frame()

root.mainloop()
