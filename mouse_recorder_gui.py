import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pynput import mouse, keyboard
import pyautogui
import time
import threading
import json
import sys

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
replay_loop = False
stop_recording_signal = False

# Initial settings
REPLAY_SPEED_FACTOR = 1.0
MOUSE_MOVE_DURATION = 0.005 # Make mouse movements smoother

# Color scheme
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#34495e',
    'accent': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'background': '#ecf0f1',
    'card': '#ffffff',
    'text': '#2c3e50',
    'text_light': '#7f8c8d',
    'border': '#bdc3c7'
}

# --- Recording Functions ---
def on_press(key):
    """Handles key presses during recording."""
    global is_recording, actions, start_time
    if is_recording:
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).split('.')[-1]
        
        elapsed_time = time.time() - start_time
        actions.append({'type': 'key', 'state': 'press', 'key': key_name, 'time': elapsed_time})

def on_release(key):
    """Handles key releases during recording."""
    global is_recording, actions, start_time, stop_recording_signal
    if is_recording:
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).split('.')[-1]
        
        elapsed_time = time.time() - start_time
        actions.append({'type': 'key', 'state': 'release', 'key': key_name, 'time': elapsed_time})
        
        if key == keyboard.Key.esc:
            stop_recording_signal = True
            return False # Stop the listener

def on_click(x, y, button, pressed):
    """Handles mouse clicks during recording."""
    global is_recording, actions, start_time
    if is_recording:
        elapsed_time = time.time() - start_time
        if pressed:
            actions.append({'type': 'click', 'state': 'press', 'x': x, 'y': y, 'button': str(button).split('.')[-1], 'time': elapsed_time})
        else:
            actions.append({'type': 'click', 'state': 'release', 'x': x, 'y': y, 'button': str(button).split('.')[-1], 'time': elapsed_time})

def on_move(x, y):
    """Handles mouse movements during recording."""
    global is_recording, actions, start_time
    if is_recording:
        elapsed_time = time.time() - start_time
        if not actions or actions[-1].get('type') != 'move' or (actions[-1].get('x') != x or actions[-1].get('y') != y):
            actions.append({'type': 'move', 'x': x, 'y': y, 'time': elapsed_time})

def recording_countdown_tick(seconds_left):
    """Countdown function before recording starts."""
    global countdown_id, is_recording, actions, mouse_listener, keyboard_listener, start_time
    if not is_recording:
        return
        
    if seconds_left > 0:
        countdown_label.config(text=f"กำลังจะเริ่มบันทึกใน {seconds_left} วินาที...", fg=COLORS['warning'])
        countdown_id = root.after(1000, recording_countdown_tick, seconds_left - 1)
    else:
        # Start recording after countdown
        countdown_label.config(text="🔴 กำลังบันทึก...", fg=COLORS['danger'])
        status_label.config(text="สถานะ: กำลังบันทึก", fg=COLORS['danger'])
        
        actions = []
        start_time = time.time()
        
        mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        mouse_listener.start()

        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.start()
        print(">>> เริ่มการบันทึกเมาส์และคีย์บอร์ด...")

def start_recording_gui():
    """Starts the recording from the GUI button."""
    global is_recording
    if is_recording:
        return

    is_recording = True 
    update_ui_for_recording()
    recording_countdown_tick(3)
    print(">>> เตรียมบันทึกเมาส์และคีย์บอร์ด...")

def stop_recording_gui(auto_stop=False):
    """Stops the recording from GUI or hotkey."""
    global is_recording, mouse_listener, keyboard_listener, recording_file, countdown_id
    
    if not is_recording:
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
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="เลือกที่สำหรับบันทึกการกระทำ"
        )

        if filename:
            recording_file = filename
            with open(recording_file, 'w', encoding='utf-8') as f:
                json.dump(actions, f, indent=4)
            
            root.after(0, lambda: messagebox.showinfo("บันทึกเสร็จสิ้น", f"บันทึกข้อมูลการกระทำ {len(actions)} ครั้ง ลงในไฟล์\n{recording_file}"))
        else:
            root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การบันทึกถูกยกเลิก ไม่มีการบันทึกไฟล์"))
    
    update_ui_for_idle()
    print(">>> หยุดการบันทึก")

# --- Replay Functions ---
def run_replay():
    """Replays all recorded actions."""
    global is_replaying, recording_file, REPLAY_SPEED_FACTOR, replay_loop
    
    if not is_replaying:
        return
        
    # Load the action file
    try:
        with open(recording_file, 'r', encoding='utf-8') as f:
            saved_actions = json.load(f)
    except FileNotFoundError:
        root.after(0, lambda: messagebox.showerror("ไฟล์ไม่พบ", f"ไม่พบไฟล์ {recording_file}"))
        is_replaying = False
        root.after(0, update_ui_for_idle)
        return
    except json.JSONDecodeError:
        root.after(0, lambda: messagebox.showerror("ข้อผิดพลาดของไฟล์", f"ไม่สามารถอ่านไฟล์ {recording_file} ได้ กรุณาตรวจสอบรูปแบบไฟล์"))
        is_replaying = False
        root.after(0, update_ui_for_idle)
        return

    pyautogui.FAILSAFE = False
    
    root.after(0, lambda: countdown_label.config(text="▶️ กำลังเล่นซ้ำ...", fg=COLORS['accent']))
    
    while is_replaying:
        start_of_replay = time.time()
        
        for action in saved_actions:
            if not is_replaying:
                break
                
            elapsed_from_start = time.time() - start_of_replay
            # Wait for the correct time to perform the action
            while elapsed_from_start / REPLAY_SPEED_FACTOR < action.get('time', 0):
                if not is_replaying:
                    break
                time.sleep(0.005)
                elapsed_from_start = time.time() - start_of_replay

            if not is_replaying:
                break
            
            action_type = action['type']
            
            if action_type == 'move':
                # Use pyautogui.moveTo for smooth movement
                pyautogui.moveTo(action['x'], action['y'], duration=MOUSE_MOVE_DURATION)
            elif action_type == 'click':
                if action['state'] == 'press':
                    pyautogui.mouseDown(action['x'], action['y'], button=action['button'])
                else:
                    pyautogui.mouseUp(action['x'], action['y'], button=action['button'])
            elif action_type == 'key':
                try:
                    key_name = action['key']
                    if key_name == 'esc': # Don't press esc during replay
                        continue
                    if action['state'] == 'press':
                        pyautogui.keyDown(key_name)
                    else:
                        pyautogui.keyUp(key_name)
                except Exception as e:
                    print(f"Failed to press key {key_name}: {e}")

        # If not looping, stop after one run
        if not replay_loop:
            is_replaying = False
            break
            
    pyautogui.FAILSAFE = True
    is_replaying = False
    root.after(0, update_ui_for_idle)
    root.after(0, lambda: countdown_label.config(text="✅ เล่นซ้ำเสร็จสมบูรณ์แล้ว", fg=COLORS['success']))
    print(">>> เล่นซ้ำเสร็จสมบูรณ์แล้ว")

def start_replay_gui(is_loop=False):
    """Function to start replay from a button."""
    global is_replaying, recording_file, replay_thread, replay_loop
    if is_replaying:
        return

    filename = filedialog.askopenfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="เลือกไฟล์การกระทำที่ต้องการเล่นซ้ำ"
    )

    if not filename:
        root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การเล่นซ้ำถูกยกเลิก ไม่มีการเลือกไฟล์"))
        return
    
    recording_file = filename
    is_replaying = True
    replay_loop = is_loop
    update_ui_for_replaying()
    
    replay_thread = threading.Thread(target=run_replay)
    replay_thread.daemon = True
    replay_thread.start()
    print(f">>> เริ่มเล่นซ้ำจากไฟล์: {filename} (วนลูป: {replay_loop})")

def cancel_action():
    """Cancels the current action and closes the program."""
    global is_recording, is_replaying, hotkey_listener, countdown_id
    if is_recording:
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
        stop_recording_gui(auto_stop=True)
    if is_replaying:
        is_replaying = False
    
    if hotkey_listener:
        hotkey_listener.stop()
    
    root.destroy()
    sys.exit(0)
    print(">>> ปิดโปรแกรม")

# --- UI and Hotkey Functions ---
def hotkey_start_recording():
    """Thread-safe function to start recording via hotkey."""
    root.after(0, start_recording_gui)

def hotkey_stop_recording():
    """Thread-safe function to stop recording via hotkey."""
    root.after(0, stop_recording_gui)

def hotkey_start_replay_once():
    """Thread-safe function to start single replay via hotkey."""
    root.after(0, lambda: start_replay_gui(is_loop=False))
    
def hotkey_start_replay_loop():
    """Thread-safe function to start looped replay via hotkey."""
    root.after(0, lambda: start_replay_gui(is_loop=True))
    
def hotkey_cancel_action():
    """Thread-safe function to cancel current action via hotkey."""
    root.after(0, cancel_action)

def update_replay_speed(value):
    """Updates the replay speed factor."""
    global REPLAY_SPEED_FACTOR
    REPLAY_SPEED_FACTOR = float(value)
    speed_value_label.config(text=f"x{REPLAY_SPEED_FACTOR:.1f}")

def create_button(parent, text, command, bg_color, width=18, height=2):
    """Creates a styled button with hover effects."""
    button = tk.Button(
        parent, 
        text=text, 
        command=command,
        width=width, 
        height=height,
        font=("Segoe UI", 11, "bold"), 
        fg="white", 
        bg=bg_color,
        bd=0,
        cursor="hand2",
        relief="flat"
    )
    
    # Add hover effects
    def on_enter(e):
        button.configure(bg=lighten_color(bg_color))
    
    def on_leave(e):
        button.configure(bg=bg_color)
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    return button

def lighten_color(color):
    """Lightens a hex color by 10%."""
    if color.startswith('#'):
        hex_color = color[1:]
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        lightened = tuple(min(255, int(c * 1.1)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*lightened)
    return color

def update_ui_for_recording():
    """Updates the UI state for recording."""
    record_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    play_once_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    loop_play_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    cancel_button.config(text="⏹️ หยุดบันทึก (Esc)", bg=COLORS['danger'])
    status_label.config(text="สถานะ: กำลังเตรียมการ...", fg=COLORS['warning'])
    countdown_label.config(text="", fg=COLORS['accent'])

def update_ui_for_replaying():
    """Updates the UI state for replaying."""
    record_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    play_once_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    loop_play_button.config(state=tk.DISABLED, bg=COLORS['text_light'])
    cancel_button.config(text="⏹️ หยุดเล่นซ้ำ (Esc)", bg=COLORS['danger'])
    status_label.config(text="สถานะ: กำลังเตรียมการ...", fg=COLORS['accent'])
    countdown_label.config(text="")

def update_ui_for_idle():
    """Updates the UI state for idle."""
    record_button.config(state=tk.NORMAL, bg=COLORS['success'])
    play_once_button.config(state=tk.NORMAL, bg=COLORS['accent'])
    loop_play_button.config(state=tk.NORMAL, bg=COLORS['accent'])
    cancel_button.config(text="❌ ยกเลิก (ESC)", bg=COLORS['warning'])
    status_label.config(text="สถานะ: พร้อมใช้งาน", fg=COLORS['success'])
    countdown_label.config(text="")

# --- Main Window Setup ---
root = tk.Tk()
root.title("🎮 Gaming Automator Pro")
root.geometry("600x480")
root.config(bg=COLORS['background'])
root.resizable(False, False)

# Main container with padding
main_frame = tk.Frame(root, bg=COLORS['background'])
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

# Header Section
header_frame = tk.Frame(main_frame, bg=COLORS['card'], relief="flat", bd=2)
header_frame.pack(fill=tk.X, pady=(0, 15))

title_label = tk.Label(
    header_frame, 
    text="🎮 Gaming Automator Pro", 
    font=("Segoe UI", 20, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['primary']
)
title_label.pack(pady=10)

# Status Section
status_frame = tk.Frame(main_frame, bg=COLORS['card'], relief="flat", bd=2)
status_frame.pack(fill=tk.X, pady=(0, 15))

status_label = tk.Label(
    status_frame, 
    text="สถานะ: พร้อมใช้งาน", 
    font=("Segoe UI", 14, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['success']
)
status_label.pack(pady=10)

countdown_label = tk.Label(
    status_frame, 
    text="", 
    font=("Segoe UI", 12), 
    bg=COLORS['card'], 
    fg=COLORS['accent']
)
countdown_label.pack(pady=(0, 10))

# Control Buttons Section
controls_frame = tk.Frame(main_frame, bg=COLORS['card'], relief="flat", bd=2)
controls_frame.pack(fill=tk.X, pady=(0, 15))

controls_title = tk.Label(
    controls_frame, 
    text="📱 ควบคุมการทำงาน", 
    font=("Segoe UI", 14, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['primary']
)
controls_title.pack(pady=(10, 8))

# Buttons grid
buttons_frame = tk.Frame(controls_frame, bg=COLORS['card'])
buttons_frame.pack(pady=10)

record_button = create_button(
    buttons_frame, 
    "🎬 บันทึก (F5)", 
    start_recording_gui, 
    COLORS['success']
)
record_button.grid(row=0, column=0, padx=10, pady=8)

play_once_button = create_button(
    buttons_frame, 
    "▶️ เล่นซ้ำ 1 รอบ (F7)", 
    lambda: start_replay_gui(is_loop=False), 
    COLORS['accent']
)
play_once_button.grid(row=0, column=1, padx=10, pady=8)

loop_play_button = create_button(
    buttons_frame, 
    "🔄 เล่นซ้ำวนลูป (F8)", 
    lambda: start_replay_gui(is_loop=True), 
    COLORS['accent']
)
loop_play_button.grid(row=1, column=1, padx=10, pady=8)

cancel_button = create_button(
    buttons_frame, 
    "❌ ยกเลิก (ESC)", 
    cancel_action, 
    COLORS['warning']
)
cancel_button.grid(row=1, column=0, padx=10, pady=8)

# Add some padding at the bottom of controls
tk.Label(controls_frame, text="", bg=COLORS['card']).pack(pady=5)

# Speed Control Section
speed_frame = tk.Frame(main_frame, bg=COLORS['card'], relief="flat", bd=2)
speed_frame.pack(fill=tk.X, pady=(0, 20))

speed_title = tk.Label(
    speed_frame, 
    text="⚡ ควบคุมความเร็ว", 
    font=("Segoe UI", 14, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['primary']
)
speed_title.pack(pady=(15, 5))

speed_control_frame = tk.Frame(speed_frame, bg=COLORS['card'])
speed_control_frame.pack(pady=10)

speed_info_frame = tk.Frame(speed_control_frame, bg=COLORS['card'])
speed_info_frame.pack()

tk.Label(
    speed_info_frame, 
    text="ช้า", 
    font=("Segoe UI", 10), 
    bg=COLORS['card'], 
    fg=COLORS['text_light']
).pack(side=tk.LEFT)

speed_value_label = tk.Label(
    speed_info_frame, 
    text="x1.0", 
    font=("Segoe UI", 14, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['accent'],
    width=8
)
speed_value_label.pack(side=tk.LEFT, padx=20)

tk.Label(
    speed_info_frame, 
    text="เร็ว", 
    font=("Segoe UI", 10), 
    bg=COLORS['card'], 
    fg=COLORS['text_light']
).pack(side=tk.RIGHT)

# Style the slider
style = ttk.Style()
style.theme_use('clam')
style.configure("Custom.Horizontal.TScale", 
               background=COLORS['card'],
               troughcolor=COLORS['border'],
               sliderrelief='flat',
               sliderlength=30)

speed_slider = ttk.Scale(
    speed_control_frame, 
    from_=0.1, 
    to=3.0, 
    value=1.0, 
    orient=tk.HORIZONTAL, 
    length=400, 
    command=update_replay_speed,
    style="Custom.Horizontal.TScale"
)
speed_slider.pack(pady=10)

# Add some padding at the bottom
tk.Label(speed_frame, text="", bg=COLORS['card']).pack(pady=5)

# Hotkey Info Section
hotkey_frame = tk.Frame(main_frame, bg=COLORS['card'], relief="flat", bd=2)
hotkey_frame.pack(fill=tk.X)

hotkey_title = tk.Label(
    hotkey_frame, 
    text="⌨️ คีย์ลัดสำหรับใช้งาน", 
    font=("Segoe UI", 12, "bold"), 
    bg=COLORS['card'], 
    fg=COLORS['primary']
)
hotkey_title.pack(pady=(15, 5))

hotkey_info = tk.Label(
    hotkey_frame, 
    text="F5: บันทึก • Esc: หยุด/ยกเลิก • F7: เล่นซ้ำ 1 รอบ • F8: เล่นซ้ำวนลูป", 
    font=("Segoe UI", 10), 
    bg=COLORS['card'], 
    fg=COLORS['text_light']
)
hotkey_info.pack(pady=(0, 15))

# Setup hotkeys
hotkeys = {
    '<f5>': hotkey_start_recording,
    '<esc>': cancel_action,
    '<f7>': hotkey_start_replay_once,
    '<f8>': hotkey_start_replay_loop,
}
hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
hotkey_listener.start()

# Initialize UI state
update_ui_for_idle()

# Center the window on screen
root.update_idletasks()
x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
root.geometry(f"+{x}+{y}")

root.mainloop()