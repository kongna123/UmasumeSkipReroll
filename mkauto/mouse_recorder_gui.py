import tkinter as tk
from tkinter import messagebox, filedialog
from pynput import mouse, keyboard
import pyautogui
import time
import threading
import ast

# ตัวแปรสถานะและการกระทำ
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

# ตัวแปรสถานะใหม่สำหรับจัดการการทำงานจากคีย์ลัด
stop_recording_signal = False
start_replay_signal = False

# ตัวแปรสำหรับกำหนดความเร็ว
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
        countdown_label.config(text=f"กำลังจะเริ่มบันทึกใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, recording_countdown_tick, seconds_left - 1)
    else:
        # เริ่มการบันทึกจริงๆ เมื่อนับถอยหลังเสร็จ
        countdown_label.config(text="กำลังบันทึก...")
        status_label.config(text="สถานะ: กำลังบันทึก", fg="#e74c3c")
        
        actions = []
        start_time = time.time()
        
        mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        mouse_listener.start()

        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        print(">>> เริ่มการบันทึกเมาส์และคีย์บอร์ด...")


def start_recording_hotkey():
    """ฟังก์ชันสำหรับการเริ่มบันทึกจากปุ่ม GUI และคีย์ลัด"""
    global is_recording, actions, mouse_listener, keyboard_listener, start_time, countdown_id
    if is_recording:
        print(">>> การบันทึกกำลังดำเนินอยู่...")
        return

    is_recording = True # ตั้งค่าสถานะเพื่อเริ่มการนับถอยหลัง
    update_ui_for_recording()
    recording_countdown_tick(10)
    print(">>> เตรียมบันทึกเมาส์และคีย์บอร์ด...")

def stop_recording_hotkey(auto_stop=False):
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
    
    update_ui_for_idle()
    print(">>> หยุดการบันทึก")
    stop_recording_signal = False

def replay_countdown_tick(seconds_left, filename):
    """ฟังก์ชันนับถอยหลังสำหรับการเล่นซ้ำ"""
    global countdown_id, is_replaying
    if not is_replaying:
        return
        
    if seconds_left > 0:
        countdown_label.config(text=f"กำลังจะเริ่มเล่นซ้ำใน {seconds_left} วินาที...")
        countdown_id = root.after(1000, replay_countdown_tick, seconds_left - 1, filename)
    else:
        replay_thread = threading.Thread(target=run_replay, args=(filename,))
        replay_thread.start()

def run_replay(filename):
    """ฟังก์ชันสำหรับเล่นซ้ำการกระทำทั้งหมด (แบบวนลูป)"""
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
        root.after(0, update_ui_for_idle)
        return
        
    pyautogui.FAILSAFE = False
    original_pause = pyautogui.PAUSE
    pyautogui.PAUSE = 0
    
    root.after(0, lambda: countdown_label.config(text="กำลังเล่นซ้ำ..."))
    
    while is_replaying:
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
        
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = original_pause
    is_replaying = False
    root.after(0, update_ui_for_idle)
    print(">>> เล่นซ้ำเสร็จสมบูรณ์แล้ว")

def start_replay_with_dialog():
    """
    ฟังก์ชันที่ถูกเรียกจากปุ่มหรือคีย์ลัดเพื่อจัดการการเลือกไฟล์
    """
    global is_replaying
    if is_replaying:
        return

    filename = filedialog.askopenfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        title="เลือกไฟล์การกระทำที่ต้องการเล่นซ้ำ"
    )

    if not filename:
        root.after(0, lambda: messagebox.showinfo("ยกเลิก", "การเล่นซ้ำถูกยกเลิก ไม่มีการเลือกไฟล์"))
        return
    
    is_replaying = True
    update_ui_for_replaying()
    
    replay_countdown_tick(10, filename)
    print(f">>> เตรียมเล่นซ้ำเมาส์จากไฟล์: {filename} ...")


def cancel_action():
    """ยกเลิกการทำงานปัจจุบันและปิดโปรแกรม"""
    global is_recording, is_replaying, hotkey_listener, countdown_id
    if is_recording:
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
        stop_recording_hotkey()
    if is_replaying:
        is_replaying = False
        if countdown_id:
            root.after_cancel(countdown_id)
            countdown_id = None
    
    if hotkey_listener:
        hotkey_listener.stop()
        
    root.destroy()
    print(">>> ปิดโปรแกรม")

# ฟังก์ชันจัดการคีย์ลัดให้ Thread-safe
def hotkey_start_recording():
    root.after(0, start_recording_hotkey)

def hotkey_stop_recording():
    global stop_recording_signal
    stop_recording_signal = True
    
def hotkey_start_replay():
    global start_replay_signal
    start_replay_signal = True

def hotkey_cancel_action():
    root.after(0, cancel_action)

# ฟังก์ชันที่จะถูกเรียกทุกๆ 100ms เพื่อตรวจสอบสัญญาณจากคีย์ลัด
def check_hotkey_signal():
    global stop_recording_signal, start_replay_signal
    
    if stop_recording_signal:
        stop_recording_hotkey()
        stop_recording_signal = False
    
    if start_replay_signal:
        start_replay_with_dialog()
        start_replay_signal = False
    
    root.after(100, check_hotkey_signal)


def update_ui_for_recording():
    record_button.config(state=tk.DISABLED)
    replay_button.config(state=tk.DISABLED)
    cancel_button.config(text="หยุดบันทึก", bg="#e74c3c")
    status_label.config(text="สถานะ: กำลังเตรียมการ...", fg="#e74c3c")
    countdown_label.config(text="", fg="blue")

def update_ui_for_replaying():
    record_button.config(state=tk.DISABLED)
    replay_button.config(state=tk.DISABLED)
    cancel_button.config(text="หยุดเล่นซ้ำ", bg="#e74c3c")
    status_label.config(text="สถานะ: กำลังเตรียมการ...", fg="#2980b9")
    countdown_label.config(text="", fg="red")

def update_ui_for_idle():
    record_button.config(state=tk.NORMAL, bg="#2ecc71")
    replay_button.config(state=tk.NORMAL, bg="#3498db")
    cancel_button.config(text="ยกเลิกการใช้งาน", bg="#f1c40f")
    status_label.config(text="สถานะ: พร้อมใช้งาน", fg="#27ae60")
    countdown_label.config(text="")

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("Mouse & Keyboard Automation")
root.geometry("400x250")
root.config(bg="#ecf0f1")

button_frame = tk.Frame(root, bg="#ecf0f1")
button_frame.pack(pady=20)

record_button = tk.Button(button_frame, text="บันทึก (รอ 10วิ)", command=start_recording_hotkey, width=15, height=2,
                          font=("Arial", 12, "bold"), fg="white", bg="#2ecc71", bd=0, activebackground="#27ae60")
record_button.grid(row=0, column=0, padx=10, pady=10)

replay_button = tk.Button(button_frame, text="เล่นซ้ำ (รอ 10วิ)", command=start_replay_with_dialog, width=15, height=2,
                          font=("Arial", 12, "bold"), fg="white", bg="#3498db", bd=0, activebackground="#2980b9")
replay_button.grid(row=0, column=1, padx=10, pady=10)

cancel_button = tk.Button(root, text="ยกเลิกการใช้งาน", command=cancel_action, width=35, height=2,
                          font=("Arial", 12, "bold"), fg="white", bg="#f1c40f", bd=0, activebackground="#f39c12")
cancel_button.pack(pady=10)

status_label = tk.Label(root, text="สถานะ: พร้อมใช้งาน", font=("Arial", 12), bg="#ecf0f1", fg="#27ae60")
status_label.pack(pady=10)

countdown_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#ecf0f1", fg="blue")
countdown_label.pack(pady=5)

# <<< แก้ไข: เปลี่ยนปุ่มลัดใหม่ทั้งหมด >>>
hotkey_label = tk.Label(root, text="คีย์ลัด: บันทึก (F5), หยุดบันทึก (F6), เล่นซ้ำ (F7), ยกเลิก (F8)", font=("Arial", 10), bg="#ecf0f1", fg="#555")
hotkey_label.pack(pady=5)

hotkeys = {
    '<f5>': hotkey_start_recording,
    '<f6>': hotkey_stop_recording,
    '<f7>': hotkey_start_replay,
    '<f8>': hotkey_cancel_action
}
hotkey_listener = keyboard.GlobalHotKeys(hotkeys)
hotkey_listener.start()

update_ui_for_idle()

root.after(100, check_hotkey_signal)

root.mainloop()