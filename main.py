
import asyncio
import threading
import tkinter as tk
from tkinter import messagebox
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import pywt
import io
import cv2
import os
from tkinter import ttk
from datetime import datetime
import pyttsx3
from tensorflow.keras.models import load_model

# --- CONFIG ---
DEVICE_NAME = "ESP32_PPG_Red"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.h5")
image_path = os.path.join(os.path.dirname(__file__), "vnit.png")
MAX_SAMPLES = 3500
TARGET_WIDTH, TARGET_HEIGHT = 70, 75

# --- GLOBALS ---
client = None
device_address = None
is_connected = False
loop = asyncio.new_event_loop()
ppg_data = []
tts_engine = pyttsx3.init()

# --- Load Model ---
model = load_model(MODEL_PATH)


# --- Helper Functions ---
def create_scalogram(signal_data):
    scales = np.arange(1, 65)
    coeffs, _ = pywt.cwt(signal_data, scales=scales, wavelet='gaus1')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.abs(coeffs), aspect='auto', cmap='viridis')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def handle_data(data):
    global ppg_data
    if not is_connected:
        return
    try:
        ppg_value = np.frombuffer(data, dtype=np.float32)[0]
        ppg_data.append(ppg_value)
        if len(ppg_data) > MAX_SAMPLES:
            ppg_data = ppg_data[-MAX_SAMPLES:]
        if len(ppg_data) % 100 == 0:
            update_plot()
        sample_progress['value'] = len(ppg_data)
        progress_value_label.config(text=f"{len(ppg_data)} / {MAX_SAMPLES}")
        if len(ppg_data) == MAX_SAMPLES:
            segment = ppg_data.copy()
            scalogram = create_scalogram(segment)
            prediction = model.predict(scalogram, verbose=0)
            prediction_value = prediction[0][0]
            result = "AF DETECTED" if prediction_value > 0.5 else "NON_AF DETECTED"

            display_text = result
            prediction_label.config(text=f"Prediction: {display_text}")
            tts_engine.say(display_text)
            tts_engine.runAndWait()

            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = f"[{timestamp}] {display_text}"
            history_listbox.insert(tk.END, entry)
            history_listbox.itemconfig(tk.END, {'fg': 'red' if 'AF' in result else 'green'})

            root.update()
            ppg_data.clear()
            sample_progress['value'] = 0
            progress_value_label.config(text=f"0 / {MAX_SAMPLES}")

    except Exception as e:
        print(f"Error processing data: {e}")

# --- BLE Functions ---
async def notification_handler(sender, data):
    handle_data(data)

def clear_history():
    history_listbox.delete(0, tk.END)

def restart_collection():
    global ppg_data
    ppg_data.clear()
    ax.clear()
    ax.set_title("PPG Signal")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    canvas.draw()
    sample_progress['value'] = 0
    progress_value_label.config(text=f"0 / {MAX_SAMPLES}")
    prediction_label.config(text="Waiting for Prediction...")



async def connect_bluetooth():
    global client, device_address, is_connected
    connect_button.config(state='disabled')
    status_label.config(text="Status: Scanning...")
    try:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name and DEVICE_NAME in d.name:
                device_address = d.address
                break
        if not device_address:
            status_label.config(text="Status: Device not found.")
            connect_button.config(state='normal')
            return
        client = BleakClient(device_address)
        await client.connect()
        if client.is_connected:
            is_connected = True
            status_label.config(text=f"Status: Connected to {DEVICE_NAME}")
            await client.start_notify(CHAR_UUID, notification_handler)
        else:
            status_label.config(text="Status: Failed to connect.")
    except Exception as e:
        status_label.config(text=f"Status: Error: {e}")
    finally:
        connect_button.config(state='normal')

async def disconnect_bluetooth():
    global is_connected
    try:
        if client:
            await client.stop_notify(CHAR_UUID)
            await client.disconnect()
            if not client.is_connected:
                is_connected = False
                status_label.config(text="Status: Disconnected")
            else:
                status_label.config(text="Status: Failed to disconnect")
        else:
            status_label.config(text="Status: No active connection")
    except Exception as e:
        status_label.config(text=f"Status: Error: {e}")

def connect_thread():
    asyncio.run_coroutine_threadsafe(connect_bluetooth(), loop)

def disconnect_thread():
    asyncio.run_coroutine_threadsafe(disconnect_bluetooth(), loop)

def update_plot():
    ax.clear()
    ax.plot(ppg_data[-500:], color='red')
    ax.set_title("PPG Signal")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    canvas.draw()

def run_event_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

# --- GUI Setup ---
root = tk.Tk()
root.title("PPG AF Detection via ESP32 BLE")
root.geometry("1800x1000")
root.resizable(False, False)
root.configure(bg='#e3f2fd')

main_frame = tk.Frame(root, bg='#e3f2fd')
main_frame.pack(fill='both', expand=True)

title_frame = tk.Frame(main_frame, bg='#e3f2fd')
title_frame.pack(pady=10)

def load_image(path, size):
    try:
        img = Image.open(path)
        img = img.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

logo_img = load_image(image_path, (200, 150))
if logo_img:
    logo_label = tk.Label(title_frame, image=logo_img, bg='#e3f2fd')
    logo_label.image = logo_img
    logo_label.pack(side='left', padx=10)

text_frame = tk.Frame(title_frame, bg='#e3f2fd')
text_frame.pack(side='left')

tk.Label(text_frame, text="Visvesvaraya National Institute of Technology", font=("Arial", 40, "bold"), bg='#e3f2fd', fg='#0d47a1').pack()
tk.Label(text_frame, text="Department of Electronics and Communication Engineering", font=("Arial", 34), bg='#e3f2fd', fg='#0d47a1').pack(pady=(0,10))
tk.Label(main_frame, text="PPG AF Detection System", font=("Arial", 34, "bold"), bg='#e3f2fd', fg='#0d47a1').pack(pady=5)


button_frame = tk.Frame(main_frame, bg='#e3f2fd')
button_frame.pack(pady=15)

connect_button = tk.Button(button_frame, text="Scan & Connect", command=connect_thread, width=20,font=("Arial", 20), bg='#003366', fg='white', activebackground='#002244', activeforeground='white')
connect_button.pack(side='left', padx=10)

disconnect_button = tk.Button(button_frame, text="Disconnect", command=disconnect_thread, width=20, font=("Arial", 20), bg='#003366', fg='white', activebackground='#002244', activeforeground='white')
disconnect_button.pack(side='left', padx=10)

restart_button = tk.Button(
    button_frame,
    text="Restart",
    command=restart_collection,
    width=20,
    font=("Arial", 20),
    bg='#003366',
    fg='white',
    activebackground='#002244',
    activeforeground='white'
)
restart_button.pack(side='left', padx=10)


status_label = tk.Label(main_frame, text="Status: Not Connected", font=("Arial", 20), bg='#e3f2fd', fg='#0d47a1')
status_label.pack(pady=10)

# NEW: Top Frame to hold plot and history side-by-side
# Top Frame to hold plot and history side-by-side
top_frame = tk.Frame(main_frame, bg='white')
top_frame.pack(pady=10, padx=40, fill='x')  # ⬅ Added horizontal padding

# Plot Frame (left side)
plot_frame = tk.Frame(top_frame, bg='white', bd=2, relief='groove')  # ⬅ Optional border
plot_frame.pack(side='left', padx=20, pady=10)  # ⬅ Spacing between plot and history

fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='white')
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(padx=10, pady=10)

# History Frame (right side of plot)
history_frame = tk.Frame(top_frame, bg='#e3f2fd', bd=2, relief='groove')
history_frame.pack(side='left', padx=20, pady=10, fill='both', expand=True)

tk.Label(
    history_frame,
    text="Prediction History",
    font=("Arial", 15, "bold"),
    bg='#e3f2fd',
    fg='#0d47a1'
).pack(pady=(0, 5))

history_listbox = tk.Listbox(
    history_frame, 
    width=40, 
    height=10, 
    font=("Arial", 20)
)
history_listbox.pack(fill='both', expand=True, pady=(0, 5), padx=10)  # ⬅ Added padx

clear_button = tk.Button(
    history_frame,
    text="Clear History",
    command=clear_history,
    font=("Arial", 15),
    bg='#003366',
    fg='white',
    relief='raised'
)
clear_button.pack(pady=5)

# Progress Frame for showing sample collection visually
# Status frame to hold progress and prediction side by side
status_frame = tk.Frame(main_frame, bg='#e3f2fd')
status_frame.pack(pady=20, fill='x', padx=40)

# Progress Frame (left side)
progress_frame = tk.Frame(status_frame, bg='#e3f2fd')
progress_frame.pack(side='left', padx=30, pady=10)

progress_label = tk.Label(
    progress_frame, 
    text="Samples Collected:", 
    font=("Arial", 14), 
    bg='#e3f2fd', 
    fg='#0d47a1'
)
progress_label.pack()

sample_progress = ttk.Progressbar(
    progress_frame, 
    orient='horizontal', 
    length=1000, 
    mode='determinate', 
    maximum=MAX_SAMPLES
)
sample_progress.pack(pady=5)

progress_value_label = tk.Label(
    progress_frame, 
    text=f"0 / {MAX_SAMPLES}", 
    font=("Arial", 20), 
    bg='#e3f2fd', 
    fg='#0d47a1'
)
progress_value_label.pack()

# Prediction Frame (right side)
prediction_frame = tk.Frame(status_frame, bg='#e3f2fd')
prediction_frame.pack(side='right', padx=30, pady=10)

prediction_label = tk.Label(
    prediction_frame,
    text="Waiting for Prediction...",
    font=("Helvetica", 20, "bold"),
    bg='#fffde7',
    fg='#0d47a1',
    padx=20,
    pady=10,
    relief='ridge',
    borderwidth=2
)
prediction_label.pack()


threading.Thread(target=run_event_loop, daemon=True).start()
root.mainloop()