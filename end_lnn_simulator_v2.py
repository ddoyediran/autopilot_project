import tensorflow as tf
import numpy as np
import cv2
import os
import random
import datetime
from tensorflow.keras import layers

# --- CONFIGURATION ---
IMG_HEIGHT = 80
IMG_WIDTH = 160
# Display Resolution (Upscaled for visibility)
DISP_W = 640
DISP_H = 320

CROP_TOP = 40
CROP_BOTTOM = 60
LATENT_DIM = 32
HIDDEN_UNITS = 64  # Size of the LNN memory
FPS = 30 

# ==========================================
# 1. LOAD MODELS
# ==========================================
print("Loading Liquid Neural Engine...")

# Load the Keras Decoder (Used strictly for visual rendering on the Mac)
try:
    decoder = tf.keras.models.load_model('vae_decoder.keras', safe_mode=False)
except Exception as e:
    print(f"Error loading Keras Decoder: {e}")
    exit()

# Load the TFLite Models (The actual Robot Brain)
try:
    enc_interp = tf.lite.Interpreter(model_path='wm_encoder_fixed.tflite')
    enc_interp.allocate_tensors()
    enc_in, enc_out = enc_interp.get_input_details()[0]['index'], enc_interp.get_output_details()[0]['index']

    ctrl_interp = tf.lite.Interpreter(model_path='wm_controller_fixed.tflite')
    ctrl_interp.allocate_tensors()
    ctrl_in, ctrl_out = ctrl_interp.get_input_details()[0]['index'], ctrl_interp.get_output_details()[0]['index']

    lnn_interp = tf.lite.Interpreter(model_path='wm_lnn.tflite')
    lnn_interp.allocate_tensors()

    # Foolproof shape-based mapping for the LNN
    for inp in lnn_interp.get_input_details():
        last_dim = inp['shape'][-1]
        if last_dim == 32: lnn_in_z = inp['index']
        elif last_dim == 1: lnn_in_a = inp['index']
        elif last_dim == HIDDEN_UNITS: lnn_in_hx = inp['index']

    for out in lnn_interp.get_output_details():
        last_dim = out['shape'][-1]
        if last_dim == 32: lnn_out_z = out['index']
        elif last_dim == HIDDEN_UNITS: lnn_out_hx = out['index']

except Exception as e:
    print(f"Error loading TFLite Models: {e}")
    exit()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_random_real_z():
    """Grabs a random real image and compresses it to start the dream."""
    img_dir = 'training_images' 
    if not os.path.exists(img_dir): img_dir = 'data/training_images'
    files =[f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not files: 
        print("No images found to start dream!")
        return None
    
    path = os.path.join(img_dir, random.choice(files))
    img = cv2.imread(path)
    
    if CROP_BOTTOM > 0: img = img[CROP_TOP:-CROP_BOTTOM, :, :]
    else: img = img[CROP_TOP:, :, :]
        
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = np.array(img).astype(np.float32) / 255.0
    
    # Pass through TFLite Encoder
    enc_interp.set_tensor(enc_in, np.expand_dims(img_norm, axis=0))
    enc_interp.invoke()
    return enc_interp.get_tensor(enc_out)[0] # Return the 32D vector

def draw_hud(frame, steer, autopilot_steer, mode, frame_count, paused, recording):
    """Draws the UI overlay."""
    center_x = frame.shape[1] // 2
    bar_y = frame.shape[0] - 40
    bar_height = 20
    
    # 1. Steering Bar
    cv2.rectangle(frame, (center_x - 100, bar_y), (center_x + 100, bar_y + bar_height), (50, 50, 50), -1)
    cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_height), (255, 255, 255), 1)
    
    # Actual Steering (Manual or Autopilot)
    steer_len = int(steer * 100)
    if steer_len < 0:
        cv2.rectangle(frame, (center_x + steer_len, bar_y + 2), (center_x, bar_y + bar_height - 2), (0, 0, 255), -1)
    else:
        cv2.rectangle(frame, (center_x, bar_y + 2), (center_x + steer_len, bar_y + bar_height - 2), (0, 255, 0), -1)

    # AI Ghost Dot (Shows what AI *wants* to do, even in manual mode)
    ai_x = int(center_x + (autopilot_steer * 100))
    cv2.circle(frame, (ai_x, bar_y + 10), 5, (255, 255, 0), -1)

    # 2. Text Info
    seconds = frame_count / FPS
    cv2.putText(frame, f"TIME: {seconds:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MODE: {mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 3. Status Overlays
    if paused:
        cv2.putText(frame, "PAUSED", (center_x - 60, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
    if recording:
        if (frame_count // 15) % 2 == 0: 
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 65, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
print("--- CONTROLS ---")
print(" [A] Toggle Autopilot")
print(" [J/L] Steer Left/Right (Manual)")
print(" [V] Toggle RECORDING")
print(" [Space] Pause")
print(" [R] Reset Dream")
print(" [Q] Quit")

# INITIALIZE STATE
z_current = get_random_real_z()
# THE NOTEPAD: We must start the dream with a blank memory!
hx_state = np.zeros((1, HIDDEN_UNITS), dtype=np.float32)

autopilot_mode = False
paused = False
recording = False
video_out = None
current_steering = 0.0
frame_count = 0

while True:
    if z_current is None: break

    # --- 1. DECODE VISUALS (The Dream) ---
    # Keras Decoder expects (Batch, Latent_Dim)
    reconstruction = decoder.predict(np.expand_dims(z_current, axis=0), verbose=0)
    frame = (reconstruction[0] * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Upscale for display
    big_frame = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
    
    # --- 2. GET AI OPINION ---
    ctrl_interp.set_tensor(ctrl_in, np.expand_dims(z_current, axis=0))
    ctrl_interp.invoke()
    ai_steer = float(ctrl_interp.get_tensor(ctrl_out)[0][0])
    
    # --- 3. DETERMINE ACTION ---
    if autopilot_mode:
        current_steering = ai_steer
        mode_str = "AUTOPILOT"
    else:
        mode_str = "MANUAL"
        current_steering *= 0.8 # Auto-center decay
        if abs(current_steering) < 0.05: current_steering = 0.0

    # --- 4. DRAW HUD & RENDER ---
    final_frame = draw_hud(big_frame.copy(), current_steering, ai_steer, mode_str, frame_count, paused, recording)
    
    if recording:
        if video_out is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"liquid_dream_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(filename, fourcc, float(FPS), (DISP_W, DISP_H))
            print(f"\n[REC] Started: {filename}")
        video_out.write(final_frame)

    cv2.imshow("Liquid Dream Racer", final_frame)
    
    # --- 5. INPUT HANDLING ---
    key = cv2.waitKey(int(1000/FPS))
    
    if key == ord('q'): break
    if key == ord('a'): autopilot_mode = not autopilot_mode
    if key == ord(' ') or key == ord('p'): paused = not paused
    
    if key == ord('v'):
        if recording:
            recording = False
            video_out.release()
            video_out = None
            print("[REC] Stopped and Saved.")
        else:
            recording = True
    
    if key == ord('r'): 
        print("Resetting Dream...")
        z_current = get_random_real_z()
        hx_state = np.zeros((1, HIDDEN_UNITS), dtype=np.float32) # MUST WIPE MEMORY ON RESET!
        frame_count = 0
        continue

    if paused: continue

    # Manual Keyboard Steering
    if not autopilot_mode:
        if key == ord('j'): current_steering = -0.8
        if key == ord('l'): current_steering = 0.8

    # --- 6. PHYSICS (THE LIQUID WORLD MODEL) ---
    # Predict the next frame of the dream using Z, Action, AND the Notepad (hx)
    rnn_input_z = np.reshape(z_current, (1, 1, 32)).astype(np.float32)
    rnn_input_a = np.reshape(np.array([[current_steering]], dtype=np.float32), (1, 1, 1))
    
    lnn_interp.set_tensor(lnn_in_z, rnn_input_z)
    lnn_interp.set_tensor(lnn_in_a, rnn_input_a)
    lnn_interp.set_tensor(lnn_in_hx, hx_state)
    lnn_interp.invoke()
    
    # Update the state for the next loop!
    z_next = lnn_interp.get_tensor(lnn_out_z)
    hx_state = lnn_interp.get_tensor(lnn_out_hx) 
    
    z_current = np.reshape(z_next, (32,))
    frame_count += 1

# Cleanup
if video_out is not None: video_out.release()
cv2.destroyAllWindows()