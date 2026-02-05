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
# Display Resolution (Upscaled)
DISP_W = 640
DISP_H = 320

CROP_TOP = 40
CROP_BOTTOM = 60
LATENT_DIM = 32
FPS = 30 

# --- LOAD MODELS ---
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

print("Loading Neural Engine...")
try:
    encoder = tf.keras.models.load_model('vae_encoder.keras', custom_objects={'Sampling': Sampling}, safe_mode=False)
    decoder = tf.keras.models.load_model('vae_decoder.keras', safe_mode=False)
    rnn = tf.keras.models.load_model('world_model_rnn.keras', safe_mode=False)
    controller = tf.keras.models.load_model('world_model_controller.keras', safe_mode=False)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def get_random_real_z():
    img_dir = 'training_images' 
    if not os.path.exists(img_dir): img_dir = 'data/training_images'
    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not files: return None
    path = os.path.join(img_dir, random.choice(files))
    
    img = cv2.imread(path)
    if CROP_BOTTOM > 0: img = img[CROP_TOP:-CROP_BOTTOM, :, :]
    else: img = img[CROP_TOP:, :, :]
        
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = np.array(img).astype('float32') / 255.0
    z_out = encoder(np.expand_dims(img_norm, axis=0))
    return z_out[0]

def draw_hud(frame, steer, autopilot_steer, mode, frame_count, paused, recording):
    center_x = frame.shape[1] // 2
    bar_y = frame.shape[0] - 40
    bar_height = 20
    
    # 1. Steering Bar
    cv2.rectangle(frame, (center_x - 100, bar_y), (center_x + 100, bar_y + bar_height), (50, 50, 50), -1)
    cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_height), (255, 255, 255), 1)
    
    # Actual Steering
    steer_len = int(steer * 100)
    if steer_len < 0:
        cv2.rectangle(frame, (center_x + steer_len, bar_y + 2), (center_x, bar_y + bar_height - 2), (0, 0, 255), -1)
    else:
        cv2.rectangle(frame, (center_x, bar_y + 2), (center_x + steer_len, bar_y + bar_height - 2), (0, 255, 0), -1)

    # Ghost Dot
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
        # Blinking Red Dot
        if (frame_count // 15) % 2 == 0: # Blink every half second
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 65, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# --- MAIN LOOP ---
z_current = get_random_real_z()
autopilot_mode = False
paused = False
recording = False
video_out = None
current_steering = 0.0
frame_count = 0

print("--- CONTROLS ---")
print(" [A] Toggle Autopilot")
print(" [V] Toggle RECORDING")
print(" [Space] Pause")
print(" [R] Reset")
print(" [Q] Quit")

while True:
    # 1. DECODE VISUALS
    reconstruction = decoder(z_current)
    frame = (np.array(reconstruction)[0] * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 2. UPSCALING
    big_frame = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
    
    # 3. GET AI OPINION
    ai_pred = controller(z_current)
    ai_steer = float(ai_pred[0][0])
    
    # 4. DETERMINE ACTION
    if autopilot_mode:
        current_steering = ai_steer
        mode_str = "AUTOPILOT"
    else:
        mode_str = "MANUAL"
        current_steering *= 0.8 # Decay
        if abs(current_steering) < 0.05: current_steering = 0.0

    # 5. DRAW HUD
    final_frame = draw_hud(big_frame.copy(), current_steering, ai_steer, mode_str, frame_count, paused, recording)
    
    # 6. VIDEO RECORDING
    if recording:
        if video_out is None:
            # Init Video Writer
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dream_run_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Mac compatible
            video_out = cv2.VideoWriter(filename, fourcc, float(FPS), (DISP_W, DISP_H))
            print(f"\n[REC] Started: {filename}")
        
        video_out.write(final_frame)

    cv2.imshow("Dream Racer v4", final_frame)
    
    # 7. INPUT HANDLING
    key = cv2.waitKey(int(1000/FPS))
    
    if key == ord('q'): break
    if key == ord('a'): autopilot_mode = True
    if key == ord('m'): autopilot_mode = False
    if key == ord(' ') or key == ord('p'): paused = not paused
    
    # VIDEO TOGGLE
    if key == ord('v'):
        if recording:
            recording = False
            video_out.release()
            video_out = None
            print("[REC] Stopped and Saved.")
        else:
            recording = True
    
    if key == ord('r'): 
        print("Resetting...")
        z_current = get_random_real_z()
        frame_count = 0
        continue

    if paused: continue

    # Manual Keys
    if not autopilot_mode:
        if key == ord('j'): current_steering = -0.8
        if key == ord('l'): current_steering = 0.8
        if key == 2: current_steering = -0.8
        if key == 3: current_steering = 0.8

    # 8. PHYSICS
    rnn_z = tf.expand_dims(z_current, axis=1)
    rnn_action = tf.constant([[[current_steering]]], dtype=tf.float32)
    
    z_next = rnn([rnn_z, rnn_action])
    z_current = tf.squeeze(z_next, axis=1)
    frame_count += 1

# Cleanup
if video_out is not None:
    video_out.release()
cv2.destroyAllWindows()