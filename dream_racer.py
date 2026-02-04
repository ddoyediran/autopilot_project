import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.keras import layers

# --- CONFIGURATION ---
IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60
LATENT_DIM = 32
FPS = 30  # Simulation speed

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

def draw_hud(frame, steer, autopilot_steer, mode, frame_count, paused):
    # 1. Steering Bar
    bar_width = 200
    bar_height = 20
    center_x = frame.shape[1] // 2
    bar_y = frame.shape[0] - 40
    
    # Background
    cv2.rectangle(frame, (center_x - 100, bar_y), (center_x + 100, bar_y + bar_height), (50, 50, 50), -1)
    cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_height), (255, 255, 255), 1)
    
    # Actual Steering (Red Left / Green Right)
    steer_len = int(steer * 100)
    if steer_len < 0:
        cv2.rectangle(frame, (center_x + steer_len, bar_y + 2), (center_x, bar_y + bar_height - 2), (0, 0, 255), -1)
    else:
        cv2.rectangle(frame, (center_x, bar_y + 2), (center_x + steer_len, bar_y + bar_height - 2), (0, 255, 0), -1)

    # Ghost Dot (Blue)
    ai_x = int(center_x + (autopilot_steer * 100))
    cv2.circle(frame, (ai_x, bar_y + 10), 5, (255, 255, 0), -1)

    # 2. Text Info
    # Timer
    seconds = frame_count / FPS
    cv2.putText(frame, f"TIME: {seconds:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mode
    cv2.putText(frame, f"MODE: {mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Pause Overlay
    if paused:
        cv2.putText(frame, "PAUSED", (center_x - 60, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return frame

# --- MAIN LOOP ---
z_current = get_random_real_z()
autopilot_mode = False
paused = False
current_steering = 0.0
frame_count = 0

print("--- CONTROLS ---")
print(" [A] Toggle Autopilot")
print(" [Space/P] Pause/Play")
print(" [M] Manual Mode")
print(" [J / L] Steer Left / Right")
print(" [R] RESET")
print(" [Q] Quit")

while True:
    # 1. DECODE VISUALS (Always do this so we see the pause frame)
    reconstruction = decoder(z_current)
    frame = (np.array(reconstruction)[0] * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 2. GET AI OPINION
    ai_pred = controller(z_current)
    ai_steer = float(ai_pred[0][0])
    
    # 3. DRAW HUD
    mode_str = "AUTOPILOT" if autopilot_mode else "MANUAL"
    big_frame = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_NEAREST)
    final_frame = draw_hud(big_frame, current_steering, ai_steer, mode_str, frame_count, paused)
    cv2.imshow("Dream Racer v3", final_frame)
    
    # 4. INPUT HANDLING
    key = cv2.waitKey(int(1000/FPS)) # Wait depending on FPS
    
    if key == ord('q'): break
    if key == ord('a'): autopilot_mode = True
    if key == ord('m'): autopilot_mode = False
    if key == ord(' ') or key == ord('p'): paused = not paused
    
    if key == ord('r'): 
        print("Resetting...")
        z_current = get_random_real_z()
        frame_count = 0
        continue

    # --- IF PAUSED, SKIP PHYSICS ---
    if paused:
        continue

    # 5. DETERMINE ACTION
    if autopilot_mode:
        current_steering = ai_steer
    else:
        # Manual Input decay
        current_steering *= 0.8
        if abs(current_steering) < 0.05: current_steering = 0.0
        
        if key == ord('j'): current_steering = -0.8
        if key == ord('l'): current_steering = 0.8
        if key == 2: current_steering = -0.8
        if key == 3: current_steering = 0.8

    # 6. PHYSICS (World Model Step)
    rnn_z = tf.expand_dims(z_current, axis=1)
    rnn_action = tf.constant([[[current_steering]]], dtype=tf.float32)
    
    z_next = rnn([rnn_z, rnn_action])
    z_current = tf.squeeze(z_next, axis=1)
    frame_count += 1

cv2.destroyAllWindows()