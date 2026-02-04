import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.keras import layers

# --- CONFIGURATION ---
IMG_HEIGHT = 80
IMG_WIDTH = 160
# Match your training crop!
CROP_TOP = 40
CROP_BOTTOM = 60
LATENT_DIM = 32

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
    """Grabs a fresh real Z vector from the dataset to reset the dream."""
    img_dir = 'training_images' 
    # Handle folder variations
    if not os.path.exists(img_dir): img_dir = 'data/training_images'
    
    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not files: return None
    
    # Pick Random
    path = os.path.join(img_dir, random.choice(files))
    
    img = cv2.imread(path)
    if CROP_BOTTOM > 0:
        img = img[CROP_TOP:-CROP_BOTTOM, :, :]
    else:
        img = img[CROP_TOP:, :, :]
        
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = np.array(img).astype('float32') / 255.0
    
    # Encode
    z_out = encoder(np.expand_dims(img_norm, axis=0))
    return z_out[0]

def draw_hud(frame, steer, autopilot_steer, mode):
    # 1. Steering Bar Background
    bar_width = 200
    bar_height = 20
    center_x = frame.shape[1] // 2
    bar_y = frame.shape[0] - 40
    
    # Draw Background Box
    cv2.rectangle(frame, (center_x - 100, bar_y), (center_x + 100, bar_y + bar_height), (50, 50, 50), -1)
    # Draw Center Line
    cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_height), (255, 255, 255), 1)
    
    # 2. Actual Steering (Red/Green Bar)
    steer_len = int(steer * 100) # Scale -1.0 to 1.0 -> -100 to 100 px
    if steer_len < 0:
        cv2.rectangle(frame, (center_x + steer_len, bar_y + 2), (center_x, bar_y + bar_height - 2), (0, 0, 255), -1) # Left Red
    else:
        cv2.rectangle(frame, (center_x, bar_y + 2), (center_x + steer_len, bar_y + bar_height - 2), (0, 255, 0), -1) # Right Green

    # 3. Autopilot "Ghost" (Blue Circle)
    ai_x = int(center_x + (autopilot_steer * 100))
    cv2.circle(frame, (ai_x, bar_y + 10), 5, (255, 255, 0), -1)

    # 4. Text
    cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Frames: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if mode == "MANUAL":
        cv2.putText(frame, "(Blue Dot = AI Recommendation)", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return frame

# --- MAIN LOOP ---
z_current = get_random_real_z()
autopilot_mode = False
current_steering = 0.0
frame_count = 0

print("--- CONTROLS ---")
print(" [A] Toggle Autopilot")
print(" [M] Manual Mode")
print(" [J / L] Steer Left / Right")
print(" [R] RESET (Fixes Vanishing Track)")
print(" [Q] Quit")

while True:
    # 1. DECODE DREAM
    reconstruction = decoder(z_current)
    frame = (np.array(reconstruction)[0] * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 2. GET AI OPINION (What would the robot do?)
    ai_pred = controller(z_current)
    ai_steer = float(ai_pred[0][0])
    
    # 3. DETERMINE ACTION
    if autopilot_mode:
        current_steering = ai_steer
        mode_str = "AUTOPILOT"
    else:
        mode_str = "MANUAL"
        # Manual Input decay (return to center)
        current_steering *= 0.8
        if abs(current_steering) < 0.05: current_steering = 0.0

    # 4. DRAW HUD
    frame = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_NEAREST)
    frame = draw_hud(frame, current_steering, ai_steer, mode_str)
    
    cv2.imshow("Dream Racer v2", frame)
    
    # 5. INPUT HANDLING
    key = cv2.waitKey(30) # 33ms = ~30 FPS
    
    if key == ord('q'): break
    if key == ord('a'): autopilot_mode = True
    if key == ord('m'): autopilot_mode = False
    
    if key == ord('r'): 
        # RESET!
        print("Resetting to new real location...")
        z_current = get_random_real_z()
        frame_count = 0
        continue

    # Manual Keys
    if not autopilot_mode:
        if key == ord('j'): current_steering = -0.8
        if key == ord('l'): current_steering = 0.8
        # Arrow keys support (Mac)
        if key == 2: current_steering = -0.8
        if key == 3: current_steering = 0.8

    # 6. PHYSICS (The World Model Step)
    # Z + Action -> Next Z
    rnn_z = tf.expand_dims(z_current, axis=1)
    rnn_action = tf.constant([[[current_steering]]], dtype=tf.float32)
    
    z_next = rnn([rnn_z, rnn_action])
    z_current = tf.squeeze(z_next, axis=1)
    frame_count += 1

cv2.destroyAllWindows()