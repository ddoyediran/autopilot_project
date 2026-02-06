import numpy as np
import tensorflow as tf
import cv2
import os
import time
from tensorflow.keras import layers

# --- CONFIGURATION ---
IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60

# --- 1. DEFINE CUSTOM LAYER (For Decoder) ---
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- 2. LOAD MODELS ---
def load_tflite(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def load_system():
    print("Loading models...")
    
    # A. TFLite Models (Encoder & Controller) - Testing what the robot uses
    enc = load_tflite('wm_encoder_fixed.tflite')
    ctrl = load_tflite('wm_controller_fixed.tflite')
    
    # B. Keras Models (RNN & Decoder) - Running natively on Mac
    # We use Keras for RNN to avoid the "Flex Delegate" error on Mac
    try:
        if not os.path.exists('world_model_rnn.keras'):
            print("Error: world_model_rnn.keras not found. Please download it from Colab.")
            return None, None, None, None
            
        rnn = tf.keras.models.load_model('world_model_rnn.keras', safe_mode=False)
        decoder = tf.keras.models.load_model('vae_decoder.keras', safe_mode=False)
        
    except Exception as e:
        print(f"Error loading Keras models: {e}")
        return None, None, None, None

    return enc, ctrl, rnn, decoder

def get_start_image():
    img_dir = 'training_images'
    if not os.path.exists(img_dir): img_dir = 'data/training_images'
        
    if not os.path.exists(img_dir):
        print("Error: 'training_images' folder not found.")
        return None

    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not files:
        print("Error: No images found.")
        return None
        
    # Pick Random
    choice = np.random.choice(files)
    path = os.path.join(img_dir, choice)
    print(f"Starting dream from: {choice}")
    
    img = cv2.imread(path)
    if CROP_BOTTOM > 0:
        img = img[CROP_TOP:-CROP_BOTTOM, :, :]
    else:
        img = img[CROP_TOP:, :, :]
        
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_norm, axis=0)

# --- 3. MAIN SIMULATION LOOP ---
def run_interactive_sim():
    enc, ctrl, rnn, decoder = load_system()
    if not enc or not rnn: return

    # Prepare TFLite Indices
    enc_in = enc.get_input_details()[0]['index']
    enc_out = enc.get_output_details()[0]['index']
    
    ctrl_in = ctrl.get_input_details()[0]['index']
    ctrl_out = ctrl.get_output_details()[0]['index']

    # State Variables
    z_current = None
    paused = False
    recording = False
    video_writer = None
    steering = 0.0
    
    def reset_simulation():
        img = get_start_image()
        if img is None: return None
        # 1. ENCODE (TFLite)
        enc.set_tensor(enc_in, img)
        enc.invoke()
        return enc.get_tensor(enc_out)

    z_current = reset_simulation() # Shape (1, 32)
    if z_current is None: return

    print("\n=== CONTROLS ===")
    print("[SPACE] : Pause/Resume")
    print("[ R ]   : Start/Stop Recording")
    print("[ N ]   : New Start Image")
    print("[ Q ]   : Quit")
    print("================\n")

    while True:
        if not paused:
            # --- A. CONTROL (TFLite) ---
            # Input: Z (1, 32) -> Output: Steering
            ctrl.set_tensor(ctrl_in, z_current)
            ctrl.invoke()
            steering = ctrl.get_tensor(ctrl_out)[0][0]

            # --- B. PREDICT (Keras RNN) ---
            # Prepare inputs for Keras:
            # Z needs to be (Batch, Time, Feat) -> (1, 1, 32)
            # Action needs to be (Batch, Time, Feat) -> (1, 1, 1)
            
            rnn_in_z = np.reshape(z_current, (1, 1, 32))
            rnn_in_a = np.reshape(np.array([steering], dtype=np.float32), (1, 1, 1))
            
            # Predict Next Z
            # Note: We use .predict() or direct call. Direct call is faster in loops.
            z_next = rnn([rnn_in_z, rnn_in_a], training=False)
            
            # Keras returns EagerTensor. Convert to Numpy.
            # Output is (1, 1, 32). Reshape to (1, 32)
            z_current = np.reshape(z_next.numpy(), (1, 32))
            
            # --- C. VISUALIZE (Keras Decoder) ---
            reconstruction = decoder(z_current, training=False)
            frame_data = reconstruction.numpy()[0]
            
            # Display
            frame = (frame_data * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            display_frame = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_NEAREST)

        # Draw HUD
        hud = display_frame.copy()
        cx = 160
        tx = int(cx + (steering * 80))
        cv2.line(hud, (cx, 150), (tx, 150), (0, 0, 255), 4)
        
        status = "PAUSED" if paused else "RUNNING"
        color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(hud, f"{status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(hud, f"Steer: {steering:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if recording:
            cv2.circle(hud, (300, 20), 5, (0, 0, 255), -1)
            if video_writer: video_writer.write(hud)

        cv2.imshow("Hybrid World Model Sim", hud)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('n'): z_current = reset_simulation()
        elif key == ord('r'):
            if not recording:
                recording = True
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(f'sim_{int(time.time())}.mp4', fourcc, 20.0, (320, 160))
                print("Recording Started...")
            else:
                recording = False
                if video_writer: video_writer.release()
                print("Recording Saved.")

    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_interactive_sim()