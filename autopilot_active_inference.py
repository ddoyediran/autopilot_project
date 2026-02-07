import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
import datetime

# --- LiteRT IMPORT ---
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("Error: 'ai-edge-litert' is not installed.")
    exit()

# ==========================================
#        RESEARCH CONFIGURATION
# ==========================================
ENCODER_PATH = 'wm_encoder_fixed.tflite'
CONTROLLER_PATH = 'wm_controller_fixed.tflite'
RNN_PATH = 'wm_rnn.tflite'

# Image
IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60

# Mechanics
STEER_SERVO_ID = 3
SERVO_CENTER = 1500
MAX_TURN_OFFSET = 1350
STEER_EXPONENT = 1.1

# Speed & Safety
MAX_SPEED = 0.22        
MIN_SPEED = 0.15        
CREEP_SPEED = 0.12      
BRAKE_SENSITIVITY = 0.5
SMOOTHING_FACTOR = 0.6

# --- ACTIVE INFERENCE SETTINGS ---
SURPRISE_THRESHOLD = 0.05 

class ActiveInferencePilot(Node):
    def __init__(self):
        super().__init__('active_inference_pilot')
        
        # 1. Load Models
        try:
            # A. Vision (Encoder)
            self.enc_interp = Interpreter(model_path=ENCODER_PATH)
            self.enc_interp.allocate_tensors()
            self.enc_in = self.enc_interp.get_input_details()[0]['index']
            self.enc_out = self.enc_interp.get_output_details()[0]['index']
            
            # B. Policy (Controller)
            self.ctrl_interp = Interpreter(model_path=CONTROLLER_PATH)
            self.ctrl_interp.allocate_tensors()
            self.ctrl_in = self.ctrl_interp.get_input_details()[0]['index']
            self.ctrl_out = self.ctrl_interp.get_output_details()[0]['index']

            # C. World Model (RNN) - The Dreamer
            self.rnn_interp = Interpreter(model_path=RNN_PATH)
            self.rnn_interp.allocate_tensors()
            self.rnn_out = self.rnn_interp.get_output_details()[0]['index']
            
            # --- FIX: DYNAMIC INPUT DETECTION ---
            # We check the shapes to find which input is Z (32) and which is Action (1)
            rnn_inputs = self.rnn_interp.get_input_details()
            self.rnn_in_z = None
            self.rnn_in_a = None
            
            for i, inp in enumerate(rnn_inputs):
                shape = inp['shape']
                # Shape is likely [1, 1, 32] or [1, 1, 1]
                last_dim = shape[-1]
                
                if last_dim == 32:
                    self.rnn_in_z = inp['index']
                    self.get_logger().info(f"RNN Input Z found at index {inp['index']}")
                elif last_dim == 1:
                    self.rnn_in_a = inp['index']
                    self.get_logger().info(f"RNN Input Action found at index {inp['index']}")
            
            if self.rnn_in_z is None or self.rnn_in_a is None:
                self.get_logger().error("Failed to identify RNN inputs by shape!")
                exit()
            
            self.get_logger().info("Active Inference Core Online.")
        except Exception as e:
            self.get_logger().error(f"Model Load Failed: {e}")
            exit()

        self.bridge = CvBridge()
        
        # State Tracking
        self.log_buffer = [] 
        self.start_time = time.time()
        self.last_steering = 0.0
        
        # Theory of Mind Variables
        self.prev_z = np.zeros((1, 32), dtype=np.float32)
        self.prev_action = 0.0
        self.surprise_metric = 0.0

        # ROS
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, '/autopilot/debug', 10)

    def img_callback(self, msg):
        try:
            # --- A. PERCEPTION ---
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_img.shape[0] > (CROP_TOP + CROP_BOTTOM):
                crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
            else:
                crop_img = cv_img
            input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
            rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            
            # Current Observation (z_obs)
            input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)
            self.enc_interp.set_tensor(self.enc_in, input_data)
            self.enc_interp.invoke()
            z_obs = self.enc_interp.get_tensor(self.enc_out)

            # --- B. PREDICTION (WORLD MODEL) ---
            rnn_input_z = np.reshape(self.prev_z, (1, 1, 32))
            rnn_input_a = np.reshape(np.array([[self.prev_action]], dtype=np.float32), (1, 1, 1))
            
            self.rnn_interp.set_tensor(self.rnn_in_z, rnn_input_z)
            self.rnn_interp.set_tensor(self.rnn_in_a, rnn_input_a)
            self.rnn_interp.invoke()
            z_pred = self.rnn_interp.get_tensor(self.rnn_out)
            
            # --- C. SURPRISE CALCULATION ---
            # MSE between Expectation and Reality
            z_pred_flat = z_pred.flatten()
            z_obs_flat = z_obs.flatten()
            error = np.mean((z_obs_flat - z_pred_flat) ** 2)
            self.surprise_metric = error

            # --- D. ACTION SELECTION ---
            self.ctrl_interp.set_tensor(self.ctrl_in, z_obs)
            self.ctrl_interp.invoke()
            raw_pred = self.ctrl_interp.get_tensor(self.ctrl_out)[0][0]

            # --- E. UPDATE MEMORY ---
            self.prev_z = z_obs
            self.prev_action = raw_pred

            # --- F. DRIVE ---
            self.drive_robot(raw_pred, self.surprise_metric, input_img)

        except Exception as e:
            self.get_logger().error(f"Loop Error: {e}")

    def drive_robot(self, raw_pred, surprise, debug_img):
        # 1. Standard Steering Logic
        smoothed_pred = (SMOOTHING_FACTOR * raw_pred) + ((1.0 - SMOOTHING_FACTOR) * self.last_steering)
        self.last_steering = smoothed_pred
        
        curved_pred = np.sign(smoothed_pred) * (abs(smoothed_pred) ** STEER_EXPONENT)
        pwm_offset = curved_pred * MAX_TURN_OFFSET
        pwm_target = int(SERVO_CENTER - pwm_offset)
        pwm_target = max(700, min(2300, pwm_target))

        # --- 2. SOCIAL NUDGE LOGIC ---
        
        # Calculate Base Speed (Slow down for turns)
        base_speed = MAX_SPEED - (abs(curved_pred) * BRAKE_SENSITIVITY)
        
        # Determine Status based on Surprise
        if surprise > (SURPRISE_THRESHOLD * 2.0):
            # LEVEL 3: PANIC / EMERGENCY STOP
            target_speed = 0.0 
            status = "EMERGENCY STOP"
            color = (0, 0, 255) # Red
            
        elif surprise > SURPRISE_THRESHOLD:
            # LEVEL 2: SOCIAL NUDGE
            target_speed = CREEP_SPEED
            status = "NUDGING"
            color = (0, 255, 255) # Yellow/Cyan
            
        else:
            # LEVEL 1: NORMAL DRIVING
            target_speed = max(MIN_SPEED, base_speed)
            status = "NORMAL"
            color = (0, 255, 0) # Green

        # --- 3. ACTUATE ---
        
        # Steering
        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.05
        state_part = PWMServoState()
        state_part.id = [STEER_SERVO_ID] 
        state_part.position = [pwm_target]
        state_part.offset = [0]
        servo_msg.state = [state_part] 
        self.servo_pub.publish(servo_msg)

        # Motor
        twist = Twist()
        twist.linear.x = float(target_speed)
        self.vel_pub.publish(twist)

        # --- 4. LOG & DEBUG ---
        elapsed = time.time() - self.start_time
        self.log_buffer.append([f"{elapsed:.3f}", f"{surprise:.5f}", f"{raw_pred:.3f}", f"{target_speed:.2f}"])
        self.publish_hud(debug_img, surprise, status, color)

    def publish_hud(self, img, surprise, status, color):
        hud = img.copy()
        
        # Surprise Bar
        bar_len = int(surprise * 1500) # Scale for visibility
        bar_len = min(160, bar_len)
        
        # Draw Status Text
        cv2.putText(hud, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw Bar
        cv2.rectangle(hud, (0, 70), (bar_len, 80), color, -1)
        cv2.putText(hud, f"Err: {surprise:.3f}", (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))

        msg = self.bridge.cv2_to_imgmsg(hud, encoding="bgr8")
        self.debug_pub.publish(msg)

    def save_logs(self):
        if not self.log_buffer: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surprise_log_{timestamp}.csv"
        print(f"Saving Research Data: {filename}")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Surprise_MSE", "Steering", "Speed"])
                writer.writerows(self.log_buffer)
        except Exception as e:
            print(e)

def main(args=None):
    rclpy.init(args=args)
    node = ActiveInferencePilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vel_pub.publish(Twist()) # Stop
        node.save_logs()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()