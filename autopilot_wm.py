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
import os
import signal
import sys

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("Error: 'ai-edge-litert' is not installed.")
    exit()

# --- CONFIGURATION ---
ENCODER_PATH = 'wm_encoder_fixed.tflite'
CONTROLLER_PATH = 'wm_controller_fixed.tflite'

IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60

STEER_SERVO_ID = 3
SERVO_CENTER = 1500

# --- TUNING (STABILITY UPDATE) ---
# Reduced gain slightly to stop over-correcting
MAX_TURN_OFFSET = 1400  
# Increased Deadzone to ignore small noise
STEER_DEADZONE = 0.08   
# Increased Smoothing (0.3 means trust new data less, trust history more)
SMOOTHING_FACTOR = 0.3  

MAX_SPEED = 0.22       
MIN_SPEED = 0.16       
BRAKE_SENSITIVITY = 0.6  

class WorldModelPilot(Node):
    def __init__(self):
        super().__init__('wm_pilot')
        
        # Load Models
        try:
            self.enc_interp = Interpreter(model_path=ENCODER_PATH)
            self.enc_interp.allocate_tensors()
            self.enc_in = self.enc_interp.get_input_details()[0]['index']
            self.enc_out = self.enc_interp.get_output_details()[0]['index']
            
            self.ctrl_interp = Interpreter(model_path=CONTROLLER_PATH)
            self.ctrl_interp.allocate_tensors()
            self.ctrl_in = self.ctrl_interp.get_input_details()[0]['index']
            self.ctrl_out = self.ctrl_interp.get_output_details()[0]['index']
            self.get_logger().info("World Model Loaded.")
        except Exception as e:
            self.get_logger().error(f"Model Load Failed: {e}")
            exit()

        self.bridge = CvBridge()
        self.log_buffer = [] 
        self.start_time = time.time()
        self.last_steering = 0.0 
        
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, '/autopilot/debug', 10)

        # Handle Ctrl+C correctly
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.get_logger().info("Stopping robot...")
        self.stop_robot()
        self.save_logs()
        sys.exit(0)

    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_img.shape[0] > (CROP_TOP + CROP_BOTTOM):
                crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
            else:
                crop_img = cv_img

            input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
            rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)

            t0 = time.perf_counter()
            
            self.enc_interp.set_tensor(self.enc_in, input_data)
            self.enc_interp.invoke()
            z_vector = self.enc_interp.get_tensor(self.enc_out) 

            self.ctrl_interp.set_tensor(self.ctrl_in, z_vector)
            self.ctrl_interp.invoke()
            raw_pred = self.ctrl_interp.get_tensor(self.ctrl_out)[0][0]
            
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000

            self.drive_robot(raw_pred, inference_ms, input_img)

        except Exception as e:
            self.get_logger().error(f"Loop Error: {e}")

    def drive_robot(self, raw_pred, inf_ms, debug_img):
        # 1. Deadzone
        if abs(raw_pred) < STEER_DEADZONE:
            raw_pred = 0.0

        # 2. HEAVY Smoothing (Low Pass Filter)
        smoothed_pred = (SMOOTHING_FACTOR * raw_pred) + ((1.0 - SMOOTHING_FACTOR) * self.last_steering)
        self.last_steering = smoothed_pred

        # 3. Simple Linear Gain (Removed Exponent to reduce twitchiness)
        pwm_offset = smoothed_pred * MAX_TURN_OFFSET
        pwm_target = SERVO_CENTER - pwm_offset 
        pwm_target = int(max(700, min(2300, pwm_target)))

        # 4. Dynamic Speed
        turn_severity = abs(smoothed_pred)
        target_speed = MAX_SPEED - (turn_severity * BRAKE_SENSITIVITY)
        target_speed = max(MIN_SPEED, target_speed)

        # Actuate
        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.05
        state_part = PWMServoState()
        state_part.id = [STEER_SERVO_ID] 
        state_part.position = [pwm_target]
        state_part.offset = [0]
        servo_msg.state = [state_part] 
        self.servo_pub.publish(servo_msg)

        twist = Twist()
        twist.linear.x = float(target_speed)
        self.vel_pub.publish(twist)

        # Log
        elapsed = time.time() - self.start_time
        self.log_buffer.append([
            f"{elapsed:.3f}", 
            f"{inf_ms:.1f}", 
            f"{raw_pred:.3f}", 
            f"{smoothed_pred:.3f}", 
            pwm_target, 
            f"{target_speed:.2f}"
        ])

        self.publish_hud(debug_img, smoothed_pred, raw_pred)

    def publish_hud(self, img, smooth, raw):
        hud = img.copy()
        h, w, _ = hud.shape
        cx = w // 2
        cv2.line(hud, (cx, h), (cx, h-20), (0, 255, 0), 1)
        raw_x = int(cx + (raw * (w/2)))
        cv2.line(hud, (cx, h), (raw_x, h-20), (255, 0, 0), 1)
        final_x = int(cx + (smooth * (w/2)))
        cv2.line(hud, (cx, h), (final_x, h-20), (0, 0, 255), 2)
        msg = self.bridge.cv2_to_imgmsg(hud, encoding="bgr8")
        self.debug_pub.publish(msg)

    def stop_robot(self):
        stop = Twist()
        self.vel_pub.publish(stop)
        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.5
        state_part = PWMServoState()
        state_part.id = [STEER_SERVO_ID]
        state_part.position = [SERVO_CENTER]
        state_part.offset = [0]
        servo_msg.state = [state_part]
        self.servo_pub.publish(servo_msg)

    def save_logs(self):
        if not self.log_buffer: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_log_{timestamp}.csv"
        print(f"\n--- SAVING FLIGHT RECORDER: {filename} ---")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Inference_ms", "Raw_AI", "Smoothed", "PWM", "Speed"])
                writer.writerows(self.log_buffer)
            print("Logs Saved.")
        except Exception as e:
            print(f"Log Save Failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WorldModelPilot()
    rclpy.spin(node)

if __name__ == '__main__':
    main()