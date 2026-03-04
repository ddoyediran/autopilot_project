#!/usr/bin/env python3

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
import signal
import sys

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
RNN_PATH = 'wm_lnn.tflite'  # The new Liquid World Model!

# Image
IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60
HIDDEN_UNITS = 64 # Matches your LNN config

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
# Baseline MSE is ~0.004, so 0.05 represents a massive visual anomaly!
SURPRISE_THRESHOLD = 0.05 

class ActiveInferencePilot(Node):
    def __init__(self):
        super().__init__('active_inference_pilot')
        
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

            # C. Liquid World Model (LNN)
            self.rnn_interp = Interpreter(model_path=RNN_PATH)
            self.rnn_interp.allocate_tensors()
            
            # FOOLPROOF SHAPE-BASED INPUT MAPPING
            for inp in self.rnn_interp.get_input_details():
                last_dim = inp['shape'][-1]
                if last_dim == 32: self.rnn_in_z = inp['index']
                elif last_dim == 1: self.rnn_in_a = inp['index']
                elif last_dim == HIDDEN_UNITS: self.rnn_in_hx = inp['index']
                
            # FOOLPROOF SHAPE-BASED OUTPUT MAPPING
            for out in self.rnn_interp.get_output_details():
                last_dim = out['shape'][-1]
                if last_dim == 32: self.rnn_out_z = out['index']
                elif last_dim == HIDDEN_UNITS: self.rnn_out_hx = out['index']
            
            self.get_logger().info("Active Inference Core (LNN) Online.")
        except Exception as e:
            self.get_logger().error(f"Model Load Failed: {e}")
            exit()

        self.bridge = CvBridge()
        self.is_running = True
        
        # State Tracking
        self.log_buffer =[] 
        self.start_time = time.time()
        self.last_steering = 0.0
        
        # Theory of Mind Variables (Including explicit LNN Memory!)
        self.hx_state = np.zeros((1, HIDDEN_UNITS), dtype=np.float32)
        self.prev_z = np.zeros((1, 32), dtype=np.float32)
        self.prev_action = 0.0
        self.surprise_metric = 0.0

        # ROS
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, '/autopilot/debug', 10)

    def img_callback(self, msg):
        if not self.is_running: return
        try:
            # --- A. PERCEPTION ---
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_img.shape[0] > (CROP_TOP + CROP_BOTTOM):
                crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
            else:
                crop_img = cv_img
            input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
            rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            
            input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)
            self.enc_interp.set_tensor(self.enc_in, input_data)
            self.enc_interp.invoke()
            z_obs = self.enc_interp.get_tensor(self.enc_out)

            # --- B. PREDICTION (LIQUID WORLD MODEL) ---
            rnn_input_z = np.reshape(self.prev_z, (1, 1, 32))
            rnn_input_a = np.reshape(np.array([[self.prev_action]], dtype=np.float32), (1, 1, 1))
            
            self.rnn_interp.set_tensor(self.rnn_in_z, rnn_input_z)
            self.rnn_interp.set_tensor(self.rnn_in_a, rnn_input_a)
            self.rnn_interp.set_tensor(self.rnn_in_hx, self.hx_state) # Pass explicit memory
            self.rnn_interp.invoke()
            
            z_pred = self.rnn_interp.get_tensor(self.rnn_out_z)
            self.hx_state = self.rnn_interp.get_tensor(self.rnn_out_hx) # Update explicit memory
            
            # --- C. SURPRISE CALCULATION ---
            self.surprise_metric = np.mean((z_obs.flatten() - z_pred.flatten()) ** 2)

            # --- D. ACTION SELECTION ---
            self.ctrl_interp.set_tensor(self.ctrl_in, z_obs)
            self.ctrl_interp.invoke()
            raw_pred = self.ctrl_interp.get_tensor(self.ctrl_out)[0][0]

            # --- E. UPDATE PREVIOUS STATES ---
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
        pwm_target = int(SERVO_CENTER - (curved_pred * MAX_TURN_OFFSET))
        pwm_target = max(700, min(2300, pwm_target))

        # --- 2. SOCIAL NUDGE LOGIC ---
        base_speed = MAX_SPEED - (abs(curved_pred) * BRAKE_SENSITIVITY)
        
        if surprise > (SURPRISE_THRESHOLD * 2.0):
            target_speed = 0.0 
            status = "EMERGENCY STOP"
            color = (0, 0, 255) 
        elif surprise > SURPRISE_THRESHOLD:
            target_speed = CREEP_SPEED
            status = "NUDGING"
            color = (0, 255, 255) 
        else:
            target_speed = max(MIN_SPEED, base_speed)
            status = "NORMAL"
            color = (0, 255, 0)

        # --- 3. ACTUATE ---
        servo_msg = SetPWMServoState()
        servo_msg.duration = 0.05
        state_part = PWMServoState()
        state_part.id =[STEER_SERVO_ID] 
        state_part.position = [pwm_target]
        state_part.offset =[0]
        servo_msg.state = [state_part] 
        self.servo_pub.publish(servo_msg)

        twist = Twist()
        twist.linear.x = float(target_speed)
        self.vel_pub.publish(twist)

        # --- 4. LOG & DEBUG ---
        elapsed = time.time() - self.start_time
        self.log_buffer.append([f"{elapsed:.3f}", f"{surprise:.5f}", f"{raw_pred:.3f}", f"{target_speed:.2f}"])
        self.publish_hud(debug_img, surprise, status, color)

    def publish_hud(self, img, surprise, status, color):
        hud = img.copy()
        bar_len = int(min(160, surprise * 1500))
        cv2.putText(hud, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.rectangle(hud, (0, 70), (bar_len, 80), color, -1)
        cv2.putText(hud, f"Err: {surprise:.4f}", (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
        msg = self.bridge.cv2_to_imgmsg(hud, encoding="rgb8")
        self.debug_pub.publish(msg)

    def save_logs(self):
        if not self.log_buffer: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surprise_log_{timestamp}.csv"
        print(f"\nSaving Research Data: {filename}")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Surprise_MSE", "Steering", "Speed"])
                writer.writerows(self.log_buffer)
            print("Logs Saved.")
        except Exception as e:
            print(f"Failed to save logs: {e}")

def main(args=None):
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ActiveInferencePilot()
    
    def handle_sigint(sig, frame):
        print("\n!!! EMERGENCY STOP TRIGGERED !!!")
        node.is_running = False
        
        try:
            for _ in range(3):
                # STOP MOTORS
                stop = Twist()
                stop.linear.x = 0.0
                stop.angular.z = 0.0
                node.vel_pub.publish(stop)
                
                # CENTER STEERING
                servo_msg = SetPWMServoState()
                servo_msg.duration = 0.1
                state_part = PWMServoState()
                state_part.id = [STEER_SERVO_ID]
                state_part.position = [SERVO_CENTER]
                state_part.offset = [0]
                servo_msg.state = [state_part]
                node.servo_pub.publish(servo_msg)
                time.sleep(0.1) 
        except Exception as e:
            print(f"Failed to send stop command: {e}")
            
        node.save_logs()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    main()