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

# ==========================================
#        CONFIGURATION
# ==========================================
# 1. MODELS
ENCODER_PATH = 'wm_encoder_fixed.tflite'
CONTROLLER_PATH = 'wm_controller_fixed.tflite'
RNN_PATH = 'wm_rnn.tflite'

# 2. IMAGE
IMG_HEIGHT : int = 80
IMG_WIDTH : int = 160
CROP_TOP : int = 40
CROP_BOTTOM : int = 60

# 3. MECHANICS
STEER_SERVO_ID : int = 3
SERVO_CENTER : int = 1500

# 4. TUNING (Optimized)
MAX_TURN_OFFSET : int = 1600 # High gain for sharp turns
STEER_EXPONENT : float = 0.85 # Aggressive mid-range response
STEER_DEADZONE : float = 0.05 # Ignore small noise
MAX_SPEED : float = 0.22
MIN_SPEED : float = 0.16
BRAKE_SENSITIVITY : float = 0.6
SMOOTHING_FACTOR : float = 0.5

# 5. BLIND TRIGGER
# 0 = Black, 255 = White. Below means BLIND (No Lane Visible)
DARKNESS_THRESHOLD : float = 30.0 

class BlindPilot(Node):
    def __init__(self):
        super().__init__('blind_pilot')

        # --- 1. LOAD MODELS ---
        try:
            # Vision (Encoder Model)
            self.encoder = Interpreter(model_path=ENCODER_PATH)
            self.encoder.allocate_tensors()
            self.encoder_input = self.encoder.get_input_details()[0]['index']
            self.encoder_output = self.encoder.get_output_details()[0]['index']

            # Driver (Controller Model)
            self.controller = Interpreter(model_path=CONTROLLER_PATH)
            self.controller.allocate_tensors()
            self.controller_input = self.controller.get_input_details()[0]['index']
            self.controller_output = self.controller.get_output_details()[0]['index']

            # Memory (RNN Model)
            self.rnn = Interpreter(model_path=RNN_PATH)
            self.rnn.allocate_tensors()
            self.rnn_input_z = self.rnn.get_input_details()[0]['index']
            self.rnn_input_a = self.rnn.get_input_details()[1]['index']

            # Fix potential index swapping for RNN inputs (Common issue in TFLite models)
            if self.rnn.get_input_details()[0]['shape'][-1] == 1:
                self.rnn_input_z, self.rnn_input_a = self.rnn_input_a, self.rnn_input_z

            self.rnn_output = self.rnn.get_output_details()[0]['index']

            self.get_logger().info("SYSTEMS ONLINE: Vision + Control + Memory")
        except Exception as err:
            self.get_logger().error(f"Model Load failed: {err}")
            exit()

        # --- 2. SETUP --
        self.bridge = CvBridge()
        self.log_buffer = []
        self.start_time = time.time()

        # State Vectors
        self.last_steering = 0.0
        self.current_z = np.zeros((1, 32), dtype=np.float32)

        # --- 3. ROS SUBSCRIBERS & PUBLISHERS ---
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, 'autopilot/debug', 10)

        # Handle graceful shutdown on Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.get_logger().info("Stopping robot...")
        self.stop_robot()
        self.save_logs()
        sys.exit(0)

    def img_callback(self, msg):
        try:
            t0 = time.perf_counter()

            # 1. Get Image
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 2. Check Brightness (Trigger)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            is_blind : bool = avg_brightness < DARKNESS_THRESHOLD

            # 3. Preprocess the image (also do for debug HUD even if blind)
            if cv_img.shape[0]> (CROP_TOP + CROP_BOTTOM):
                crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
            else:
                crop_img = cv_img
            input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))

            # --- Switch Logic ---
            if not is_blind:
                # Normal Operation 'steer mode' - Process through VAE + Controller (use Camera)
                rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)

                # Encoder Image -> z (latent vector)
                self.encoder.set_tensor(self.encoder_input, input_data)
                self.encoder.invoke()
                self.current_z = self.encoder.get_tensor(self.encoder_output)

                # Visuals
                cv2.putText(input_img, f"SEEING ({int(avg_brightness)})", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Blind Mode (Use RNN Memory)
                # Ignore Image. Use previous Z + previous steering to dream next Z.

                # Reshape for RNN: [1, 1, 32] and [1, 1, 1]
                rnn_z = np.reshape(self.current_z, (1, 1, 32))
                rnn_a = np.reshape(np.array([self.last_steering], dtype=np.float32), (1, 1, 1))

                # RNN: (Z_t, A_t) -> Z_t+1
                self.rnn.set_tensor(self.rnn_input_z, rnn_z)
                self.rnn.set_tensor(self.rnn_input_a, rnn_a)
                self.rnn.invoke()

                # Update State from Dream
                prediction_z = self.rnn.get_tensor(self.rnn_output)
                self.current_z = np.reshape(prediction_z, (1, 32))

                # Visuals: Darken image and show warning
                input_img = (input_img * 0.3).astype(np.uint8) # Darken
                cv2.putText(input_img, f"BLIND {int(avg_brightness)}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(input_img, f"USING RNN MEMORY", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # === C. CONTROL (Z -> Steering) ===
            # Run Controller on whatever z we have (Real or Demand)
            self.controller.set_tensor(self.controller_input, self.current_z)
            self.controller.invoke()
            raw_prediction = self.controller.get_tensor(self.controller_output)[0][0]

            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000

            # Log and Drive
            self.drive_robot(raw_prediction, inference_ms, input_img, is_blind)

        except Exception as err:
            self.get_logger().error(f"Loop Error: {err}")

    
    def drive_robot(self, raw_prediction, inference_ms, debug_img, is_blind):
        # 1. Deadzone
        if abs(raw_prediction) < STEER_DEADZONE:
            raw_prediction = 0.0

        # Smoothing
        smoothed_prediction = (SMOOTHING_FACTOR * raw_prediction) + ((1 - SMOOTHING_FACTOR) * self.last_steering)
        self.last_steering = smoothed_prediction # Update state for next loop

        # 3. Non-Linear Gain
        curved_prediction = np.sign(smoothed_prediction) * (abs(smoothed_prediction) ** STEER_EXPONENT)

        # 4. PWM Calculation 
        pwm_offset = curved_prediction * MAX_TURN_OFFSET
        pwm_target = int(SERVO_CENTER - pwm_offset)
        pwm_target = max(700, min(2300, pwm_target))

        # 5. Dynamic Speed
        turn_severity = abs(curved_prediction)
        target_speed = MAX_SPEED - (turn_severity * BRAKE_SENSITIVITY)
        target_speed = max(MIN_SPEED, target_speed)

        # 6. Actuate
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

        # 7. Log To RAM
        elapsed = time.time() - self.start_time
        mode_flag = 1 if is_blind else 0

        self.log_buffer.append([
            f"{elapsed:.3f}",
            f"{inference_ms:.1f}",
            mode_flag,
            f"{raw_prediction:.3f}",
            f"{smoothed_prediction:.3f}",
            f"{pwm_target}",
            f"{target_speed:.2f}"
        ])

        # 8. Publish Debug Image HUD
        self.publish_hud(debug_img, smoothed_prediction, is_blind)

    def publish_hud(self, img, pred, is_blind):
        hud = img.copy()
        h, w, _ = hud.shape
        cx = w // 2
        
        # Color: Green if Seeing, Red if Blind
        color = (0, 0, 255) if is_blind else (0, 255, 0)
        
        cv2.line(hud, (cx, h), (cx, h-20), (255, 255, 255), 1)
        pred_x = int(cx + (pred * (w/2)))
        cv2.line(hud, (cx, h), (pred_x, h-20), color, 3)
        
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
        if not self.log_buffer:
            print("No logs to save.")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blind_pilot_log_{timestamp}.csv"

        print(f"\n--- SAVING FLIGHT RECORDER: {filename} ---")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Inference_ms", "Is_Blind", "Raw_AI", "Smoothed", "PWM", "Speed"])
                writer.writerows(self.log_buffer)
            print("Logs Saved.")
        except Exception as err:
            print(f"Log Save Error: {err}")

def main(args=None):
    rclpy.init(args=args)
    node = BlindPilot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # The signal_handler will handle the cleanup/logging


if __name__ == '__main__':
    main()