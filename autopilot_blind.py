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
ENCODER_PATH = 'wm_encoder_fixed.tflite'
CONTROLLER_PATH = 'wm_controller_fixed.tflite'
RNN_PATH = 'wm_rnn.tflite'

IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60

STEER_SERVO_ID = 3
SERVO_CENTER = 1500

# ==========================================
#        TUNING (ALIGNED WITH WM_PILOT)
# ==========================================
MAX_TURN_OFFSET = 1400   # Match wm_pilot for stability
STEER_DEADZONE = 0.08    # Increased to ignore jitter
SMOOTHING_FACTOR = 0.3   # Lower means more stability (trust history more)
MAX_SPEED = 0.22
MIN_SPEED = 0.16
BRAKE_SENSITIVITY = 0.6 
DARKNESS_THRESHOLD = 30.0 

class BlindPilot(Node):
    def __init__(self):
        super().__init__('blind_pilot')

        # --- 1. LOAD MODELS ---
        try:
            self.encoder = Interpreter(model_path=ENCODER_PATH)
            self.encoder.allocate_tensors()
            self.encoder_input = self.encoder.get_input_details()[0]['index']
            self.encoder_output = self.encoder.get_output_details()[0]['index']

            self.controller = Interpreter(model_path=CONTROLLER_PATH)
            self.controller.allocate_tensors()
            self.controller_input = self.controller.get_input_details()[0]['index']
            self.controller_output = self.controller.get_output_details()[0]['index']

            self.rnn = Interpreter(model_path=RNN_PATH)
            self.rnn.allocate_tensors()
            self.rnn_input_z = self.rnn.get_input_details()[0]['index']
            self.rnn_input_a = self.rnn.get_input_details()[1]['index']

            # Fix potential index swapping for RNN inputs
            if self.rnn.get_input_details()[0]['shape'][-1] == 1:
                self.rnn_input_z, self.rnn_input_a = self.rnn_input_a, self.rnn_input_z

            self.rnn_output = self.rnn.get_output_details()[0]['index']
            self.get_logger().info("SYSTEMS ONLINE: Vision + Control + Memory")
        except Exception as err:
            self.get_logger().error(f"Model Load failed: {err}")
            exit()

        self.bridge = CvBridge()
        self.log_buffer = []
        self.start_time = time.time()
        self.last_steering = 0.0
        self.current_z = np.zeros((1, 32), dtype=np.float32)

        # --- 3. ROS ---
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, 'autopilot/debug', 10)

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.stop_robot()
        self.save_logs()
        sys.exit(0)

    def img_callback(self, msg):
        try:
            t0 = time.perf_counter()
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 1. Darkness Detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            is_blind = avg_brightness < DARKNESS_THRESHOLD

            # 2. Crop/Resize
            if cv_img.shape[0] > (CROP_TOP + CROP_BOTTOM):
                crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
            else:
                crop_img = cv_img
            input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))

            if not is_blind:
                # --- SEEING MODE ---
                rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)

                self.encoder.set_tensor(self.encoder_input, input_data)
                self.encoder.invoke()
                self.current_z = self.encoder.get_tensor(self.encoder_output)
                
                status_text = f"SEEING ({int(avg_brightness)})"
                status_color = (0, 255, 0)
            else:
                # --- BLIND MODE (RNN Dream) ---
                rnn_z = np.reshape(self.current_z, (1, 1, 32))
                # Use the last raw prediction for the RNN feedback
                rnn_a = np.reshape(np.array([self.last_steering], dtype=np.float32), (1, 1, 1))

                self.rnn.set_tensor(self.rnn_input_z, rnn_z)
                self.rnn.set_tensor(self.rnn_input_a, rnn_a)
                self.rnn.invoke()

                prediction_z = self.rnn.get_tensor(self.rnn_output)
                self.current_z = np.reshape(prediction_z, (1, 32))

                input_img = (input_img * 0.3).astype(np.uint8) # Darken UI
                status_text = "BLIND - RNN MEMORY"
                status_color = (0, 0, 255)

            # --- CONTROL ---
            self.controller.set_tensor(self.controller_input, self.current_z)
            self.controller.invoke()
            raw_prediction = self.controller.get_tensor(self.controller_output)[0][0]

            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000

            cv2.putText(input_img, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            self.drive_robot(raw_prediction, inference_ms, input_img, is_blind)

        except Exception as err:
            self.get_logger().error(f"Loop Error: {err}")

    def drive_robot(self, raw_prediction, inference_ms, debug_img, is_blind):
        # 1. Deadzone
        if abs(raw_prediction) < STEER_DEADZONE:
            raw_prediction = 0.0

        # 2. Smoothing (Match WorldModelPilot math)
        smoothed_prediction = (SMOOTHING_FACTOR * raw_prediction) + ((1.0 - SMOOTHING_FACTOR) * self.last_steering)
        self.last_steering = smoothed_prediction

        # 3. PWM Calculation (Linear - Match WorldModelPilot)
        pwm_offset = smoothed_prediction * MAX_TURN_OFFSET
        pwm_target = int(SERVO_CENTER - pwm_offset)
        pwm_target = max(700, min(2300, pwm_target))

        # 4. Dynamic Speed
        turn_severity = abs(smoothed_prediction)
        target_speed = MAX_SPEED - (turn_severity * BRAKE_SENSITIVITY)
        target_speed = max(MIN_SPEED, target_speed)

        # 5. Actuate
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

        # 6. Logging
        elapsed = time.time() - self.start_time
        self.log_buffer.append([f"{elapsed:.3f}", f"{inference_ms:.1f}", 1 if is_blind else 0, f"{raw_prediction:.3f}", f"{smoothed_prediction:.3f}", pwm_target, f"{target_speed:.2f}"])

        self.publish_hud(debug_img, smoothed_prediction, raw_prediction, is_blind)

    def publish_hud(self, img, smooth, raw, is_blind):
        hud = img.copy()
        h, w, _ = hud.shape
        cx = w // 2
        # Center line
        cv2.line(hud, (cx, h), (cx, h-20), (0, 255, 0), 1)
        # Raw AI (Blue)
        raw_x = int(cx + (raw * (w/2)))
        cv2.line(hud, (cx, h), (raw_x, h-20), (255, 0, 0), 1)
        # Smoothed Output (Red)
        final_x = int(cx + (smooth * (w/2)))
        cv2.line(hud, (cx, h), (final_x, h-20), (0, 0, 255), 2)
        
        msg = self.bridge.cv2_to_imgmsg(hud, encoding="bgr8")
        self.debug_pub.publish(msg)

    def stop_robot(self):
        self.vel_pub.publish(Twist())
        msg = SetPWMServoState()
        msg.duration = 0.5
        s = PWMServoState()
        s.id, s.position, s.offset = [STEER_SERVO_ID], [SERVO_CENTER], [0]
        msg.state = [s]
        self.servo_pub.publish(msg)

    def save_logs(self):
        if not self.log_buffer: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"blind_pilot_log_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Inf_ms", "Blind", "Raw", "Smooth", "PWM", "Speed"])
            writer.writerows(self.log_buffer)

def main(args=None):
    rclpy.init(args=args)
    node = BlindPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()