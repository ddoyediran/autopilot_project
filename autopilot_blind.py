import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("Error: 'ai-edge-litert' is not installed.")
    exit()

# --- FILES ---
ENCODER_PATH = 'wm_encoder_fixed.tflite'
CONTROLLER_PATH = 'wm_controller_fixed.tflite'
RNN_PATH = 'wm_rnn.tflite'

# --- CONFIG ---
IMG_HEIGHT = 80
IMG_WIDTH = 160
CROP_TOP = 40
CROP_BOTTOM = 60

# Mechanics (Your Tuned Values)
STEER_SERVO_ID = 3
SERVO_CENTER = 1500
MAX_TURN_OFFSET = 1600  
STEER_EXPONENT = 0.85   
STEER_DEADZONE = 0.05
MAX_SPEED = 0.22       
MIN_SPEED = 0.16       
BRAKE_SENSITIVITY = 0.6  
SMOOTHING_FACTOR = 0.5 

# --- BLIND EXPERIMENT SETTINGS ---
BLIND_INTERVAL = 8.0  # Go blind every 8 seconds
BLIND_DURATION = 1.0  # Stay blind for 1.0 seconds (Start small!)

class BlindPilot(Node):
    def __init__(self):
        super().__init__('blind_pilot')
        
        # 1. Load Models
        try:
            # Vision
            self.enc = Interpreter(model_path=ENCODER_PATH)
            self.enc.allocate_tensors()
            self.enc_in = self.enc.get_input_details()[0]['index']
            self.enc_out = self.enc.get_output_details()[0]['index']
            
            # Driver
            self.ctrl = Interpreter(model_path=CONTROLLER_PATH)
            self.ctrl.allocate_tensors()
            self.ctrl_in = self.ctrl.get_input_details()[0]['index']
            self.ctrl_out = self.ctrl.get_output_details()[0]['index']
            
            # Memory (RNN)
            self.rnn = Interpreter(model_path=RNN_PATH)
            self.rnn.allocate_tensors()
            self.rnn_in_z = self.rnn.get_input_details()[0]['index'] # input_1 (Z)
            self.rnn_in_a = self.rnn.get_input_details()[1]['index'] # input_2 (Action)
            self.rnn_out = self.rnn.get_output_details()[0]['index'] # Output Z
            
            self.get_logger().info("ALL SYSTEMS ONLINE: Vision + Control + Memory")
        except Exception as err:
            self.get_logger().error(f"Model Load Failed: {err}")
            exit()

        self.bridge = CvBridge()
        self.start_time = time.time()
        self.last_steering = 0.0 # For smoothing
        self.current_z = np.zeros((1, 32), dtype=np.float32) # Keep track of state
        
        # ROS
        self.sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_callback, 1)
        self.vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)
        self.debug_pub = self.create_publisher(Image, '/autopilot/debug', 10)

    def img_callback(self, msg):
        try:
            # Timer logic for Blind Mode
            now = time.time()
            elapsed = now - self.start_time
            cycle_time = elapsed % BLIND_INTERVAL
            
            # Are we blind right now?
            is_blind = cycle_time < BLIND_DURATION
            
            if not is_blind:
                # === NORMAL MODE (USE CAMERA) ===
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # Preprocess
                if cv_img.shape[0] > (CROP_TOP + CROP_BOTTOM):
                    crop_img = cv_img[CROP_TOP:-CROP_BOTTOM, :, :]
                else:
                    crop_img = cv_img
                input_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
                rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_data = (np.expand_dims(rgb_img, axis=0).astype(np.float32) / 255.0)

                # 1. Vision -> Z
                self.enc.set_tensor(self.enc_in, input_data)
                self.enc.invoke()
                # Update our global state
                self.current_z = self.enc.get_tensor(self.enc_out) 
                
                debug_img = input_img # For visualization
                
            else:
                # === BLIND MODE (USE RNN) ===
                # We do NOT process the image. We use the 'self.current_z' from the last loop
                # and the 'self.last_steering' we just performed.
                
                # RNN Inputs: [Batch, Time, Features] -> (1, 1, 32)
                rnn_z = np.expand_dims(self.current_z, axis=1)
                rnn_a = np.array([[[self.last_steering]]], dtype=np.float32)
                
                # Predict Future Z
                self.rnn.set_tensor(self.rnn_in_z, rnn_z)
                self.rnn.set_tensor(self.rnn_in_a, rnn_a)
                self.rnn.invoke()
                
                # Update global state with the dream
                # Output shape is (1, 1, 32) -> Squeeze to (1, 32)
                prediction_z = self.rnn.get_tensor(self.rnn_out)
                self.current_z = np.reshape(prediction_z, (1, 32))
                
                # Create a blank/static image for debug to show we are blind
                debug_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                cv2.putText(debug_img, "BLIND", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # === CONTROL (Use Z to Steer) ===
            # Whether Z came from Camera or RNN, the Controller doesn't care.
            self.ctrl.set_tensor(self.ctrl_in, self.current_z)
            self.ctrl.invoke()
            raw_pred = self.ctrl.get_tensor(self.ctrl_out)[0][0]

            # Drive
            self.drive_robot(raw_pred, debug_img, is_blind)

        except Exception as err:
            self.get_logger().error(f"Loop Error: {err}")

    def drive_robot(self, raw_pred, debug_img, is_blind):
        # Deadzone
        if abs(raw_pred) < STEER_DEADZONE: raw_pred = 0.0

        # Smoothing
        smoothed_pred = (SMOOTHING_FACTOR * raw_pred) + ((1.0 - SMOOTHING_FACTOR) * self.last_steering)
        self.last_steering = smoothed_pred

        # Math
        curved_pred = np.sign(smoothed_pred) * (abs(smoothed_pred) ** STEER_EXPONENT)
        pwm_target = int(SERVO_CENTER - (curved_pred * MAX_TURN_OFFSET))
        pwm_target = max(700, min(2300, pwm_target))

        turn_severity = abs(curved_pred)
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

        # Debug HUD
        self.publish_hud(debug_img, smoothed_pred, is_blind)

    def publish_hud(self, img, pred, is_blind):
        hud = img.copy()
        h, w, _ = hud.shape
        cx = w // 2
        
        color = (0, 0, 255) if is_blind else (0, 255, 0)
        
        cv2.line(hud, (cx, h), (cx, h-20), (255, 255, 255), 1)
        pred_x = int(cx + (pred * (w/2)))
        cv2.line(hud, (cx, h), (pred_x, h-20), color, 2)
        
        msg = self.bridge.cv2_to_imgmsg(hud, encoding="bgr8")
        self.debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BlindPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            stop = Twist()
            node.vel_pub.publish(stop)
            
            servo_msg = SetPWMServoState()
            servo_msg.duration = 0.5
            state_part = PWMServoState()
            state_part.id = [STEER_SERVO_ID]
            state_part.position = [SERVO_CENTER]
            state_part.offset = [0]
            servo_msg.state = [state_part]
            node.servo_pub.publish(servo_msg)
            
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()