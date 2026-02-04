import os
import logging
import cv2
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_types_from_msg, get_typestore, Stores

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ------ CONFIGURATION ------
BAG_FOLDERS = ['my_road_data_b', 'my_road_data_d', 'my_road_data_c', 'my_road_data_a']
OUTPUT_IMG_DIR = 'training_images'
OUTPUT_CSV = 'driving_log.csv'

# Servo Settings (ID 3)
SERVO_CENTER = 1500
SERVO_MAX_LEFT = 2300
SERVO_MAX_RIGHT = 700


# 1. INITIALIZE TYPESTORE (The modern way to handle custom msgs)
typestore = get_typestore(Stores.ROS2_HUMBLE)


# 2. DEFINE AND REGISTER CUSTOM MESSAGES
PWM_STATE_DEF = """
uint16[] id
uint16[] position
int16[] offset
"""

SET_PWM_DEF = """
ros_robot_controller_msgs/PWMServoState[] state
float64 duration
"""

# Register the raw types into our store
typestore.register(get_types_from_msg(PWM_STATE_DEF, 'ros_robot_controller_msgs/msg/PWMServoState'))
typestore.register(get_types_from_msg(SET_PWM_DEF, 'ros_robot_controller_msgs/msg/SetPWMServoState'))

def get_steering_from_msg(msg):
    """
    Extract steering angle from the servo message (ID 3)

    msg: Deserialized SetPWMServoState message
    """
    # Loop through the list of servos in the message
    # msg.state is a list of PWMServoState objects
    try:
        # log.info("Extracting steering from servo message")
        for s in msg.state:
            # s.id is a list of integers (e.g. [3])
            # s.position is a list of integers
            for i, servo_id in enumerate(s.id):
                if servo_id == 3:
                    pos = s.position[i]

                    diff = float(pos) - SERVO_CENTER
                    raw_val = diff / (SERVO_MAX_LEFT - SERVO_CENTER)

                    # Invert if necessary depending on your specific training logic
                    # Usually: Left = Negative, Right = Positive
                    # If 2300 is Left, (2300-1500)/800 = 1.0. So we need to multiply by -1
                    return -1.0 * raw_val
                
    except Exception as err:
        log.error(f"Error extracting steering: {err}")
        # pass
    return None

def process_bag_folder():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    data_records = []
    total_images = 0

    # Store the last known steering to sync with images
    current_steering = 0.0

    log.info("Starting to process bag folders")

    for bag_folder in BAG_FOLDERS:
        log.info(f"Processing bag folder: {bag_folder}")

        # Check if folder exists
        if not os.path.exists(bag_folder):
            log.warning(f"Bag folder {bag_folder} does not exist. Skipping.")
            continue
        
        # Skip Bad Bags using TRY/EXCEPT
        try:
            with Reader(bag_folder) as reader:
                # log.info(f"Opened bag folder '{bag_folder}'")

                # Identify connections
                camera_conn = [x for x in reader.connections if '/ascamera' in x.topic]
                servo_conn = [x for x in reader.connections if 'pwm_servo/set_state' in x.topic]

                # Create a map for fast lookup if needed, but iteration is fine
                connections = camera_conn + servo_conn

                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    try:
                        # Deserialize using our custom store
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                        
                        # CASE A: STEERING UPDATE
                        if 'pwm_servo' in connection.topic:
                            new_steer = get_steering_from_msg(msg)
                            if new_steer is not None:
                                current_steering = new_steer
                        
                        # CASE B: CAMERA IMAGE
                        elif 'ascamera' in connection.topic:
                            # Reshape standard RGB image
                            # Convery ROS Image to OpenCV format
                            # Assuming rgb8 encoding
                            width = msg.width
                            height = msg.height

                            # Access raw byte data
                            data = np.frombuffer(msg.data, dtype=np.uint8)

                            # Reshape standard RGB image
                            img = data.reshape((height, width, 3))

                            #Convert RGB to BGR for OpenCV saving
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                            # Create filename
                            fname = f"{bag_folder}_{timestamp}.jpg"
                            save_path = os.path.join(OUTPUT_IMG_DIR, fname)

                            # Save image
                            cv2.imwrite(save_path, img)

                            # Save Record (Image path + Label)
                            data_records.append([os.path.join(OUTPUT_IMG_DIR, fname), current_steering])

                            total_images += 1
                            if total_images % 500 == 0:
                                log.info(f"Extracted {total_images} images so far.")
                    except Exception as err:
                        continue # skip bad frames
        
        except Exception as err:
            log.error(f"SKIPPING CORRUPT BAG {bag_folder}, error: {err}")

    # Save all records to CSV
    header = ['image_path', 'steering'] 
    df = pd.DataFrame(data_records, columns=header)
    df.to_csv(OUTPUT_CSV, index=False)

    log.info("-"*50)
    log.info(f"Processing complete. Saved Total images: {total_images}")
    log.info(f"Dataset saved to: {OUTPUT_CSV}")
    log.info("-"*50)

if __name__ == "__main__":
    process_bag_folder()