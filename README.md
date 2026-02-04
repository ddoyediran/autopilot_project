# Autopilot Project

## Overview
This project implements an AI-powered autopilot End-to-End ML system for autonomous vehicles, featuring data collection, model training, and real-time inference. It includes tools for processing driving data, training world models, and running both simulated and real-time control loops using deep learning models.

## Features
- **Data Processing:** Extracts images and steering data from ROS2 bag files for training.
- **World Model Inference:** Uses trained neural networks (VAE encoder/decoder, RNN, controller) for simulating and controlling vehicle behavior.
- **Dream Racer Simulation:** Visualizes the world model and allows manual/autopilot control in a simulated environment.
- **ROS2 Integration:** Real-time inference and control using ROS2 topics and custom messages.
- **Flight Recorder:** Logs inference and control data for analysis.

## Directory Structure
```
├── autopilot_new.py
├── autopilot_wm.py
├── autopilot.py (ignore)
├── dream_racer.py
├── process_data.py
├── training_images/
├── *.keras (model files)
├── *.tflite (TensorFlow Lite models)
├── README.md
```

## Main Components
- **process_data.py:** Extracts images and steering labels from ROS2 bag files and saves them for training.
- **dream_racer.py:** Simulates the world model, visualizes predictions, and allows manual/autopilot control.
- **autopilot_wm.py:** ROS2 node for real-time inference and control using TensorFlow Lite models.

## Setup
1. **Install Dependencies:**
	- Python 3.10+
	- TensorFlow, OpenCV, NumPy, pandas, ROS2, cv_bridge
	- Custom wheels: `ai_edge_litert`, `tflite_runtime` (provided in repo - ignore)
	- Other Python packages as needed (see `requirements.txt`)

2. **Prepare Data:**
	- Place ROS2 bag folders in the project directory.
	- Run `process_data.py` to extract images and steering data:
	  ```bash
	  python process_data.py
	  ```
	- Output: `training_images/` and `driving_log.csv`

3. **Train Models:**
	- Use your preferred training pipeline to train VAE encoder/decoder, RNN, and controller models.
	- Save models as `.keras` and convert to `.tflite` for deployment.
    - You can download all models from the releases here on Github. 

4. **Run Dream Racer Simulation:**
	```bash
	python dream_racer.py
	```
	- Controls:
	  - `A`: Toggle Autopilot
	  - `M`: Manual Mode
	  - `J/L`: Steer Left/Right
	  - `R`: Reset
	  - `Q`: Quit
    
    We have a Red/Green bar so you can see how hard the robot is steering. In Manual Mode, it shows a Blue Dot representing what the Autopilot would do if it were in charge.

5. **Deploy on ROS2 Robot:**
	- Launch `autopilot_wm.py` as a ROS2 node for real-time control.

## File Descriptions
- **autopilot_new.py / autopilot.py:** Alternative or legacy autopilot implementations.
- **autopilot_wm.py:** World model autopilot for ROS2 integration (To be run in the physical Robot).
- **dream_racer.py:** Simulated world model and control loop.
- **process_data.py:** Data extraction from ROS2 bags.
- **training_images/:** Extracted images for training.
- **training_data_version/:** Versioned model files.
- ***.keras / *.tflite:** Trained model files.
- **driving_log.csv:** Steering labels for training.

## Notes
- Ensure all model files are present before running inference scripts.
- For ROS2 integration, install all required ROS2 packages and custom message types.
- For TensorFlow Lite inference, use the provided wheels for ARM/aarch64 platforms.

## License
This project is provided for educational and research purposes. See LICENSE for details (to be added).

## Acknowledgements
- ROS2
- TensorFlow
- OpenCV
- Contributors to open-source driving datasets and world model research
