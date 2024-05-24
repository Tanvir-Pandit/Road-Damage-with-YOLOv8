# Crack and Pothole Detection using YOLO and Object Tracking

This project uses a YOLO (You Only Look Once) model to detect cracks and potholes in video footage and tracks these detections using an object tracking algorithm.

## Requirements

Install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

## Files

- **main.py**: The main script for detection and tracking.
- **model/best.pt**: The trained YOLO model file.
- **objects/object_list.txt**: A file containing class names for the model.
- **tracker/tracker.py**: The object tracking script.
- **videos/crack 2.mp4**: Example video file to be processed.

## Usage

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the script**:
    ```bash
    python main.py
    ```

3. **View the results**:
    - The video will display with bounding boxes and warnings for detected cracks or potholes.
    - Press `q` to exit the video display window.

