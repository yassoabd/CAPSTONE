
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarning
import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to your video file
video_path = r'C:\Users\Jonathan\Projects\DrivingTest.mp4'  # Replace with the path to your video file

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the output (optional)
output_file = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Open a text file to save detection details (optional)
with open('detections.txt', 'w') as f:
    f.write("Frame, Class, Confidence, X1, Y1, X2, Y2\n")  # Write header

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if the video ends

        # Convert the frame to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(rgb_frame)

        # Parse the results
        detections = results.xyxy[0].numpy()  # Get detected objects
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:  # Confidence threshold
                label = model.names[int(cls)]  # Get the class label

                # Save detection details to the text file (optional)
                f.write(f"{frame_count}, {label}, {conf:.2f}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}\n")

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the output video file (optional)
        out.write(frame)

        # Display the frame
        cv2.imshow('Vehicle Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# Release the video capture, VideoWriter, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()