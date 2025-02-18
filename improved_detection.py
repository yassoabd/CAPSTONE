from ultralytics import YOLO
import cv2
import numpy as np
import time

class VehicleDetector:
    def __init__(self):
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
            
        print("Setup complete. You should see the video window shortly.")
        print("Press 'q' to quit")
        
    def run_detection(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run detection with verbose=False to suppress terminal output
            results = self.model(frame, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name
                    class_name = result.names[cls]
                    
                    # Only show if confidence is high enough
                    if conf > 0.5:
                        # Calculate distance (rough estimation)
                        height = y2 - y1
                        distance = (100 * 50) / height  # rough formula
                        
                        # Set color based on distance
                        if distance < 3:
                            color = (0, 0, 255)  # Red
                            alert = "DANGER"
                        elif distance < 5:
                            color = (0, 255, 255)  # Yellow
                            alert = "Warning"
                        else:
                            color = (0, 255, 0)  # Green
                            alert = "Safe"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add labels
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add distance
                        distance_text = f"{alert}: {distance:.1f}m"
                        cv2.putText(frame, distance_text, (x1, y2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show the frame
            cv2.imshow('Object Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = VehicleDetector()
        detector.run_detection()
    except Exception as e:
        print(f"An error occurred: {str(e)}")