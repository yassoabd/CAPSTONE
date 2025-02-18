import cv2
import numpy as np

def detect_vehicles():
    
    cap = cv2.VideoCapture(0)
    
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    print("Detection started. Press 'q' to quit.")
    
    while True:
       
        ret, frame = cap.read()
        if not ret:
            break
            
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects
        objects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
 
        for (x, y, w, h) in objects:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(frame, 'Object Detected', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate approximate distance 
            distance = (100 * 50) / w 
            cv2.putText(frame, f'~{distance:.1f}m', (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Vehicle Detection', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_vehicles()