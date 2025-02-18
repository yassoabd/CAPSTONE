import cv2

def test_camera():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Camera initialized successfully! You should see a video window.")
    print("Press 'q' to quit the program.")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Display the frame
        cv2.imshow('Webcam Test', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()