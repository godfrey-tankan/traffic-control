import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize the frame
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    return blurred_frame

def main():
    source_type = input("Enter 'camera' for camera detection or 'video' for video upload: ")

    if source_type.lower() == 'camera':
        cap = cv2.VideoCapture(0)  # Use default camera (index 0)
    elif source_type.lower() == 'video':
        # video_path = input("Enter the path to the video file: ")
        # print("path: ", video_path)
        cap = cv2.VideoCapture("/home/tnqn/Documents/personal/detecting_cars/task1/cars.mp4")
    else:
        print("Invalid input. Exiting...")
        return

    if not cap.isOpened():
        print("Failed to open the video source. Exiting...")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    # Define the region of interest (ROI)
    roi_x, roi_y, roi_w, roi_h = 200, 200, 400, 200

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive a frame. Exiting...")
            break

        preprocessed_frame = preprocess_frame(frame)

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(preprocessed_frame)

        # Apply thresholding to obtain binary image
        _, binary_frame = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Apply ROI
        roi_frame = binary_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Find contours of moving objects
        contours, _ = cv2.findContours(roi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter contours based on area
            if cv2.contourArea(contour) > 1000:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                if area > 1000 and aspect_ratio > 1 and aspect_ratio < 2.5:
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x+roi_x, y+roi_y), (x+w+roi_x, y+h+roi_y), (0, 255, 0), 2)
                # Draw bounding box on the frame

        # Display the frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()