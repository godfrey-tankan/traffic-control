import cv2
import numpy as np

def main():
    source_type = input("Enter 'camera' for camera detection or 'video' for video upload: ")

    if source_type.lower() == 'camera':
        cap = cv2.VideoCapture(0) 
    elif source_type.lower() == 'video':
        video_path = input("Enter the path to the video file: ")
        print("path: ", video_path)
        path ="/home/tnqn/Documents/personal/detecting_cars/task1/test2.mp4"
        cap = cv2.VideoCapture(path)
    else:
        print("Invalid input. Exiting...")
        return

    if not cap.isOpened():
        print("Failed to open the video source. Exiting...")
        return


    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=40)
    while True:
        ret, frame = cap.read()
    
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            if cv2.contourArea(cnt) >= 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # else:
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)


        cv2.imshow('Frame', frame)

        key = cv2.waitKey(30)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    