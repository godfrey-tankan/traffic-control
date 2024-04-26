import cv2
from tracker import *

cap = cv2.VideoCapture('task1/test2.mp4')

car_tracker_ob = EuclideanDistTracker()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=100)
while True:
    ret, frame = cap.read()
    roi = frame[200:-100, 300:-690]
   
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_cars = []
    for cnt in contours:
        # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        if cv2.contourArea(cnt) > 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            detected_object = [x, y, w, h]
            if detected_object not in detected_cars:
                detected_cars.append([x, y, w, h])
    cars_tracked = car_tracker_ob.update(detected_cars)

    for car in cars_tracked:
        x, y, w, h, id = car
        cv2.putText(frame,str(id), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)


    cv2.imshow('Frame', roi)
    # cv2.imshow('Frame', frame)
    # cv2.imshow('Mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

    