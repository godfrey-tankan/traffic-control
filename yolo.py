import cv2
import numpy as np

def main():
    source_type = input("Enter 'camera' for camera detection or 'video' for video upload: ")

    if source_type.lower() == 'camera':
        cap = cv2.VideoCapture(0)  # Use default camera (index 0)
    elif source_type.lower() == 'video':
        cap = cv2.VideoCapture("/home/tnqn/Documents/personal/detecting_cars/task1/cars.mp4")
    else:
        print("Invalid input. Exiting...")
        return

    if not cap.isOpened():
        print("Failed to open the video source. Exiting...")
        return

    # Load pre-trained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet("darknet/cfg/yolov3.cfg", "darknet/yolov3.weights")

    # Define the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive a frame. Exiting...")
            break

        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the detected objects
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 2:  # Class ID for cars is 2
                    # Scale the bounding box coordinates
                    width = frame.shape[1]
                    height = frame.shape[0]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Store the detected car information
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Apply non-maximum suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes on the frame
        for i in indices:
            i = i[0]
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()