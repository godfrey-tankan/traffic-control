import cv2
import numpy as np

# Load YOLO


net = cv2.dnn.readNet("./darknet/yolov3.weights", "./darknet/cfg/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(image):
    # Load image
    img = cv2.imread(image)
    height, width, channels = img.shape

    # Preprocess image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input to the network
    net.setInput(blob)
    # Forward pass through the network
    outs = net.forward(output_layers)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # BGR format
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with detections
    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def perform_object_detection(choice):
    if choice == "image":
        image_path = input("Enter the path to the image file: ")
        detect_objects(image_path)
    elif choice == "video":
        video_path = input("Enter the path to the video file: ")
        detect_objects_from_video(video_path)
    elif choice == "camera":
        detect_objects_from_camera()
    else:
        print("Invalid choice. Please choose 'image', 'video', or 'camera'.")

def detect_objects_from_video(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        detect_objects(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detect_objects_from_camera():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()

        detect_objects(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Prompt the user to choose between image, video, or camera
    choice = input("Choose 'image', 'video', or 'camera' for object detection: ")
    perform_object_detection(choice)