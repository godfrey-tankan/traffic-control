import darknet

def detect_objects(image_path):
    # Load YOLOv3 model
    model_path = "darknet/cfg/yolov3.cfg"
    weights_path = "darknet/yolov3.weights"
    meta_path = "darknet/cfg/coco.data"
    network, class_names, class_colors = darknet.load_network(model_path, meta_path, weights_path)

    # Load image
    image = darknet.load_image(image_path, 0, 0)

    # Perform object detection
    detections = darknet.detect_image(network, class_names, image)

    # Draw bounding boxes and labels on detected objects
    darknet.print_detections(detections, class_colors)

    # Display the image with detections
    darknet.show_image(image)