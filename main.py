import cv2
import numpy as np

# Load modle
net = cv2.dnn.readNet(r"D:\Facedetection\yolov\yolov3.weights", r"D:\Facedetection\yolov\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset includes names like human, car, etc.)
with open(r"D:\Facedetection\yolov\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the video file
video = cv2.VideoCapture(r"D:\Facedetection\WhatsApp Video 2024-10-13 at 11.17.45_e62bc56f.mp4")

# Get the frame rate of the video (frames per second)
fps = video.get(cv2.CAP_PROP_FPS)

# Loop through frames of the video
frame_skip = 2  # Process every 2nd frame to reduce load
frame_count = 0

while True:
    # Read the current frame
    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    # Get frame dimensions
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter only objects with high confidence
            if confidence > 0.5:
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

    # Apply non-maxima suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # You can customize colors for different objects

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detected objects
    cv2.imshow('Object Detection in Video', frame)

    # Wait to ensure video plays at the correct speed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
