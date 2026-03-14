from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("test.jpg")

# Run detection
results = model(image)

# Draw detection boxes
annotated = results[0].plot()

# Show image
cv2.imshow("Detection", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()