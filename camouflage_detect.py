import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Read image
image = cv2.imread("test.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection (Canny)
edges = cv2.Canny(gray, 50, 150)

# Texture enhancement using Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Combine edges + texture
combined = cv2.addWeighted(edges, 0.7, laplacian, 0.3, 0)

# Run YOLO detection
results = model(image)

# Draw object detections
annotated = results[0].plot()

# Overlay camouflage edges
colored_edges = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(annotated, 0.7, colored_edges, 0.3, 0)

# Show results
cv2.imshow("Original Detection", annotated)
cv2.imshow("Camouflage Pattern Detection", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()