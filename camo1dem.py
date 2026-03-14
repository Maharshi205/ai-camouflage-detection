import cv2
import numpy as np

# load image
image = cv2.imread("test.jpg")

# convert color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# split channels
l, a, b = cv2.split(lab)

# compute saliency map
saliency = cv2.Laplacian(l, cv2.CV_64F)
saliency = np.absolute(saliency)
saliency = np.uint8(saliency)

# normalize
saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)

# threshold suspicious regions
_, mask = cv2.threshold(saliency, 120, 255, cv2.THRESH_BINARY)

# highlight suspicious regions
overlay = image.copy()
overlay[mask == 255] = [0, 0, 255]

cv2.imshow("Original", image)
cv2.imshow("Suspicious Camouflage Regions", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()