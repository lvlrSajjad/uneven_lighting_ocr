import cv2
import numpy as np

# Load image and resize as per your existing steps
img = cv2.imread('../img.png')

scale_percent = 1000
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('p6/resized_img.png', resized_img)

alpha = 1.5
beta = -120
contrast_img = cv2.convertScaleAbs(resized_img, alpha=alpha, beta=beta)
cv2.imwrite('p6/contrast_img.png', contrast_img)

# Convert to grayscale
gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('p6/gray.png', gray)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Perform morphological closing to reduce noise
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

# Optionally apply another round of noise reduction with median blur
closed = cv2.medianBlur(closed, 5)

cv2.imwrite('p6/closed.png', closed)

# Rotate image if needed (same as your existing code)
(h, w) = closed.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -8, 1.0)
rotated = cv2.warpAffine(closed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite('p6/rotated.png', rotated)

# Extract text using Tesseract
import pytesseract
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(rotated, config=custom_config)

print("Extracted Text:")
print(text)

with open('p6/extracted_text_adaptive_threshold.txt', 'w') as f:
    f.write(text)
