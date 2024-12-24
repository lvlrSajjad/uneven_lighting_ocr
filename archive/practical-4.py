import cv2
import pytesseract
import numpy as np

# Load the image
img = cv2.imread('../img.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply adaptive thresholding for better text isolation
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations (increased kernel size)
kernel = np.ones((2, 2), np.uint8)  # Slightly larger kernel for clearer text separation
binary = cv2.erode(binary, kernel, iterations=1)
binary = cv2.dilate(binary, kernel, iterations=1)

# Correct rotation manually (8 degrees counter-clockwise)
(h, w) = binary.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -8, 1.0)  # Rotate by -8 degrees
rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Save the adjusted image (optional)
cv2.imwrite('./p4/corrected_receipt_v4.png', rotated)

# OCR using pytesseract with custom config
custom_config = r'--oem 3 --psm 6'  # Using default OEM and setting PSM to 6 for better segmentation
text = pytesseract.image_to_string(rotated, config=custom_config)

# Output the extracted text
print("Extracted Text:")
print(text)

# Save the text to a file
with open('./p4/extracted_text_v4.txt', 'w') as f:
    f.write(text)
