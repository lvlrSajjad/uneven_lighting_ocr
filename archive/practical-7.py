import cv2
import numpy as np

img = cv2.imread('../img.png')

scale_percent = 1000
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite('./p7/resized_img.png', resized_img)

alpha = 1.5
beta = -120

contrast_img = cv2.convertScaleAbs(resized_img, alpha=alpha, beta=beta)

cv2.imwrite('./p7/contrast_img.png', contrast_img)


sharpening_kernel = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]])

sharpened_img = cv2.filter2D(contrast_img, -1, sharpening_kernel)

cv2.imwrite('./p7/sharpened_img.png', sharpened_img)
#
# smoothed_img = cv2.GaussianBlur(sharpened_img, (1, 1), 0)  # Kernel size of 5x5
#
# cv2.imwrite('./p7/smoothed_img.png', smoothed_img)

processed_img = sharpened_img

gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('./p7/gray.png', gray)

binary_image = np.ones_like(gray) * 255

threshold_value = 102
white_threshold_value = 158

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        current_pixel = gray[i, j]
        new_threshold = threshold_value
        block_size = 25
        if j - block_size >= 0  and j + block_size < gray.shape[1] and i - block_size >= 0 and i + block_size < gray.shape[0]:
            if gray[i, j - block_size] > threshold_value and gray[i, j + block_size] > threshold_value and gray[i-block_size, j] > threshold_value and gray[i+block_size, j] > threshold_value:
                neighbor_pixels_avg = (int(gray[i, j - block_size]) + int(gray[i, j + block_size]) +
                                       int(gray[i - block_size, j]) + int(gray[i + block_size, j])) / 4
                if (neighbor_pixels_avg > white_threshold_value):
                    new_threshold = threshold_value + 42

        if current_pixel < new_threshold:
            binary_image[i, j] = 255
        else:
            binary_image[i, j] = 0

cv2.imwrite('./p7/binary_image.png', binary_image)

(h, w) = binary_image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -8, 1.0)
rotated = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

import pytesseract
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(rotated, config=custom_config)

print("Extracted Text:")
print(text)

with open('./p7/extracted_text_manual_threshold.txt', 'w') as f:
    f.write(text)
