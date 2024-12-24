import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import statistics
import imutils

plt.rcParams['figure.figsize'] = [20, 10]

# Load image
img = imutils.rotate(cv2.imread('../img.png', 0), -8)

# Display the image
imgplot = plt.imshow(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cmap='gray')

# Add border and threshold the image
img1 = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255])
(thresh, th3) = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

plt.imsave('practical-3-1.png', img1)

# Invert the image
th3 = 255 - th3

# Set kernels based on image size
if th3.shape[0] < 1000:
    ver = np.ones((7, 1), np.uint8)
    hor = np.ones((1, 6), np.uint8)
else:
    ver = np.ones((19, 1), np.uint8)
    hor = np.ones((1, 15), np.uint8)

# Detect vertical lines of table borders
img_temp1 = cv2.erode(th3, ver, iterations=4)
verticle_lines_img = cv2.dilate(img_temp1, ver, iterations=4)

plt.imsave('practical-3-2.png', img_temp1)

# Detect horizontal lines of table borders
img_hor = cv2.erode(th3, hor, iterations=4)
hor_lines_img = cv2.dilate(img_hor, hor, iterations=5)

# Add horizontal and vertical lines
hor_ver = cv2.add(hor_lines_img, verticle_lines_img)

# Invert the image
hor_ver = 255 - hor_ver

# Construct table borders for the image
temp = cv2.subtract(th3, hor_ver)

# Bitwise XOR to refine table borders
tt = cv2.bitwise_xor(img1, temp)
iii = cv2.bitwise_not(tt)

# Kernel for morphological operations
ver1 = np.ones((9, 2), np.uint8)
hor1 = np.ones((2, 10), np.uint8)

# Morphological operations to detect vertical and horizontal lines
temp1 = cv2.erode(iii, ver1, iterations=3)
verticle_lines_img1 = cv2.dilate(temp1, ver1, iterations=3)

temp12 = cv2.erode(iii, hor1, iterations=1)
hor_lines_img2 = cv2.dilate(temp12, hor1, iterations=3)

# Detecting text area
hor_ver = cv2.add(hor_lines_img2, verticle_lines_img1)

# Resizing for better visualization
dim = (hor_ver.shape[1] * 2, hor_ver.shape[0] * 2)
resized = cv2.resize(hor_ver, dim, interpolation=cv2.INTER_AREA)

# Invert for text detection
want = cv2.bitwise_not(resized)

# Kernel for further dilation
kernel1 = np.ones((1, 3), np.uint8)

# Dilate image
tt1 = cv2.dilate(want, kernel1, iterations=5)

# Resize image back
resized1 = cv2.resize(tt1, (hor_ver.shape[1], hor_ver.shape[0]), interpolation=cv2.INTER_AREA)

# Find contours for bounding boxes
contours1, _ = cv2.findContours(resized1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Sort contours
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


cnts, boundingBoxes = sort_contours(contours1, method="top-to-bottom")

# Process bounding boxes and extract text
heightlist = [boundingBox[3] for boundingBox in boundingBoxes]
heightlist.sort()

sportion = int(0.35 * len(heightlist))
eportion = int(0.1 * len(heightlist))

print(f"sportion: {sportion}, eportion: {eportion}")
print(f"Sliced heightlist: {heightlist[-sportion:-eportion]}")

if heightlist[-sportion:-eportion]:
    medianheight = statistics.mean(heightlist[-sportion:-eportion])
else:
    print("Warning: No data points found for the given portion!")
    medianheight = 0  # Or handle it as you see fit

box = []
imag = iii.copy()

for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    if h >= 0.7 * medianheight and w / h > 0.9:
        image = cv2.rectangle(imag, (x + 4, y - 2), (x + w - 5, y + h), (0, 255, 0), 1)
        box.append([x, y, w, h])

# Display final bounding boxes
imgplot = plt.imshow(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cmap='gray')

# Save the result
cv2.imwrite('outputtest.png', image)

# Extract and save text
todump = []
for i in range(len(box)):
    x, y, w, h = box[i]
    roi = iii[y:y + h, x:x + w]
    img = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    out = pytesseract.image_to_string(img)
    if len(out) == 0:
        out = pytesseract.image_to_string(img, config='--psm 10')

    todump.append(out.strip())

# Store the extracted text into an Excel file
df = pd.DataFrame(todump)
df.to_excel("imageout.xlsx", index=False)

print("Text extracted and saved to imageout.xlsx")
