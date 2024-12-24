# Import necessary libraries
from PIL import Image
import pytesseract
import openpyxl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def save_to_excel(filename, image, excel_path):
    # Extract text using Tesseract
    text = pytesseract.image_to_string(image)

    # Create a new Excel workbook and select the active sheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Scanned Data"

    # Split the extracted text by lines
    lines = text.splitlines()

    # Write each line of the extracted text to a new row in the Excel sheet
    for row_index, line in enumerate(lines, start=1):
        sheet.cell(row=row_index, column=1, value=line)

    # Save the workbook to the specified path
    workbook.save(f"{filename}{excel_path}")
    print(f"Excel file saved successfully at {excel_path}")

# Define the image-to-excel processing task
def process_scanned_image_to_excel(image_path, excel_path):
    try:
        # Read the image using OpenCV
        img0 = imutils.rotate(cv2.imread(image_path), -8)

        # Convert to grayscale
        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        # plt.imshow(gray)

        # Increase contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        contrast_enhanced = clahe.apply(gray)

        plt.imshow(contrast_enhanced)

        # Sharpen the image
        kernel = np.array([[0, -1, 0],
                           [-1, 5.1, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)

        # plt.imshow(sharpened)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Save processed images for visualization
        plt.imsave("out_image_enhanced.png", thresh, cmap='gray')

        # Display the processed image
        plt.subplot(1, 2, 1)
        plt.title("Processed Image")
        # plt.imshow(thresh, cmap='gray')
        plt.show()

        # Save to Excel
        save_to_excel("processed_file", thresh, excel_path)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
image_path = "../img.png"  # Path to the scanned image
excel_path = "output_data.xlsx"  # Path where the Excel sheet will be saved

process_scanned_image_to_excel(image_path, excel_path)
