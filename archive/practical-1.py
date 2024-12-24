# Import necessary libraries
from PIL import Image
import pytesseract
import openpyxl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

out_image_p1 = "out_image_p1.png"  # Path to the scanned image
out_image_p2 = "out_image_p2.png"  # Path to the scanned image
out_image_p3 = "out_image_p3.png"  # Path to the scanned image

# Define the image-to-excel processing task
def process_scanned_image_to_excel(image_path, excel_path):
    try:


        # Open the scanned image
        with Image.open(image_path) as img00:

            img0 = cv2.imread(image_path)

            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img0)


            brightness = 0
            contrast = 1.0
            img = cv2.addWeighted(img0, contrast, np.zeros(img0.shape, img0.dtype), 0, brightness)
            img = imutils.rotate(img, -8)
            img = cv2.bitwise_not(img)

            text = pytesseract.image_to_string(img)

        return
        # Create a new Excel workbook and select the active sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Scanned Data"

        # Split the extracted text by lines
        lines = sorted(cnt_list, key=lambda x: x[1])

        # Write each line of the extracted text to a new row in the Excel sheet
        for row_index, line in enumerate(lines, start=1):
            sheet.cell(row=row_index, column=1, value=line)

        # Save the workbook to the specified path
        workbook.save(excel_path)
        print(f"Excel file saved successfully at {excel_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
image_path = "../img.png"  # Path to the scanned image


excel_path = "output_data.xlsx"  # Path where the Excel sheet will be saved

process_scanned_image_to_excel(image_path, excel_path)