import os
import shutil

import pytesseract
import cv2
import numpy as np

# -----------------------
# 1) GLOBAL ADJUSTABLE VARIABLES
# -----------------------
IMAGE_PATH = './img.png'
OUTPUT_DIR = './p5'
BLOCKS_SUBDIR = 'blocks'
STITCHED_FILENAME = 'stitched_image_thresholded_custom.png'

# -----------------------
# Preprocessing Toggles and Parameters
# -----------------------
# Contrast adjustment
PRE_CONTRAST_DENOISE_ENABLED = False
CONTRAST_ADJUSTMENT_ENABLED = True
ALPHA = 1.6
BETA = -125

# Upscaling
UPSCALING_ENABLED = True
POST_UPSCALE_DENOISE_ENABLED = True
SCALE_PERCENT = 400  # e.g., 1000 = 10x
INTERPOLATION = cv2.INTER_CUBIC

# -----------------------
# Super Resolution Options
# -----------------------
SUPER_RESOLUTION_ENABLED = False
SUPER_RES_MODEL_PATH = './models/EDSR_x4.pb'  # Update path to where you saved the model
SUPER_RES_MODEL_NAME = 'edsr'  # 'edsr', 'espcn', 'fsrcnn', 'lapsrn'
SUPER_RES_SCALE = 4            # 2, 3, 4, or 8 (depends on the model)


# Sharpening
SHARPENING_ENABLED = True
SHARPENING_KERNEL = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# Rotation
ROTATION_ENABLED = True
ROTATION_ANGLE = -7.8  # degrees
FLAGS = cv2.INTER_CUBIC
BORDER_MODE = cv2.BORDER_REPLICATE

# Cropping
CROPPING_ENABLED = True
H_CROP_MARGIN_SCALE = 0.09
W_CROP_MARGIN_SCALE = 0.17

# -----------------------
# Block Generation
# -----------------------"
# Choose one of: "linear", "variable", "cubic"
BLOCK_GENERATION_METHOD = "variable"

# For "linear" block generation
BLOCK_SIZE = 100

# For "variable" block generation
VARIABLE_BLOCK_BASE_SIZE = 100
VARIABLE_BLOCK_DISTANCE_FACTOR = 90
VARIABLE_BLOCK_MIN_SIZE = 16

# -----------------------
# Thresholding Parameters for All Methods
# -----------------------
# block_threshold = max(BLOCK_THRESHOLD_BASE, block_mean - BLOCK_THRESHOLD_OFFSET)
BLOCK_THRESHOLD_BASE = 75
BLOCK_THRESHOLD_OFFSET = 60

# -----------------------
# Tesseract Parameters
# -----------------------
# You can create multiple sets or just one. Shown here as two sets for demonstration.

# --- Set 1 For Comparison to set 2
ENABLE_SET_1 = False
TESS_OEM_1 = 1  # other options: 0, 1, 2, 3
TESS_PSM_1 = 4  # other options: 3, 4, 6, 11, etc.
TESS_LANG_1 = 'eng'
TESS_BLACKLIST_1 = r'"\°\{\}\"»«\$]¢abcdefghijklmnopqrstuvwxyz§|\!@"\%\\/\\*\',©“['
TESS_WHITELIST_1 = "*"

# --- Set 2 For Comparison to set 1
ENABLE_SET_2 = True
TESS_OEM_2 = 1
TESS_PSM_2 = 6
TESS_LANG_2 = 'eng'
TESS_BLACKLIST_2 = r'"\°\{\}\"»«\$]¢abcdefghijklmnopqrstuvwxyz§|\!@"\%\\/\\*\',©“['
TESS_WHITELIST_2 = "*"


def super_res_upscale(img, output_dir, step_number, model_path, model_name, scale):
    """
    Upscale an image using OpenCV's DNN Super Resolution module.
    """

    if not SUPER_RESOLUTION_ENABLED:
        return img

    # Create the super resolution instance
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # Read the pre-trained model
    sr.readModel(model_path)

    # Set the model name and scale
    # model_name: 'edsr', 'espcn', 'fsrcnn', or 'lapsrn'
    # scale: 2,3,4,8 (depending on the chosen model)
    sr.setModel(model_name, scale)

    # Upsample the image
    upscaled_image = sr.upsample(img)

    # Save the upsampled image for debugging
    output_path = os.path.join(output_dir, f'{step_number}-superres_upscaled.png')
    cv2.imwrite(output_path, upscaled_image)

    return upscaled_image


def build_tesseract_config(oem, psm, lang, blacklist, whitelist):
    """
    Dynamically build the Tesseract configuration string.
    """
    return (
        f'--oem {oem} '
        f'--psm {psm} '
        f'-l {lang} '
        f'-c tessedit_char_blacklist={blacklist} '
        f'tessedit_char_whitelist={whitelist}'
    )


# -------------------------------------------------------
#               GENERAL UTILITY FUNCTIONS
# -------------------------------------------------------
def clean_dir(output_dir):
    """
    Clean up the output directory by removing all its contents if it exists.
    If the directory does not exist, it creates it.
    """
    if os.path.exists(output_dir):
        # Remove all contents of the directory
        shutil.rmtree(output_dir)
    # Recreate the empty directory
    os.makedirs(output_dir)


# -------------------------------------------------------
#               PREPROCESSING STEPS
# -------------------------------------------------------
def convert_to_grayscale(image_path, output_dir, step_number):
    img = image_path
    gray_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # If they are nearly identical, use one or the other
    if np.abs(gray_bgr.mean() - gray_rgb.mean()) < 1e-3:
        cv2.imwrite(os.path.join(output_dir, f'{step_number}-gray.png'), gray_bgr)
        return gray_bgr
    else:
        cv2.imwrite(os.path.join(output_dir, f'{step_number}-gray.png'), gray_rgb)
        return gray_rgb


def adjust_contrast(gray, output_dir, step_number):
    if not CONTRAST_ADJUSTMENT_ENABLED:
        return gray

    pre_contrast = gray
    if PRE_CONTRAST_DENOISE_ENABLED:
        pre_contrast = cv2.fastNlMeansDenoising(pre_contrast, h=10, templateWindowSize=7, searchWindowSize=21)

    contrast_img = cv2.convertScaleAbs(pre_contrast, alpha=ALPHA, beta=BETA)
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-contrast_img.png'), contrast_img)


    return contrast_img


def upscale(img, output_dir, step_number):
    if not UPSCALING_ENABLED:
        return img

    width = int(img.shape[1] * SCALE_PERCENT / 100)
    height = int(img.shape[0] * SCALE_PERCENT / 100)

    # Try INTER_CUBIC
    resized_img = cv2.resize(img, (width, height), interpolation=INTERPOLATION)
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-resized_img.png'), resized_img)

    if not POST_UPSCALE_DENOISE_ENABLED:
        return resized_img

    # Optional denoising step
    resized_img = cv2.fastNlMeansDenoising(resized_img, h=10, templateWindowSize=7, searchWindowSize=21)
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-resized_denoised.png'), resized_img)

    return resized_img


def adjust_sharpening(resized_img, output_dir, step_number):
    if not SHARPENING_ENABLED:
        return resized_img

    sharpened_img = cv2.filter2D(resized_img, -1, SHARPENING_KERNEL)
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-sharpened_img.png'), sharpened_img)

    # sharpened_img = cv2.fastNlMeansDenoising(sharpened_img, h=10, templateWindowSize=7, searchWindowSize=21)
    # cv2.imwrite(os.path.join(output_dir, f'{step_number}-sharpened_img_denoised.png'), sharpened_img)

    return sharpened_img


def adjust_rotation(sharpened_img, output_dir, step_number):
    if not ROTATION_ENABLED:
        return sharpened_img

    h, w = sharpened_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, ROTATION_ANGLE, 1.0)
    rotated_img = cv2.warpAffine(sharpened_img, M, (w, h),
                                 flags=FLAGS,
                                 borderMode=BORDER_MODE)
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-rotated.png'), rotated_img)
    return rotated_img


def crop(rotated_img, output_dir, step_number):
    if not CROPPING_ENABLED:
        return rotated_img

    h, w = rotated_img.shape[:2]
    H_CROP_MARGIN = int(h * H_CROP_MARGIN_SCALE)
    W_CROP_MARGIN = int(w * W_CROP_MARGIN_SCALE)
    if h > 2 * H_CROP_MARGIN and w > 2 * W_CROP_MARGIN:
        cropped_img = rotated_img[H_CROP_MARGIN:h - H_CROP_MARGIN,
                      W_CROP_MARGIN:w - W_CROP_MARGIN]
    else:
        raise ValueError("Image is too small to crop.")
    cv2.imwrite(os.path.join(output_dir, f'{step_number}-cropped.png'), cropped_img)

    # cropped_img = cv2.fastNlMeansDenoising(cropped_img, h=10, templateWindowSize=7, searchWindowSize=21)
    # cv2.imwrite(os.path.join(output_dir, f'{step_number}-cropped_denoised.png'), cropped_img)

    return cropped_img


def preprocess_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    functions_queue = [
        upscale,
        super_res_upscale,
        convert_to_grayscale,
        adjust_contrast,
        adjust_sharpening,
        adjust_rotation,
        crop
    ]

    result_image = cv2.imread(image_path)

    for i, func in enumerate(functions_queue):
        if func.__name__ == 'super_res_upscale':
            result_image = super_res_upscale(
                img=result_image,
                output_dir=output_dir,
                step_number=i,
                model_path=SUPER_RES_MODEL_PATH,
                model_name=SUPER_RES_MODEL_NAME,
                scale=SUPER_RES_SCALE
            )
        else:
            result_image = func(result_image, output_dir, i)

    return result_image


# -------------------------------------------------------
#      MULTIPLE BLOCK-GENERATION FUNCTIONS
# -------------------------------------------------------
def generate_blocks_linear(image, block_size, output_dir):
    """Generate thresholded blocks of the image in a linear grid."""
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous blocks
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Step through the image in grid fashion
    for y in range(0, image.shape[0], block_size):
        for x in range(0, image.shape[1], block_size):
            y_end = min(y + block_size, image.shape[0])
            x_end = min(x + block_size, image.shape[1])
            block = image[y:y_end, x:x_end]

            if block.size > 0:
                block_mean = np.mean(block)
                block_threshold = max(BLOCK_THRESHOLD_BASE, block_mean - BLOCK_THRESHOLD_OFFSET)
                binary_block = np.where(block < block_threshold, 255, 0).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f'block_{y}_{x}.png'), binary_block)


def generate_blocks_variable(image, output_dir):
    """Generate blocks with varying sizes based on proximity to the center."""
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous blocks
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Calculate the image center
    h, w = image.shape[:2]
    center = (h // 2, w // 2)

    # Iterate over the image
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Calculate the distance from the center
            dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

            # Define block size as inversely proportional to distance from the center
            # e.g.,  block_size = max(MIN, base_size - (some_factor * ratio_of_distance))
            # Below is just an example formula:
            block_size = max(VARIABLE_BLOCK_MIN_SIZE,
                             int(VARIABLE_BLOCK_BASE_SIZE - (dist / max(h, w)) * VARIABLE_BLOCK_DISTANCE_FACTOR))

            # Compute block boundaries
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = image[y:y_end, x:x_end]

            if block.size > 0:
                block_mean = np.mean(block)
                block_threshold = max(BLOCK_THRESHOLD_BASE, block_mean - BLOCK_THRESHOLD_OFFSET)
                binary_block = np.where(block < block_threshold, 255, 0).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f'block_{y}_{x}.png'), binary_block)

            x += block_size
        y += block_size


def generate_blocks_cubic(image, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous blocks
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Calculate the image center
    h, w = image.shape[:2]
    center = (h // 2, w // 2)

    y = 0
    while y < h:
        x = 0
        while x < w:
            # Distance from the current point to the center
            dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)

            # Example "cubic" formula for block size
            block_size = max(
                VARIABLE_BLOCK_MIN_SIZE,
                int(
                    VARIABLE_BLOCK_BASE_SIZE
                    - ( (dist / max(h, w))**3 ) * VARIABLE_BLOCK_DISTANCE_FACTOR
                )
            )

            # Compute block boundaries
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = image[y:y_end, x:x_end]

            if block.size > 0:
                # Calculate threshold
                block_mean = np.mean(block)
                block_threshold = max(
                    BLOCK_THRESHOLD_BASE,
                    block_mean - BLOCK_THRESHOLD_OFFSET
                )
                binary_block = np.where(block < block_threshold, 255, 0).astype(np.uint8)

                # Save each block
                cv2.imwrite(os.path.join(output_dir, f'block_{y}_{x}.png'), binary_block)

            x += block_size
        y += block_size

# -------------------------------------------------------
#             STITCHING & OCR
# -------------------------------------------------------
def stitch_blocks(image_shape, blocks_dir, output_path):
    """Stitch thresholded blocks back into a single image."""
    stitched_image = np.zeros(image_shape, dtype=np.uint8)

    for filename in os.listdir(blocks_dir):
        if filename.endswith('.png'):
            # Filenames have the format: block_y_x.png
            y, x = map(int, filename.replace('block_', '').replace('.png', '').split('_'))
            block = cv2.imread(os.path.join(blocks_dir, filename), cv2.IMREAD_GRAYSCALE)
            stitched_image[y:y + block.shape[0], x:x + block.shape[1]] = block

    cv2.imwrite(output_path, stitched_image)
    return stitched_image


def extract_text(image, config):
    """Extract text from the image using Tesseract."""
    return pytesseract.image_to_string(image, config=config)


def main():
    # 1) Clean and prepare directories
    clean_dir(OUTPUT_DIR)
    blocks_dir = os.path.join(OUTPUT_DIR, BLOCKS_SUBDIR)
    os.mkdir(blocks_dir)
    stitched_output = os.path.join(OUTPUT_DIR, STITCHED_FILENAME)

    # 2) Preprocess the image
    cropped_img = preprocess_image(IMAGE_PATH, OUTPUT_DIR)

    # 3) Generate blocks based on selected method
    if BLOCK_GENERATION_METHOD == "linear":
        generate_blocks_linear(cropped_img, block_size=BLOCK_SIZE, output_dir=blocks_dir)
    elif BLOCK_GENERATION_METHOD == "variable":
        generate_blocks_variable(cropped_img, output_dir=blocks_dir)
    elif BLOCK_GENERATION_METHOD =="cubic":
        generate_blocks_cubic(cropped_img, output_dir=blocks_dir)
    else:
        raise ValueError(f"Unknown block generation method: {BLOCK_GENERATION_METHOD}")

    # 4) Stitch blocks
    stitched_img = stitch_blocks(cropped_img.shape, blocks_dir, stitched_output)

    # 5) Build Tesseract config strings (dynamic from the top)
    custom_config1 = build_tesseract_config(
        oem=TESS_OEM_1,
        psm=TESS_PSM_1,
        lang=TESS_LANG_1,
        blacklist=TESS_BLACKLIST_1,
        whitelist=TESS_WHITELIST_1
    )

    custom_config2 = build_tesseract_config(
        oem=TESS_OEM_2,
        psm=TESS_PSM_2,
        lang=TESS_LANG_2,
        blacklist=TESS_BLACKLIST_2,
        whitelist=TESS_WHITELIST_2
    )

    # 6) Extract text with two different configs
    if ENABLE_SET_1:
        text_set_1 = extract_text(stitched_img, config=custom_config1)
        print("Extracted Text Set 1:")
        print('Config:', custom_config1)
        print(text_set_1)

    if ENABLE_SET_2:
        test_set_2 = extract_text(stitched_img, config=custom_config2)
        print("Extracted Text Set 2:")
        print('Config:', custom_config2)
        print(test_set_2)

if __name__ == "__main__":
    main()
