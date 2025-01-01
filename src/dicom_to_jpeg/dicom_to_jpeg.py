import os
import cv2
import numpy as np
import easyocr
import pydicom
import pandas as pd
from PIL import Image


# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def remove_text_using_easyocr(input_image_path, output_image_path):
    """
    Remove text from image using EasyOCR and save the result.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the processed image.
    """
    # Read the image
    img = cv2.imread(input_image_path)

    # Use EasyOCR to detect text
    results = reader.readtext(img)

    # Loop over the detected text regions and remove the text
    for (bbox, text, prob) in results:
        # Unpack the bounding box and draw a rectangle to remove the text
        top_left, top_right, bottom_right, bottom_left = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Black out the region where the text is detected
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)  # Fill with black color

    # Save the image with text removed
    cv2.imwrite(output_image_path, img)
    print(f"Processed and saved: {output_image_path}")


def remove_isolated_white_pixels_and_apply_clahe(input_image_path, output_image_path):
    """
    Remove isolated white pixels and apply CLAHE to the given JPEG image.
    
    Args:
        input_image_path (str): Path to the input JPEG image.
        output_image_path (str): Path to save the processed image.
    """
    try:
        # Read the image
        img = cv2.imread(input_image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove isolated white pixels
        thresh = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        
        result = img.copy()
        
        result[mask == 255] = (0,0,0)

        # Apply CLAHE
        img = cv2.medianBlur(result, 7)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save the processed image
        cv2.imwrite(output_image_path, img)
        print(f"Processed and saved: {output_image_path}")
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")


class DCM_PNJ():
    def __init__(self, img_path, img_data, output_folder):
        self.img_path = img_path
        self.img_data = img_data
        self.output_folder = output_folder

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index): 
        try:
            img_name = os.path.join(self.img_path, str(self.img_data.loc[index, 'patient_id']), str(self.img_data.loc[index, 'image_id']) + '.dcm')
            # Read the DICOM file
            dicom_file = pydicom.dcmread(img_name)

            # Extract the image data as a NumPy array
            ds = dicom_file.pixel_array

            # Normalize the image data to the range 0-255
            ds = (ds - np.min(ds)) / (np.max(ds) - np.min(ds))
            ds = (ds * 255.0).astype(np.uint8)

            # Handle photometric interpretation
            photometric_interpretation = dicom_file.PhotometricInterpretation if 'PhotometricInterpretation' in dicom_file else None
            if photometric_interpretation == "MONOCHROME1":
                ds = cv2.bitwise_not(ds)  # Invert using OpenCV

            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(ds)

            # Define class and output path
            class_name = "cancer" if self.img_data.loc[index, "cancer"] == 1 else "normal"
            jpeg_filename = str(self.img_data.loc[index, 'image_id']) + ".jpg"
            output_dir = os.path.join(self.output_folder, class_name)
            os.makedirs(output_dir, exist_ok=True)
            jpeg_path = os.path.join(output_dir, jpeg_filename)

            # Save the image as JPEG
            image.save(jpeg_path, 'JPEG')
            print(f"Converted {class_name} to {jpeg_filename}")

        except Exception as e:
            print(f"Error processing index {index}: {e}")


# Example Usage: Step 1 - Convert DICOM to JPEG
input_folder = '/home/whif/Data/RSNA/train_images'
input_csv = pd.read_csv("~/Data/balanced_dataset.csv")
output_folder = '/home/whif/Data/RSNA/train_jpeg_images'

#convert = DCM_PNJ(input_folder, input_csv, output_folder)
#for i in range(len(convert)):
#    convert.__getitem__(i)


jpeg_input_folder = '/home/whif/Data/RSNA/train_jpeg_images'
processed_output_folder = '/home/whif/Data/RSNA/processed_images'

for class_name in os.listdir(jpeg_input_folder):
    class_folder = os.path.join(jpeg_input_folder, class_name)
    output_class_folder = os.path.join(processed_output_folder, class_name)
    os.makedirs(output_class_folder, exist_ok=True)

    for jpeg_file in os.listdir(class_folder):
        input_image_path = os.path.join(class_folder, jpeg_file)
        output_image_path = os.path.join(output_class_folder, jpeg_file)

        # Step 1: Remove isolated white pixels and apply CLAHE
        remove_isolated_white_pixels_and_apply_clahe(input_image_path, output_image_path)

        # Step 2: Remove text from the image using EasyOCR
        remove_text_using_easyocr(output_image_path, output_image_path)