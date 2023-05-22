import cv2
import numpy as np
import pytesseract
import streamlit as st
from io import BytesIO
from PIL import Image
import rembg

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def remove_background(image):
    output = rembg.remove(image)
    
    # Get the dimensions of the output image
    output_height, output_width, _ = output.shape
    
    # Crop the processed image to the output image size
    cropped_image = output[:output_height, :output_width]
    
    return cropped_image

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

def save_image(image, file_path):
    cv2.imwrite(file_path, image)

def main():
    st.title("Hand-drawn Image Text Extraction")

    # Upload multiple images
    uploaded_images = st.file_uploader("Upload Hand-drawn Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_images is not None:
        images = [cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED) for image in uploaded_images]

        # Extract text from each image
        for i, image in enumerate(images):
            extracted_text = extract_text(image)

            # Display the processed image and extracted text
            st.image(image, caption='Original Image', use_column_width=True)
            processed_image = remove_background(image)
            st.image(processed_image, caption='Processed Image', use_column_width=True)
            st.write("Extracted Text:")
            st.write(extracted_text)
            st.write("---")

            # Save the extracted text to a txt file
            text_file_path = f"extracted_text_{i}.txt"
            save_text_to_file(extracted_text, text_file_path)
            st.write(f"Extracted text saved as: {text_file_path}")

            # Save the processed image
            image_file_path = f"processed_image_{i}.png"
            save_image(processed_image, image_file_path)
            st.write(f"Processed image saved as: {image_file_path}")

if __name__ == "__main__":
    main()

