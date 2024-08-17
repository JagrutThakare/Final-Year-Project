import os
import re
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from auth import save_number_plate_data
from gradio_client import Client, handle_file
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def query(filename, retries=20, delay=5):
    # Connect to your Hugging Face Space
    
    for _ in range(retries):
        try:
            # Send the local file instead of URL
            client = Client("jagrutthakare/TrOCR")
            result = client.predict(
                image=handle_file(filename),  # pass local image file
                api_name="/predict"
            )
            return result
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            time.sleep(delay)
    return None

def run_number_plate_detection():
    st.sidebar.title("🛵 Number Plate Detection Settings")
    detection_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)

    st.sidebar.title("About Number Plate Detection")
    st.sidebar.write(
        """
        Number plate detection uses AI and computer vision to identify and extract vehicle license plate information 
        from images or videos. It is widely used for:
        - Traffic enforcement.
        - Parking management.
        - Toll collection automation.
        """
    )

    st.sidebar.title("How It Works")
    st.sidebar.write(
        """
        1. **Camera Input**: Captures real-time footage of vehicles.
        2. **Image Processing**: Detects and extracts the number plate area from the image.
        3. **Optical Character Recognition (OCR)**: Reads and translates the characters on the plate into digital text.
        4. **Action**: Stores, validates, or uses the extracted data for enforcement, tolls, or parking.
        """
    )

    st.sidebar.title("Technologies Used")
    st.sidebar.write(
        """
        - **Computer Vision**: Identifies and processes the number plate region.
        - **OCR Tools**: Extracts alphanumeric data from plates.
        - **AI Models**: Handles varying plate designs, fonts, and conditions.
        """
    )

    st.sidebar.title("Key Benefits")
    st.sidebar.write(
        """
        - Enhances traffic rule enforcement.
        - Simplifies parking and toll management.
        - Provides accurate and automated solutions.
        """
    )

    st.sidebar.title("Challenges")
    st.sidebar.write(
        """
        - Variations in plate designs and fonts.
        - Low-quality images due to lighting or motion.
        - Ensuring system scalability for large data.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit 🚗📷")   

    st.title("🛵 License Plate Detection App")
    st.write("Upload an image to detect license plates.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        st.write("Detecting license plates... (Please have patience, this may take a while depending on whether the huggingface server is awake or not)")
        detections = detect_license_plate(image)
        if isinstance(detections, dict) and 'predictions' in detections and detections['predictions']:
            image_with_boxes = draw_bounding_boxes(image.copy(), detections)
            plate_data = extract_text_with_trocr(image, detections, detection_threshold)
            st.image(image_with_boxes, caption="Detected License Plates")
            if plate_data:
                # Cleaning string
                for data in plate_data:
                    data["Plate Number"] = re.sub(r"^-|-$", "", data["Plate Number"])
                    save_number_plate_data(data["Plate Number"], data["Detection Confidence"], image)
                    
                st.write("**Detected License Plate Data:**")
                table_data = {
                    "Plate Number": [data["Plate Number"] for data in plate_data],
                    "Detection Confidence (%)": [data["Detection Confidence"] for data in plate_data],
                    "Detection Threshold (%)": [detection_threshold * 100] * len(plate_data),
                }
                st.table(table_data)
            else:
                st.write("No text detected in license plates.")
        else:
            st.write("No license plates detected.")

def detect_license_plate(image):
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    result = CLIENT.infer(image, model_id="number-plate-detection-rgmog/1")
    return result

def draw_bounding_boxes(image, detections):
    draw = ImageDraw.Draw(image)

    img_width, img_height = image.size

    box_thickness = max(2, int(img_width * 0.005))  
    font_size = max(10, int(img_width * 0.03))  

    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    for detection in detections['predictions']:
        x1, y1 = detection['x'] - detection['width'] / 2, detection['y'] - detection['height'] / 2
        x2, y2 = detection['x'] + detection['width'] / 2, detection['y'] + detection['height'] / 2

        # Draw bounding box with dynamic thickness
        draw.rectangle([x1, y1, x2, y2], outline="green", width=box_thickness)

        confidence = detection['confidence'] * 100
        label = f"{confidence:.2f}%"

        # Get text size dynamically
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_position = (x1, y1 - text_height - 5)

        # Draw label background box with scaled text
        draw.rectangle([label_position, (label_position[0] + text_width, label_position[1] + text_height)], fill="green")
        draw.text(label_position, label, fill="white", font=font)

    return image

def extract_text_with_trocr(image, detections, detection_threshold):
    extracted_data = []
    for detection in detections['predictions']:
        if detection['confidence'] >= detection_threshold:  # Filter by detection threshold
            x1, y1 = detection['x'] - detection['width'] / 2, detection['y'] - detection['height'] / 2
            x2, y2 = detection['x'] + detection['width'] / 2, detection['y'] + detection['height'] / 2
            cropped_plate = image.crop((x1, y1, x2, y2))
            
            cropped_plate.save("temp_plate.jpg") # Save the cropped image temporarily
            
            # Perform OCR with trocr
            ocr_result = query("temp_plate.jpg")
            print(ocr_result)

            extracted_data.append({
                "Plate Number": ocr_result,
                "Detection Confidence": detection['confidence'] * 100,  # Use detection confidence
            })

            if ocr_result is not None:
                print(f"Extracted Plate Number: {ocr_result}")
                return extracted_data
    
    return extracted_data

if __name__ == "__main__":
    run_number_plate_detection()
