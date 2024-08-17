import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import tempfile
import os
from auth import save_triple_ride_data


def run_triple_ride_detection():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key  
    )

    def draw_boxes(image, predictions, confidence_threshold):
        for pred in predictions:
            if pred['confidence'] * 100 >= confidence_threshold:
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                width = int(pred["width"])
                height = int(pred["height"])
                start_point = (x, y)
                end_point = (x + width, y + height)
                color = (0, 255, 0)  # Green
                thickness = 2

                
                cv2.rectangle(image, start_point, end_point, color, thickness)

                
                label = f"{pred['class']} {pred['confidence'] * 100:.2f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_width, label_height = label_size
                label_background_start = (x, y - 20)
                label_background_end = (x + label_width, y - 5)
                cv2.rectangle(image, label_background_start, label_background_end, color, -1)  

                label_position = (x, y - 7)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image

    st.sidebar.title("Triple Ride Detection Options")

    confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=50, max_value=100, value=75, step=1)
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

    st.title("🛵 Triple Ride Detection App")
    st.markdown("""
    Welcome to the **Triple Ride Detection App**! Upload an image, and we'll detect triple rides using AI.  
    You can customize the detection confidence threshold and choose whether to display bounding boxes around the detected riders.
    """)

    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image_np, caption="Uploaded Image")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = temp_file.name
            image.save(temp_file, format="JPEG")

        with st.spinner("Running inference..."):
            result = CLIENT.infer(temp_file_path, model_id="triple-ride/1")  
        
        predictions = result.get("predictions", [])

        if predictions:
    
            if show_boxes:
                processed_image = draw_boxes(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), predictions, confidence_threshold)
            else:
                processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            _, processed_image_buffer = cv2.imencode(".jpg", processed_image)
            processed_image_bytes = io.BytesIO(processed_image_buffer)

            col1, col2 = st.columns(2)

            with col1:
                st.image(processed_image, channels="BGR", caption="Processed Image")
                st.download_button(
                    label="Download Processed Image",
                    data=processed_image_bytes,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

            with col2:
                st.markdown("### 🛠 Detection Summary:")
                st.markdown(f"**Total Detections:** {len(predictions)}")
                
                results_data = []
                for pred in predictions:
                    if pred['confidence'] * 100 >= confidence_threshold:
                        detection_class = pred["class"]
                        confidence = pred["confidence"] * 100
                        bounding_box = f"({int(pred['x'])}, {int(pred['y'])}, {int(pred['width'])}, {int(pred['height'])})"
                        
                        save_triple_ride_data(detection_class, confidence, bounding_box, image)

                        results_data.append({
                            "Class": detection_class,
                            "Confidence (%)": f"{confidence:.2f}%",
                            "Bounding Box": bounding_box
                        })

                if results_data:
                    st.table(results_data)  
                else:
                    st.info("No triple ride detections above the confidence threshold.")

        else:
            st.warning("No triple rides detected in the image!")

    else:
        st.info("Please upload an image to begin.")

    st.sidebar.title("About Triple Ride Detection")
    st.sidebar.write(
        """
        Triple Ride Detection uses AI and computer vision to identify when three riders are on a single vehicle, which is considered a traffic violation in many regions.  
        This system automatically detects such instances from images, helping authorities in:
        - Enforcing traffic laws.
        - Promoting road safety.
        - Reducing manual surveillance effort.
        """
    )

    st.sidebar.title("How It Works")
    st.sidebar.write(
        """
        1. **Image Capture**: The app receives an image that may contain a triple ride scenario.
        2. **AI Model Processing**: The image is analyzed by a trained deep learning model.
        3. **Detection**: The model identifies the presence of three riders on a single vehicle.
        4. **Action**: If detected, the system highlights the violation and provides the detection details.
        """
    )

    st.sidebar.title("Technologies Used")
    st.sidebar.write(
        """
        - **Computer Vision**: Processes the visual data to detect the triple ride scenario.
        - **Deep Learning**: Trains AI models to detect such violations based on labeled data.
        - **Image Processing**: Analyzes images with bounding boxes and labels for easy identification.
        """
    )

    st.sidebar.title("Key Benefits")
    st.sidebar.write(
        """
        - Promotes safe road practices.
        - Assists law enforcement in monitoring traffic violations.
        - Reduces manual enforcement effort.
        """
    )

    st.sidebar.title("Challenges")
    st.sidebar.write(
        """
        - Handling images with unclear or partial views.
        - Environmental factors affecting image quality.
        - Accuracy in diverse traffic conditions.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit 🖥️📊✨")

if __name__ == "__main__":
    run_triple_ride_detection()