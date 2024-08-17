import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import os
from auth import save_helmet_data

def run_helmet_detection():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    # Function to draw bounding boxes with labels
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

                # Draw rectangle
                cv2.rectangle(image, start_point, end_point, color, thickness)

                # Add label with background
                label = f"{pred['class']} {pred['confidence'] * 100:.2f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_width, label_height = label_size
                label_background_start = (x, y - 20)
                label_background_end = (x + label_width, y - 5)
                cv2.rectangle(image, label_background_start, label_background_end, color, -1)  # Filled rectangle

                # Add label text
                label_position = (x, y - 7)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image

    # Streamlit App Layout
    st.sidebar.title("Helmet Detection Options")

    # Sidebar Options
    confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=50, max_value=100, value=75, step=1)
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

    # Title and Info
    st.title("⛑️ Helmet Detection App")
    st.markdown("""
    Welcome to the **Helmet Detection App**! Upload an image, and we'll detect helmets using AI.  
    You can customize the detection confidence threshold and choose whether to display bounding boxes.
    """)

    # Upload Image
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image_np, caption="Uploaded Image")

        # Run inference
        with st.spinner("Running inference..."):
            result = CLIENT.infer(image, model_id="bike-helmet-detection-2vdjo-mfttu/1")
        
        # Parse predictions
        predictions = result.get("predictions", [])

        if predictions:
            # Process the image
            if show_boxes:
                processed_image = draw_boxes(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), predictions, confidence_threshold)
            else:
                processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Convert processed image back for Streamlit
            _, processed_image_buffer = cv2.imencode(".jpg", processed_image)
            processed_image_bytes = io.BytesIO(processed_image_buffer)

            # Create two columns for layout
            col1, col2 = st.columns(2)

            # Display processed image in the first column
            with col1:
                st.image(processed_image, channels="BGR", caption="Processed Image")
                # Add a download button
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
                        
                        save_helmet_data(detection_class, confidence, bounding_box, image)

                        results_data.append({
                            "Class": detection_class,
                            "Confidence (%)": f"{confidence:.2f}%",
                            "Bounding Box": bounding_box
                        })

                if results_data:
                    st.table(results_data) 
                else:
                    st.info("No detections above the confidence threshold.")

        else:
            st.warning("No helmets detected in the image!")

    else:
        st.info("Please upload an image to begin.")


    # Sidebar content
    st.sidebar.title("About Helmet Detection")
    st.sidebar.write(
        """
        Helmet detection uses AI and computer vision to identify whether two-wheeler riders are wearing helmets. 
        It helps enforce traffic regulations and promotes road safety by:
        - Encouraging helmet usage.
        - Reducing manual enforcement.
        - Providing real-time monitoring.
        """
    )

    # Detailed explanation in the main area
    st.sidebar.title("How It Works")
    st.sidebar.write(
        """
        1. **Camera Input**: Captures footage of riders in real-time.
        2. **Image Processing**: Analyzes video or images using AI models.
        3. **Helmet Identification**: Detects if a rider is wearing a helmet.
        4. **Action**: Generates alerts, fines, or logs violations.
        """
    )

    st.sidebar.title("Technologies Used")
    st.sidebar.write(
        """
        - **Computer Vision**: Processes visual data to detect helmets.
        - **Deep Learning**: Trains AI models to identify helmeted riders.
        - **IoT Integration**: Enables real-time processing and alerts.
        """
    )

    st.sidebar.title("Key Benefits")
    st.sidebar.write(
        """
        - Enhances road safety.
        - Automates rule enforcement.
        - Reduces accident-related costs.
        """
    )

    st.sidebar.title("Challenges")
    st.sidebar.write(
        """
        - Accuracy under diverse conditions.
        - Balancing privacy concerns.
        - High implementation costs.
        """
    )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit 🖥️📊✨")

if __name__ == "__main__":
    run_helmet_detection()