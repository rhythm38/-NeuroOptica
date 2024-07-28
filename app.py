import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Function to classify the uploaded image
def teachable_machine_classification(img, model_path):
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Assuming labels are in 'labels.txt' in the same directory
    class_names = open('labels.txt', "r").readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="NeuroOptica - AI-Based Medical Image Analyzer", page_icon="🧠", layout="centered")
    
    # Set a custom background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f0f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🧠 AI-Based Medical Image Analyzer")
    st.header("Brain Tumor Classification")
    st.write("Upload a Brain MRI image for classification into tumor or non-tumor categories.")
    st.markdown("---")

    # File uploader for the user to upload a brain MRI image
    uploaded_file = st.file_uploader("Choose a Brain MRI...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image_file = Image.open(uploaded_file)

        # Convert image to RGB if not already
        if image_file.mode != 'RGB':
            image_file = image_file.convert('RGB')

        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        # Classify the uploaded image
        model_path = 'keras_model.h5'  # Path to the uploaded model
        with st.spinner("Classifying... Please wait."):
            label, confidence_score = teachable_machine_classification(image_file, model_path)
        
        if "Yes" in label:
            st.error(f"MRI scan indicates a brain tumor with a confidence score of {confidence_score:.2f}.")
        else:
            st.success(f"No brain tumor found in the provided MRI image. Confidence score: {confidence_score:.2f}.")
    
    # Add additional resources or instructions
    st.markdown("---")
    st.subheader("How to Use This Application:")
    st.write("""
        1. Upload a Brain MRI image in PNG or JPEG format.
        2. Wait for the model to classify the image.
        3. Review the results displayed below the image.
    """)

    # Footer with contact information or additional links
    st.markdown("---")
    st.write("Developed by Rhythm Arya")
    st.write("For questions or feedback, please contact: [rhythmarya@gmail.com](mailto:rhythmarya38@gmail.com)")

if __name__ == "__main__":
    main()
