import streamlit as st
from PIL import Image
from util import predictor


def main():
    st.title("Image Classifier App")

    # Sidebar for user input

    crop_image = st.sidebar.checkbox("Crop Image", value=False)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = Image.open(uploaded_file).convert("RGB")

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run prediction when the user clicks the button
        if st.button("Run Prediction"):
            probability, index = predictor(image)

            if index == 0:
                class_name = "Foot Ulcer"
            else:
                class_name = "Normal"

            if class_name is not None:
                st.success(f"Predicted Class: {class_name}, Probability: {probability*100:.2f}%")
            else:
                st.warning("Prediction failed.")
    else:
        st.info("Please upload an image.")


# Run the Streamlit app
if __name__ == "__main__":
    main()
