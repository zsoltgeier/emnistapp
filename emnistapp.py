import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import cv2
import numpy as np

@st.cache_resource
def load_models():
    model_cnn = tf.keras.models.load_model('models/CNN_20_epoch_64')
    model_cnn_augmented = tf.keras.models.load_model('models/CNN_20_epoch_64_augmented')
    return model_cnn, model_cnn_augmented

# Load models
model_cnn, model_cnn_augmented = load_models()

label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U',
    30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'
}


def sort_images_by_label(images):
    def image_sort_key(image):
        # Extracting the basename (e.g., 'A' from 'A.png')
        base_name = os.path.splitext(image)[0]
        # Finding the index of each image label in the ordered label_map
        for key, value in label_map.items():
            if value == base_name:
                return key
        return len(label_map)  # Place unknown labels at the end

    return sorted(images, key=image_sort_key)


def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input
    image_np = np.array(image)
    image_np = image_np.astype(np.float32) / 255.0  # Normalize
    image_np = image_np.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return image_np

def apply_affine_transformation(image, rotation=0, scale=1.0, translate_x=0, translate_y=0):
    image_np = np.array(image)
    (height, width) = image_np.shape[:2]
    center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, rotation, scale)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])

    nW, nH = int((height * sin) + (width * cos)), int((height * cos) + (width * sin))

    M[0, 2] += (nW / 2) - center[0] + translate_x
    M[1, 2] += (nH / 2) - center[1] + translate_y

    rotated = cv2.warpAffine(image_np, M, (nW, nH))
    image_pil = Image.fromarray(rotated)
    return image_pil

def get_prediction_and_confidence(model, image):
    predictions = model.predict(image)
    class_id = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    return class_id, confidence

def main():
    st.title("EMNIST Handwrritten Digit and Letter Recognition App")

    images_folder = 'class_images'
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    image_files = sort_images_by_label(image_files)

    image_files_no_ext = [os.path.splitext(f)[0] for f in image_files]

    key_suffix = st.session_state.get('reset_key_suffix', 0)

    if image_files:
        selected_image_no_ext = st.selectbox("Select an image", image_files_no_ext, key=f'select_image_{key_suffix}')
        selected_image = next((f for f in image_files if f.startswith(selected_image_no_ext)), None)
        

        if selected_image:
            image_path = os.path.join(images_folder, selected_image)
            image = Image.open(image_path)

            st.sidebar.header("Affine Transformations")
            rotation = st.sidebar.slider("Rotation", -180, 180, 0, key=f'rotation_{key_suffix}')
            scale = st.sidebar.slider("Scale", 0.1, 5.0, 1.0, 0.01, key=f'scale_{key_suffix}')
            translate_x = st.sidebar.slider("Translate X", -28, 28, 0, key=f'translate_x_{key_suffix}')
            translate_y = st.sidebar.slider("Translate Y", -28, 28, 0, key=f'translate_y_{key_suffix}')

            if st.sidebar.button("Reset"):
                st.session_state.reset_key_suffix = key_suffix + 1
                st.experimental_rerun()

            transformed_image = apply_affine_transformation(image, rotation, scale, translate_x, translate_y)

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(transformed_image, use_column_width=True)

                if st.button("Predict"):
                    preprocessed_image = preprocess_image(transformed_image)
                    cnn_class_id, cnn_confidence = get_prediction_and_confidence(model_cnn, preprocessed_image)
                    cnn_augmented_class_id, cnn_augmented_confidence = get_prediction_and_confidence(model_cnn_augmented, preprocessed_image)

                    st.markdown("""
                    **CNN**:   
                    Prediction: `{}`   
                    Confidence: `{:.2%}`   
                    """.format(label_map[cnn_class_id], cnn_confidence))

                    st.markdown("""
                    **CNN trained on augmented data**:   
                    Prediction: `{}`   
                    Confidence: `{:.2%}`   
                    """.format(label_map[cnn_augmented_class_id], cnn_augmented_confidence))
    else:
        st.write("No images found in the folder.")

if __name__ == "__main__":
    main()
