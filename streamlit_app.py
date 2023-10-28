import streamlit as st
import cv2
from keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np

char_list = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

saved_model = load_model("full_model.h5")

# Define custom CSS styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('background.jpg');  # Replace 'your_image.jpg' with your image file's path.
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def process_image(img):
    original_height, original_width = img.shape
    source_aspect_ratio = original_width / original_height

    # if image width is smaller, fill the right side with white space
    if source_aspect_ratio < 4:
        new_height = 32
        new_width = int(original_width * new_height / original_height)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        img = np.concatenate((img, np.full((new_height, 128-new_width), 255)), axis=1)

    # in terms of a longer text, squeeze it into the desired shape
    if source_aspect_ratio > 4:
        new_height = 32
        new_width = 128
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # just making sure it's in correct shape
    if img.shape != (32,128):
        img = cv2.resize(img, (128,32))

    # adding an extra dimension in the end and normalizing
    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)
    img = img / 255

    return img

def predict_image(img):
    pred = saved_model.predict(np.expand_dims(img, axis=0))
    out = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    letters = ''
    for x in out[0]:
        if int(x) != -1:
            letters += char_list[int(x)]
    return letters

st.title("Handwriting Extraction from images")
st.subheader('Detects handwritten text from an image and transcribes it')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 0)
    processed_image = process_image(image)
    result = str(predict_image(processed_image))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted word : ", result)
else:
    st.warning("Upload an image above.")