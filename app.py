import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow model prediction function
def model_prediction(image):
    cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions) #Return index of max element
    return result_index



# UI setup
st.set_page_config(page_title="PlantMD", page_icon="<?", layout="centered")
# st.title("Plant Disease Detection App")

st.markdown("""
    <style>
        /* Remove space at the top of the page */
        .block-container {
            padding-top: 10px !important;
        }

        /* Optional: reduce space between elements inside */
        .green-wrapper {
            margin-top: 10px;
            padding: 25px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: #3d7a47;'><? PlantMD</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #7a9979;'>Your AI-powered plant disease specialist.</p>", unsafe_allow_html=True)
st.write("Upload an image of a plant leaf to detect its disease.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=100)
    
    # Make prediction
    result_index = model_prediction(uploaded_file)
    
    # Display the result
    #st.write(f"Predicted Disease Index: {result_index}")
    # You can map the index to actual disease names if you have a dictionary
    disease_names = {
        0: "Potato Early blight",
        1: "Potato Late blight",
        2: "Potato healthy",
        3: "Tomato Bacterial spot",
        4: "Tomato Early blight",
        5: "Tomato Late blight",
        6: "Tomato Leaf Mold",
        7: "Tomato Target Spot",
        8: "Tomato healthy",
    }
    if result_index in disease_names:
        st.write(f"Predicted Disease: {disease_names[result_index]}")
    else:
        st.write("Unknown disease detected.")
    st.success("Prediction complete!")
else:
    st.warning("Please upload an image to get a prediction.")