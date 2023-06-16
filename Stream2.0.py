import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# Title and information
st.title('Birds and Animal Species Prediction')

# Add your name as a suffix using HTML and CSS
st.markdown('<p style="text-align: right; font-size: small;">By Bhavesh Kumar</p>', unsafe_allow_html=True)

# Add a sidebar for more options
st.sidebar.header('Options')
selected_model = st.sidebar.selectbox('Select Animal', ['All animals', 'Bird', 'Cat', 'Cow', 'Dog', 'Horse', 'Monkey'])

# Text based on selected model
if selected_model == 'All animals':
    st.write('Predict species of Birds, Cats, Cows, Dogs, Horses and Monkeys')
else:
    st.write(f'Predict {selected_model} species')

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

df = pd.read_csv(r"C:\Users\Aman\Downloads\Animal Species.csv")

button = st.button('Predict')

@st.cache_resource()
def loading_models():
    # Load models
    bird_model = tf.keras.models.load_model(r'E:\Animal Models\bird.h5')
    bird_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cat_model = tf.keras.models.load_model(r'E:\Animal Models\cat_no-norm.h5')
    cat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cow_model = tf.keras.models.load_model(r'E:\Animal Models\ensemble_cow.h5')
    cow_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    dog_model = tf.keras.models.load_model(r'E:\Animal Models\dog_cv2_no-norm.h5')
    dog_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    horse_model = tf.keras.models.load_model(r'E:\Animal Models\ensemble_horse.h5')
    horse_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    monkey_model = tf.keras.models.load_model(r'E:\Animal Models\ensemble_monkey.h5')
    monkey_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    import tensorflow_hub as hub
    custom_objects = {
        'KerasLayer': hub.KerasLayer
    }
    animal = tf.keras.models.load_model(r'E:\Animal Models\ensemble_model.h5', custom_objects=custom_objects)
    animal.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return bird_model, cat_model, cow_model, dog_model, horse_model, monkey_model, animal

bird_model, cat_model, cow_model, dog_model, horse_model, monkey_model, animal = loading_models()

# Preprocess and predict the image
if uploaded_file and button:
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = image.resize((75, 75))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    col1, col2 = st.columns(2)

    if selected_model == 'All animals':
        predictions = animal.predict(image_array)
        name = df['Animals'][np.argmax(predictions)]
        
        if np.argmax(predictions) == 0:
            species = df['Bird'][np.argmax(bird_model.predict(image_array))]
        elif np.argmax(predictions) == 1:
            species = df['Cat'][np.argmax(cat_model.predict(np.expand_dims(np.array(image), axis=0)))]
        elif np.argmax(predictions) == 2:
            species = df['Cow'][np.argmax(cow_model.predict(image_array))]
        elif np.argmax(predictions) == 3:
            image_cv2 = np.expand_dims(np.array(cv2.resize(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR), (75, 75))), axis=0)
            species = df['Dog'][np.argmax(dog_model.predict(image_cv2))]
        elif np.argmax(predictions) == 4:
            species = df['Horse'][np.argmax(horse_model.predict(image_array))]
        elif np.argmax(predictions) == 5:
            species = df['Monkey'][np.argmax(monkey_model.predict(image_array))]

    else:
        name = selected_model
        if selected_model == 'Bird':
            species = df['Bird'][np.argmax(bird_model.predict(image_array))]
        elif selected_model == 'Cat':
            species = df['Cat'][np.argmax(cat_model.predict(np.expand_dims(np.array(image), axis=0)))]
        elif selected_model == 'Cow':
            species = df['Cow'][np.argmax(cow_model.predict(image_array))]
        elif selected_model == 'Dog':
            image_cv2 = np.expand_dims(np.array(cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), (75, 75))), axis=0)
            species = df['Dog'][np.argmax(dog_model.predict(image_cv2))]
        elif selected_model == 'Horse':
            species = df['Horse'][np.argmax(horse_model.predict(image_array))]
        elif selected_model == 'Monkey':
            species = df['Monkey'][np.argmax(monkey_model.predict(image_array))]

    st.image(cv2.cvtColor(np.array(Image.open(uploaded_file)), cv2.COLOR_RGB2BGR), channels="BGR")
    with col1:
        st.write(f'Animal(s): {name}s')
    with col2:
        st.write(f'Species: {species}')
    
