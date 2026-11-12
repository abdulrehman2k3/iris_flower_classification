import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("iris_model.keras")
classes_names = ["setosa","versicolor","virginica"]
st.set_page_config(page_title="Iris Species Prediction", page_icon="🌸", layout="centered")
st.title("🌸Iris Species Classifier")
st.write("Enter flower measurements to predict it's species.")
sepal_length = st.number_input("Sepal Length (cm)",min_value=0.0,max_value=10.0,step=0.1)
sepal_width = st.number_input("Sepal Width (cm)",min_value=0.0,max_value=10.0,step=0.1)
petal_length = st.number_input("Petal Length (cm)",min_value=0.0,max_value=10.0,step=0.1)
petal_width = st.number_input("Petal Width (cm)",min_value=0.0,max_value=10.0,step=0.1)

if st.button("Predict"):
    new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]],dtype=np.float32)
    prediction = model.predict(new_flower)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_species = classes_names[predicted_class]
    st.success(f"The predicted species is: {predicted_species}")
    st.subheader("Prediction Probabilities")

    for i, species in enumerate(classes_names):
        probability = prediction[0][i] * 100

        st.write(f"{species.title()}: {probability:.2f}%")
        st.progress(float(prediction[0][i]))