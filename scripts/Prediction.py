from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import pickle

st.title("Предсказание вероятностей")

uploaded_file = st.file_uploader("Выберите файл датасета")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Загруженный датасет:", df)

st.title("Получить предсказания на основе данных")

est_diameter_min = st.number_input("est_diameter_min:", value=0.81)
est_diameter_max = st.number_input("est_diameter_max:", value=1.81)
relative_velocity = st.number_input("relative_velocity:", value=1072340.79)
miss_distance = st.number_input("miss_distance:", value=2473712.69)
absolute_magnitude = st.number_input("absolute_magnitude:", value=17.58)


data = pd.DataFrame({'est_diameter_min': [est_diameter_min],
                    'est_diameter_max': [est_diameter_max],
                    'relative_velocity': [relative_velocity],
                    'miss_distance': [miss_distance],
                    'absolute_magnitude': [absolute_magnitude],})

knn = pickle.load(open(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Knn.pickle', 'rb'))
kmeans_model = pickle.load(open(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Kmeans.pickle', 'rb'))
bagging_model = pickle.load(open(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Bagging.pickle', 'rb'))
gradient_model = pickle.load(open(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Gradient.pickle', 'rb'))
stacking_model = pickle.load(open(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Stacking.pickle', 'rb'))
nn_model = load_model(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\models\Neural.h5')

button_clicked = st.button("Предсказать")

if button_clicked:

    knn_pred = knn.predict(data)[0]
    st.write(f"KNN: {knn_pred}")

    kmeans_pred = kmeans_model.predict(data)[0]
    st.write(f"Kmeans: {kmeans_pred}")

    bagging_pred = bagging_model.predict(data)[0]
    st.write(f"Bagging: {bagging_pred}")

    gradient_pred = gradient_model.predict(data)[0]
    st.write(f"Gradient: {gradient_pred}")

    stacking_pred = stacking_model.predict(data)[0]
    st.write(f"Stacking: {stacking_model.predict(data)[0]}")

    nn_pred = round(nn_model.predict(data)[0][0])
    st.write(f"Perceptron: {nn_pred}")