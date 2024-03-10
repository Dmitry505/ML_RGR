import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Osnova\Programming\Python\ML\test\ML_RGR\data\classification.csv')

st.title("Визуализация")

st.header("Тепловая карта")
correct = ['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude','hazardous']
new_df = df[correct]
plt.figure(figsize=(12, 8))
sns.heatmap(new_df.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта')
st.pyplot(plt)

st.header("Парная диограмма")
plt.figure(figsize=(12, 8))
sns.pairplot(data=df)
plt.title('Парная диограмма')
st.pyplot(plt)

st.header("Круговая диаграмма")
plt.figure(figsize=(8, 8))
df['hazardous'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('hazardous')
st.pyplot(plt)

st.header("Гистограммы")
correct_4 = ['relative_velocity','miss_distance','absolute_magnitude']
for graf in correct_4:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)[graf], bins=100, kde=True)
    plt.title(graf)
    st.pyplot(plt)