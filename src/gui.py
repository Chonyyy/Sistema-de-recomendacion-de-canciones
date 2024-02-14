
import streamlit as st
import pandas as pd
import codee
import string


st.title("Mi aplicación de recomendación de canciones")
st.subheader("Introduce el texto que desees:")


#  input_text es una variable que contiene el texto introducido por el usuario
input_text = st.text_input("Text", key="name")

# Calcula las canciones recomendadas basándose en el texto de entrada
recommended_songs,fixed_words = codee.recommend_songs(input_text)

st.write("Entrada corregida:")
st.write(" ".join(fixed_words))

# Muestra las canciones recomendadas en la aplicación Streamlit
st.markdown("<h2 style='text-align: center;'>Canciones Recomendadas</h2>", unsafe_allow_html=True)
st.table(pd.DataFrame(recommended_songs, columns=["Nombre de la Canción"]))

