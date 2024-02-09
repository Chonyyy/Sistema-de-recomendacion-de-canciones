
import streamlit as st
import pandas as pd
import codee
import string

def expand_contractions(text):
    contraction_dict = {"ain't": "am not", "aren't": "are not","can't": "cannot", "'cause": "because",  
                        "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",  
                        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not"}
    words = text.split()
    expanded_words = []
    for word in words:
        if word in contraction_dict:
            expanded_words.extend(contraction_dict[word].split())
        else:
            expanded_words.append(word)
    return ' '.join(expanded_words)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def preprocess_song(song):
    stop_words = set(stopwords.words('english'))
    lyrics = song['lyrics'].lower()
    lyrics = expand_contractions(lyrics)
    lyrics = remove_punctuation(lyrics)
    tokens = word_tokenize(lyrics)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_input(input_text):
    stop_words = set(stopwords.words('english'))
    input_text = expand_contractions(input_text)
    input_text = remove_punctuation(input_text)
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


st.title("Mi aplicación de recomendación de canciones")
st.subheader("Introduce el texto que desees:")


#  input_text es una variable que contiene el texto introducido por el usuario
input_text = st.text_input("Text", key="name")

# Calcula las canciones recomendadas basándose en el texto de entrada
recommended_songs = codee.recommend_songs(input_text)

# Muestra las canciones recomendadas en la aplicación Streamlit
st.markdown("<h2 style='text-align: center;'>Canciónes Recomendadas</h2>", unsafe_allow_html=True)
st.table(pd.DataFrame(recommended_songs, columns=["Nombre de la Canción"]))

