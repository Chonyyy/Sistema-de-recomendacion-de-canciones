import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

input_text=" neck cca jdark messy hotel "

with open('data/songs.json', 'r') as f:
    try:
        songs = json.load(f)
    except:
        print("Empty")

# convertir todo a minúsculas y eliminar las palabras irrelevantes
stop_words = set(stopwords.words('english'))#Las stopwords son palabras que se omiten durante el procesamiento de texto en tareas de Procesamiento del Lenguaje Natural (NLP) porque suelen no aportar mucho significado en el contexto de un texto.

def preprocess_song(song):
    lyrics = song['lyrics'].lower().split()
    tokens = word_tokenize(' '.join(lyrics))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_input(input_text):
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

preprocessed_songs = {title: preprocess_song(song) for title, song in songs.items()}
preprocessed_text = preprocess_input(input_text)

#crear los embeddings de las letras usando Word2Vec:
sentences = [list(map(str, lyric.split())) for lyric in preprocessed_songs.values()]

model = Word2Vec(sentences, window=5, min_count=1, workers=4)
model.save("word2vec.model")

#calcular la similitud entre las canciones:
def get_song_similarities(input_text):
    input_tokens = preprocess_input(input_text).split()
    input_vectors = [model.wv[token] for token in input_tokens if token in model.wv]
    similarities = []
    try:
        for song_title, song_lyrics in preprocessed_songs.items():
            song_tokens = preprocess_input(song_lyrics).split()
            song_vectors = [model.wv[token] for token in song_tokens if token in model.wv]
            similarity = cosine_similarity(input_vectors, song_vectors)[0, 0] # Obtén el valor de similitud de la matriz
            similarities.append((song_title, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
    except:
        print("No matches")
        
    return similarities


#generar recomendaciones:
def recommend_songs(input_text):
    similarities = get_song_similarities(input_text)
    return [song for song, _ in similarities[:2]]

print(recommend_songs(input_text))
