import json
import string
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

def expand_contractions(text):
    '''
    Expande las contracciones en el texto. Por ejemplo, "ain't" se expande a "am not".
    
    Parameters:
        text (str): Texto de entrada para expandir.
    
    Returns:
        str: Texto con las contracciones expandidas.
    '''
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
    '''
    Elimina la puntuación del texto.
    
    Parameters:
        text (str): Texto de entrada.
    
    Returns:
        str: Texto sin puntuación.
    '''
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)



def correct_spelling(text):
    '''
    Corrige la ortografía de las palabras en el texto.
    
    Parameters:
        text (str): Texto de entrada.
    
    Returns:
        tuple: Tupla que contiene el texto con las palabras corregidas y una lista de las palabras corregidas.
    '''
    
    english_words = set(words.words())
    spell = SpellChecker()
    input_words = text.split()
    correction_cache = {}
    corrected_words = []
    for word in input_words:  
        if word not in correction_cache:
            if word.lower() not in english_words:
                correction = spell.correction(word)
                correction_cache[word] = correction if correction is not None else word
            else:
                correction_cache[word] = word
        corrected_words.append(correction_cache[word])
    return ' '.join(corrected_words),corrected_words


def preprocess_song(song):
    '''
    Preprocesa una canción eliminando la puntuación, las palabras vacías y las contracciones.
    
    Parameters:
        song (dict): Diccionario que contiene los datos de una canción.
    
    Returns:
        str: Letra de la canción preprocesada.
    '''
    stop_words = set(stopwords.words('english'))
    lyrics = song['lyrics'].lower()
    lyrics = expand_contractions(lyrics)
    lyrics = remove_punctuation(lyrics)
    tokens = word_tokenize(lyrics)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_input(input_text):
    '''
    Preprocesa el texto de entrada eliminando la puntuación, las palabras vacías y las contracciones.
    
    Parameters:
        input_text (str): Texto de entrada.
    
    Returns:
        tuple: Tupla que contiene el texto de entrada preprocesado y una lista de las palabras corregidas.
    '''
    stop_words = set(stopwords.words('english'))
    input_text = expand_contractions(input_text)
    input_text = remove_punctuation(input_text)
    input_text,fixed_words = correct_spelling(input_text)  # Agrega la corrección ortográfica aquí
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens),fixed_words

def preprocess_lyrics(input_text):
    '''
    Preprocesa la letra de una canción eliminando la puntuación y las palabras vacías.
    
    Parameters:
        input_text (str): Texto de entrada.
    
    Returns:
        str: Letra de la canción preprocesada.
    '''
    stop_words = set(stopwords.words('english'))
    input_text = expand_contractions(input_text)
    input_text = remove_punctuation(input_text)
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


#calcular la similitud entre las canciones:
def get_song_similarities(input_text,model,preprocessed_songs):
    '''
    Calcula la similitud entre el texto de entrada y las canciones preprocesadas.
    
    Parameters:
        input_text (str): Texto de entrada.
        model: Modelo Word2Vec utilizado para calcular las similitudes.
        preprocessed_songs (dict): Diccionario que contiene las canciones preprocesadas.
    
    Returns:
        list: Lista de tuplas que contiene el título de la canción y su similitud con el texto de entrada.
        list: Lista de las palabras corregidas en el texto de entrada.
    '''
    input_tokens,fixed_words = preprocess_input(input_text)
    input_tokens = input_tokens.split()
    input_vectors = [model.wv[token] for token in input_tokens if token in model.wv]
    similarities = []
    
    try:
        for song_title, song_lyrics in preprocessed_songs.items():
            song_tokens = preprocess_lyrics(song_lyrics).split()
            song_vectors = [model.wv[token] for token in song_tokens if token in model.wv]
            similarity = cosine_similarity(input_vectors, song_vectors)[0, 0] # Obtén el valor de similitud de la matriz
            similarities.append((song_title, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
    except:
        print("Empty")    
        
    return similarities,fixed_words


#generar recomendaciones:
def recommend_songs(input_text):
    '''
    Genera recomendaciones de canciones basadas en el texto de entrada.
    
    Parameters:
        input_text (str): Texto de entrada.
    
    Returns:
        list: Lista de títulos de canciones recomendadas.
        list: Lista de las palabras corregidas en el texto de entrada.
    '''
    with open('data/songs1.json', 'r',encoding='utf-8') as f:
        try:
            songs = json.load(f)
        except:
            print("Empty")
            
            
    preprocessed_songs = {title: preprocess_song(song) for title, song in songs.items()}
    
    #crear los embeddings de las letras usando Word2Vec:
    sentences = [list(map(str, lyric.split())) for lyric in preprocessed_songs.values()] 
    model = Word2Vec(sentences, window=5, min_count=1, workers=4)
    model.save("word2vec.model")       
    similarities,fixed_words = get_song_similarities(input_text,model,preprocessed_songs)
    return [song for song, _ in similarities[:10]],fixed_words


