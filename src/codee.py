import json
import string
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

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



def correct_spelling(text):
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
    input_text,fixed_words = correct_spelling(input_text)  # Agrega la corrección ortográfica aquí
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens),fixed_words

def preprocess_lyrics(input_text):
    stop_words = set(stopwords.words('english'))
    input_text = expand_contractions(input_text)
    input_text = remove_punctuation(input_text)
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


#calcular la similitud entre las canciones:
def get_song_similarities(input_text,model,preprocessed_songs):
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
    
    print(input_text)
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


#test
input_text="certified quality a dat da girl dem need and dem not stop cry without apology buck dem da right waydat my policy sean paul alongsidenow hear what da man say beyonce dutty ya dutty ya dutty ya beyonce sing it now ya baby boy you stay on my mind fulfill my fantasies i think about you all the time i see you in my dreams baby boy not a day goes by without my fantasies i think about you all the time i see you in my dreams ah oh my baby's fly baby oh yes no hurt me so good baby oh i'm so wrapped up in your love let me go let me breathe stay out my fantasies ya ready gimme da ting dat ya ready get ya live and tell me all about da tings that you will fantasize i know you dig da way me step da way me make my stride follow your feelings "
input_text2="wofgh"
print(recommend_songs(input_text2))
