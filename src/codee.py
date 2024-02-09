import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_song(song):
    stop_words = set(stopwords.words('english'))
    lyrics = song['lyrics'].lower().split()
    tokens = word_tokenize(' '.join(lyrics))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_input(input_text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)



#calcular la similitud entre las canciones:
def get_song_similarities(input_text,model,preprocessed_songs):
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
        print("Empty")    
        
    return similarities


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
    similarities = get_song_similarities(input_text,model,preprocessed_songs)
    return [song for song, _ in similarities[:10]]


#test
#esperado:Lose Yourself
input_text="certified quality a dat da girl dem need and dem not stop cry without apology buck dem da right waydat my policy sean paul alongsidenow hear what da man say beyonce dutty ya dutty ya dutty ya beyonce sing it now ya baby boy you stay on my mind fulfill my fantasies i think about you all the time i see you in my dreams baby boy not a day goes by without my fantasies i think about you all the time i see you in my dreams ah oh my baby's fly baby oh yes no hurt me so good baby oh i'm so wrapped up in your love let me go let me breathe stay out my fantasies ya ready gimme da ting dat ya ready get ya live and tell me all about da tings that you will fantasize i know you dig da way me step da way me make my stride follow your feelings "
print(recommend_songs(input_text))
