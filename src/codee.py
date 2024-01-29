from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Paso 1: Recolección de datos
# Aquí necesitarías obtener tus datos de canciones y letras

# Paso 2: Preprocesamiento de datos
# Limpiamos las letras y las convertimos a vectores numéricos

# Paso 3: Embeddings de letras
model = Doc2Vec.load("path_to_your_trained_model")

# Paso 4: Similitud de canciones
def get_song_similarities(text):
    # Transformamos el texto en un vector de incrustación
    text_vector = model.infer_vector(text)
    
    # Calculamos la similitud con todas las canciones en nuestra base de datos
    similarities = cosine_similarity([text_vector], model.docvecs.vectors_docs)
    
    return similarities

# Paso 5: Generación de recomendaciones
def recommend_songs(text):
    similarities = get_song_similarities(text)
    
    # Obtenemos las canciones más similares
    sorted_indexes = np.argsort(-similarities)
    
    return [database[i] for i in sorted_indexes]
