# sistema de recomendación de música basado en texto utilizando el procesamiento del lenguaje natural (NLP).
## Autores:
-Sherlyn Ballestero Cruz
-Maria de Lourdes Choy Fernández
## modelo de SRI implementado.
-Modelo Vector Generalizado
## Consideraciones tomadas a la hora de desarrollar la solución.
##  Explicación de cómo ejecutar el proyecto. Definición de la consulta.
Para ejecutar el proyecto:
-streamlit run src/gui.py 
##  Explicación de la solución desarrollada
Las funciones definidas en el código son:
Para procesar los datos relacionados con la letra de las canciones se lleva a cabo la limpieza de estos, que están almacenadas en la carpeta data en un archivo json, estas se cargan y se limpian con las siguientes funciones:
**expand_contractions**: Expande las contracciones en el texto. Por ejemplo, "ain't" se expande a "am not".

**remove_punctuation**: Elimina la puntuación del texto.

**correct_spelling**: Corrige la ortografía de las palabras en el texto.

**preprocess_song**: Preprocesa una canción eliminando la puntuación, las palabras vacías y las contracciones.

**preprocess_lyrics**: Preprocesa la letra de una canción eliminando la puntuación y las palabras vacías.
Luego se procesa la consulta del usuario:
**preprocess_input**: Preprocesa el texto de entrada eliminando la puntuación, las palabras vacías y las contracciones.
La consulta del usuario puede contener palabras con errores, para garantizar una mejor experiencia de se emplea la función `correct_spelling`, esta corrige la ortografía de las palabras en un texto.Toma un texto como entrada y devuelve un texto con las palabras corregidas.La función utiliza el módulo `spellchecker` para corregir la ortografía de las palabras.El módulo `spellchecker` utiliza un diccionario de palabras en inglés para corregir la ortografía de las palabras.Primero divide el texto en palabras.Luego, se comprueba si cada palabra está en el diccionario de palabras en inglés.Si una palabra no está en el diccionario, la función utiliza el módulo `spellchecker` para corregir la ortografía de la palabra y esta devuelve un texto con las palabras corregidas. 

Se aplica el modelo vectorial,empleando el modelo escogido:
```cs 
    model = Word2Vec(sentences, window=5, min_count=1, workers=4)
    model.save("word2vec.model") 
```
Word2Vec: Entrena un modelo para aprender representaciones vectoriales de palabras basadas en su contexto.
y se vectorizan las canciones y la entrada processadas, ppara utilizando el modelo vectorial dado en clases calcular la similitud de estas con cosine_similarity.
**get_song_similarities**: Calcula la similitud entre el texto de entrada y las canciones preprocesadas.
con:
**recommend_songs**: Recomienda canciones basadas en el texto de entrada.
Se lleva a cabo el flujo del proyecto, esta recibe la consulta, llama a las funciones de procesamiento, inicializa el modelo mencionado y se calculan las similitudes a partir  de get_song_similarities.
##  Insuficiencias 
Se hizo la prueba de colocar partes grandes de ciertas letras de canciones y el modelo le da mas puntuación a otras canciones, se considera necesario verificar la efectividad de otros modelos preentrenados para garantizar mejores resultados. 





