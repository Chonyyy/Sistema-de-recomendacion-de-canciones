# Sistema de recomendación de música basado en texto utilizando el procesamiento del lenguaje natural (NLP).

## Autores:
- Sherlyn Ballestero Cruz
- María de Lourdes Choy Fernández

## Modelo implementado:
- Modelo Vector Generalizado

## Consideraciones tomadas a la hora de desarrollar la solución:
Se determinó que el modelo no funciona correctamente con un corpus de un número de canciones superior a 666. Además, se ha detectado que solo se utilizan las letras y el título, a pesar de haber podido ampliar la utilización de otros datos reunidos en la base de datos.

## Explicación de cómo ejecutar el proyecto y definición de la consulta:
Para ejecutar el proyecto:
streamlit run src/gui.py
## Explicación de la solución desarrollada:
Las funciones definidas en el código son:
- Para procesar los datos relacionados con la letra de las canciones se lleva a cabo la limpieza de estos, que están almacenados en la carpeta "data" en un archivo JSON. Estas se cargan y se limpian con las siguientes funciones:
    - **expand_contractions**: Expande las contracciones en el texto. Por ejemplo, "ain't" se expande a "am not".
    - **remove_punctuation**: Elimina la puntuación del texto.
    - **correct_spelling**: Corrige la ortografía de las palabras en el texto.
    - **preprocess_song**: Preprocesa una canción eliminando la puntuación, las palabras vacías y las contracciones.
    - **preprocess_lyrics**: Preprocesa la letra de una canción eliminando la puntuación y las palabras vacías.

Luego se procesa la consulta del usuario:
- **preprocess_input**: Preprocesa el texto de entrada eliminando la puntuación, las palabras vacías y las contracciones.

La consulta del usuario puede contener palabras con errores, para garantizar una mejor experiencia se emplea la función  `correct_spelling` , esta corrige la ortografía de las palabras en un texto. Toma un texto como entrada y devuelve un texto con las palabras corregidas. La función utiliza el módulo  `spellchecker`  para corregir la ortografía de las palabras. El módulo  `spellchecker`  utiliza un diccionario de palabras en inglés para corregir la ortografía de las palabras. Primero divide el texto en palabras. Luego, se comprueba si cada palabra está en el diccionario de palabras en inglés. Si una palabra no está en el diccionario, la función utiliza el módulo  `spellchecker`  para corregir la ortografía de la palabra y esta devuelve un texto con las palabras corregidas.

Se aplica el modelo vectorial, empleando el modelo de word embedding:
```cs
model = Word2Vec(sentences, window=5, min_count=1, workers=4)
model.save("word2vec.model")

```
Word2Vec: Entrena un modelo para aprender representaciones vectoriales de palabras basadas en su contexto. Y se vectorizan las canciones y la entrada procesadas, utilizando el modelo vectorial dado en clases para calcular la similitud de estas con  `cosine_similarity` .

- **get_song_similarities**: Calcula la similitud entre el texto de entrada y las canciones preprocesadas.
- **recommend_songs**: Recomienda canciones basadas en el texto de entrada.

Se lleva a cabo el flujo del proyecto, esta recibe la consulta, llama a las funciones de procesamiento, inicializa el modelo mencionado y se calculan las similitudes a partir de  `get_song_similarities` .

## Insuficiencias:
Se hizo la prueba de colocar partes grandes de ciertas letras de canciones y el modelo da más puntuación a otras canciones, se considera necesario verificar la efectividad de otros modelos preentrenados para garantizar mejores resultados.