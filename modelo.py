import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import json
import pickle

# 1. Función para cargar datos desde JSONL
def cargar_datos_jsonl(archivo_jsonl):
    frases = []
    labels_numericos = []
    with open(archivo_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                texto = data.get('text')
                label_str = data.get('label')

                if texto and label_str is not None:
                    frases.append(texto)
                    labels_numericos.append(int(label_str)) 
            except json.JSONDecodeError:
                print(f"Error al decodificar JSON en la línea: {line.strip()}") 
                continue 

    return frases, labels_numericos

# Ruta de tweet
train_file = 'dataset/train/tweet.jsonl'
validation_file = 'dataset/validation/tweet.jsonl'
test_file = 'dataset/test/tweet.jsonl'
# Ruta de amazon
amazon_train_file = 'dataset/train/amazon.jsonl'
amazon_validation_file = 'dataset/validation/amazon.jsonl'
amazon_test_file = 'dataset/test/amazon.jsonl'

# Cargar datos de entrenamiento
frases_entrenamiento_tweet, labels_numericos_entrenamiento_tweet = cargar_datos_jsonl(train_file)
frases_validacion_tweet, labels_numericos_validacion_tweet = cargar_datos_jsonl(validation_file)
frases_prueba_tweet, labels_numericos_prueba_tweet = cargar_datos_jsonl(test_file)

# Cargar datos de entrenamiento de Amazon
frases_entrenamiento_amazon, labels_numericos_entrenamiento_amazon = cargar_datos_jsonl(amazon_train_file)
frases_validacion_amazon, labels_numericos_validacion_amazon = cargar_datos_jsonl(amazon_validation_file)
frases_prueba_amazon, labels_numericos_prueba_amazon = cargar_datos_jsonl(amazon_test_file)


# Mapeo de etiquetas numéricas a nombres de sentimiento (sin cambios)
indice_a_sentimiento = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
sentimiento_a_indice = {v: k for k, v in indice_a_sentimiento.items()}

# Función para convertir etiquetas de Amazon (0-4) a etiquetas unificadas (0-2)
def convertir_labels_amazon(labels_amazon):
    labels_unificadas = []
    for label in labels_amazon:
        if label <= 1: # 0 y 1 de amazon a 0 (Negativo)
            labels_unificadas.append(0)
        elif label == 2: # 2 de amazon a 1 (Neutro)
            labels_unificadas.append(1)
        elif label >= 3: # 3 y 4 de amazon a 2 (Positivo)
            labels_unificadas.append(2)
    return np.array(labels_unificadas)

# Convertir etiquetas de Amazon
labels_numericos_entrenamiento_amazon_unificados = convertir_labels_amazon(labels_numericos_entrenamiento_amazon)
labels_numericos_validacion_amazon_unificados = convertir_labels_amazon(labels_numericos_validacion_amazon)
labels_numericos_prueba_amazon_unificados = convertir_labels_amazon(labels_numericos_prueba_amazon)

# Combinar datos de Tweet y Amazon
frases_entrenamiento_combinado = frases_entrenamiento_tweet + frases_entrenamiento_amazon
labels_numericos_entrenamiento_combinado = np.concatenate([labels_numericos_entrenamiento_tweet, labels_numericos_entrenamiento_amazon_unificados])

frases_validacion_combinado = frases_validacion_tweet + frases_validacion_amazon
labels_numericos_validacion_combinado = np.concatenate([labels_numericos_validacion_tweet, labels_numericos_validacion_amazon_unificados])

frases_prueba_combinado = frases_prueba_tweet + frases_prueba_amazon
labels_numericos_prueba_combinado = np.concatenate([labels_numericos_prueba_tweet, labels_numericos_prueba_amazon_unificados])

# Preprocesamiento (Tokenización y Padding) con datos combinados
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(frases_entrenamiento_combinado)
word_index = tokenizer.word_index
secuencias_entrenamiento_combinado = tokenizer.texts_to_sequences(frases_entrenamiento_combinado)
max_longitud = max([len(seq) for seq in secuencias_entrenamiento_combinado])
secuencias_padded_entrenamiento_combinado = pad_sequences(secuencias_entrenamiento_combinado, maxlen=max_longitud, padding='post', truncating='post')

vocab_size = len(word_index) + 1
embedding_dim = 32
lstm_units = 64

# Modelo Bidireccional LSTM con Dropout 
modelo_avanzado_combinado = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_longitud),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Bidirectional(LSTM(lstm_units)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

modelo_avanzado_combinado.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

modelo_avanzado_combinado.summary()

# Entrenamiento del Modelo con datos combinados
epocas = 5
historial_avanzado_combinado = modelo_avanzado_combinado.fit(secuencias_padded_entrenamiento_combinado, labels_numericos_entrenamiento_combinado, epochs=epocas, verbose=1, validation_split=0.2)

# Guardar el modelo y el Tokenizer
modelo_file_combinado = 'modelo_sentimiento_avanzado_combinado_dataset.h5'
tokenizer_file_combinado = 'tokenizer_combinado.pickle'

modelo_avanzado_combinado.save(modelo_file_combinado)

with open(tokenizer_file_combinado, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Modelo avanzado combinado guardado como {modelo_file_combinado}")
print(f"Tokenizer combinado guardado como {tokenizer_file_combinado}")