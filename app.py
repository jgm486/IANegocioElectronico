from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle 

app = Flask(__name__)

# Cargar el modelo avanzado entrenado
modelo_cargado_avanzado = load_model('modelo_sentimiento_avanzado_combinado_dataset.h5')

# Cargar el Tokenizer desde el archivo pickle
tokenizer_file = 'tokenizer_combinado.pickle'
with open(tokenizer_file, 'rb') as handle:
    tokenizer_cargado_avanzado = pickle.load(handle)

max_longitud_cargado_avanzado = 28 
sentimiento_indices_inverso = {0: "Negativo", 1: "Neutro", 2: "Positivo"}


def preprocesar_texto_avanzado(texto):
    secuencia = tokenizer_cargado_avanzado.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(secuencia, maxlen=max_longitud_cargado_avanzado, padding='post', truncating='post')
    return secuencia_padded

# Rutas
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predecir_ui', methods=['POST'])
def predecir_ui():
    frase = request.form['frase']
    texto_preprocesado = preprocesar_texto_avanzado(frase)
    prediccion = modelo_cargado_avanzado.predict(texto_preprocesado)
    indice_sentimiento = np.argmax(prediccion)
    sentimiento_predicho = sentimiento_indices_inverso[indice_sentimiento]
    probabilidad_sentimiento = prediccion[0][indice_sentimiento]
    probabilidad_porcentaje = "{:.2f}%".format(probabilidad_sentimiento * 100)

    return render_template('index.html', frase_analizada=frase, sentimiento_predicho=sentimiento_predicho, probabilidad=probabilidad_porcentaje)

if __name__ == '__main__':
    app.run(debug=True)