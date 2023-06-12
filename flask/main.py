from flask import Flask, render_template, request

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import joblib


app = Flask(__name__)
model = tf.keras.models.load_model('model_fix.h5')

tokenizer = joblib.load('token.pkl')

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    #Contoh data, bisa dalam List of String
    data= ['saya senang sekali', 'saya benci kamu','jelek']
    data_cleaned =[]
    for i in range(len(data)):
        temp = data[i].lower()
        temp = re.sub(r'[^\w\s]', '', temp)
        temp = re.sub("\d+", "", temp)
        data_cleaned.append(temp)
    
    encode = tokenizer.texts_to_sequences(data_cleaned)
    seq = pad_sequences(encode, maxlen=120, padding='post', truncating='post')


    predictions = (model.predict(seq) > 0.5).astype("int32")
    return render_template("index.html", predicts = predictions)


if __name__ == '__main__':
    app.run(port=3000, debug=True)