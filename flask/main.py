from flask import Flask, render_template, request
import os
import json
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

@app.route('/api/predict', methods=['POST'])
def predict():
    #Contoh data, bisa dalam List of String
    # data= ['saya senang sekali', 'saya benci kamu']
    if request.headers['Content-Type'] == 'application/json':
        try:
            requestBody = request.json
            data_cleaned =[]
            data_output = []
            for i in range(len(requestBody)):
                temp = requestBody[i].lower()
                temp = re.sub(r'[^\w\s]', '', temp)
                temp = re.sub("\d+", "", temp)
                data_cleaned.append(temp)
            
            encode = tokenizer.texts_to_sequences(data_cleaned)
            seq = pad_sequences(encode, maxlen=120, padding='post', truncating='post')

            predictions = (model.predict(seq) > 0.5).astype("int32")
            # [[0 1] [1 0]]
            good = "[0 1]"
            print(predictions)
            for x in predictions:
                print(x)
                if str(x) == good:
                    data_output.append("good")
                else:
                    data_output.append("not good")

            responseBody = {'message': data_output}
            return responseBody, 200
        except Exception as e:
            return {'error': str(e)}, 400
    else:
        return {'error': 'Invalid Content-Type'}, 400


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))