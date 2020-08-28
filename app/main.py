from flask import Flask, request, jsonify

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

import imdb 
import os, sys, gc
import requests
import io
   
IMG_SIZE = 224
CHANNELS = 3

app = Flask(__name__)


@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn (default)".format(
        version
    )
    return message

@app.route('/predict', methods=['GET'])
def show_prediction():
    movie_id = request.args.get("movieId", None)

    print(f"got movie_id {movie_id}")

    model = tf.keras.models.load_model("bce_model_20200826.h5",compile=False,custom_objects={'KerasLayer':hub.KerasLayer})

    # creating instance of IMDb 
    ia = imdb.IMDb() 
    movie = ia.get_movie(movie_id[2:])

    save_path = download_image(movie['full-size cover url'],'temp', movie_id)
    img_path = save_path

    # Read and prepare image
    img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    # Generate prediction
    prediction = (model.predict(img) > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction = prediction[prediction==1].index.values

    # print("predict genre ="+str(list(prediction)))

    predict_labels = [
        'Action',
        'Adventure',
        'Animation',
        'Biography',
        'Comedy',
        'Crime',
        'Drama',
        'Family',
        'Fantasy',
        'Game-Show',
        'History',
        'Horror',
        'Music',
        'Musical',
        'Mystery',
        'News',
        'Romance',
        'Sci-Fi',
        'Sport',
        'Thriller',
        'War',
        'Western',
    ]
    response = {}
    
    response['title'] = movie['title']
    response['genres'] = movie['genres']
    response['predict_genres'] = [predict_labels[p] for p in prediction]

    response["img_path"] = f"{img_path}"
    response["MESSAGE"] = f"movie_id: {movie_id}"

    # Return the response in json format
    return jsonify(response)

def download_image(url, image_file_path='temp', filename=False):
    isdir = os.path.isdir(image_file_path)
    #check if folder exist
    if not isdir:
        #create directory
        os.makedirs(image_file_path)

    r = requests.get(url, stream = True, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    file_extension = url.split("/")[-1].split(".")[-1]
    if not filename:
      filename = url.split("/")[-1].split(".")[0]

    save_path = image_file_path+"/"+filename+"."+file_extension

    if not os.path.isfile(save_path):
      with Image.open(io.BytesIO(r.content)) as im:
          im.save(save_path)

    print('Image downloaded from url: {} \nsaved to: {}'.format(url, save_path))
    return save_path

