from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import skimage            
from random import shuffle
from tqdm import tqdm  
import scipy
from skimage.transform import resize

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/weights.hdf5'

# Load your trained model
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = load_model(MODEL_PATH, custom_objects={'swish_activation': swish_activation})
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')




def model_predict(img_path, model):
    imgs = []
    img = cv2.imread(img_path)
    #img = image.img_to_array(img)
    img = skimage.transform.resize(img, (150, 150, 3))
    img = np.asarray(img)
    imgs.append(img)
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(1,3,150,150)
    preds = model.predict(imgs)
    prob = model.predict_proba(imgs)
    class_ = model.predict_classes(imgs)
    return preds, prob, class_






@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uphttp://localhost:5000/loads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds, prob, class_= model_predict(file_path, model)
	

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred = np.argmax(preds,axis = 0) 
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        if class_[0] == 0:
            result = "Normal"
        else:
            result = "PNEUMONIA"

        #result = str(prob)
        #result = 'preds:   ' + str(preds) + '    proba:   ' + str(prob) + '    class:  ' + str(class_)         # Convert to string
        return result
    return None


if __name__ == '__main__':
    #app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
