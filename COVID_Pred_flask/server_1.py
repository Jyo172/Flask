import os

from numpy import stack
from imageio import imread
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from flask import (Flask, flash, render_template, redirect, request, session,
                   send_file, url_for)
from werkzeug.utils import secure_filename

from utils_1 import (is_allowed_file, generate_barplot, generate_random_name,
                   make_thumbnail,generate_csv,generate_pred_csv)
import pandas as pd

app = Flask(__name__)
#app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
#app.config['UPLOAD_FOLDER'] = os.environ['UPLOAD_FOLDER']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home_1.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)
        
        features = [x for x in request.form.values()]
        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # check if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            filename = secure_filename(generate_random_name(image_file.filename))
            filepath = os.path.join('Uploads/', filename)
            image_file.save(filepath)
            # HACK: Defer this to celery, might take time
            passed = make_thumbnail(filepath)
            if passed:
                features.append(filename)
                generate_csv(features)
                return redirect(url_for('predict', filename=filename))
            else:
                return redirect(request.url)


@app.errorhandler(500)
def server_error(error):
    """ Server error page handler """
    return render_template('error.html'), 500


@app.route('/images/<filename>')
def images(filename):
    """ Route for serving uploaded images """
    if is_allowed_file(filename):
        return send_file(os.path.join('Uploads/', filename))
    else:
        flash("File extension not allowed.")
        return redirect(url_for('home'))


@app.route('/predict/<filename>')
def predict(filename):
    """ After uploading the image, show the prediction of the uploaded image
    in barchart form
    """
    image_path = os.path.join('Uploads/', filename)
    image_url = url_for('images', filename=filename)
    
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join('models/COVIDNet-CXR-Large', 'model.meta'))
    saver.restore(sess, os.path.join('models/COVIDNet-CXR-Large', 'model-8485'))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name("input_1:0")
    pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
    
    x = cv2.imread(image_path)
    h, w, c = x.shape
    x = x[int(h/6):, :]
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
    output = inv_mapping[pred.argmax(axis=1)[0]]
    generate_pred_csv([filename,output])

    return render_template(
        'predict.html',
        prediction_text = 'Prediction : {}'.format(output),
        image_url=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)