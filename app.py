# using flask_restful
from flask import Flask, jsonify, request
import os
import numpy as np
import torch
import tensorflow as tf
import numpy as numpy

from PIL import Image
# from werkzeug import secure_filename
from flask import Flask, flash, request, make_response, render_template, jsonify, request
from flask_restful import Resource, Api
from preprocessing import normalize_image
from inference import extract_model
from postprocessing import predict


UPLOAD_FOLDER = 'uploads'
app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Your Route


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        image_file = Image.open(
            "uploads/{}".format(name))  # open colour image
        # convert image to black and white
        image_file = image_file.convert('1')
        image_file.save('uploads/{}'.format(name))
        
        with open('uploads/{}'.format(name), 'r+b') as f:
            with Image.open(f) as img:
                path = Image.open('uploads/{}'.format(name)).convert(mode="L")
                img = normalize_image(path)
        
        #model
        model = extract_model()
        
        #labels of images
        label_name = {
            0: "T-shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat", 
            5: "Sandal", 
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
            }
        result = predict(model, img)
        return f"{label_name[result]}"
    
    else:
        return render_template("Index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)