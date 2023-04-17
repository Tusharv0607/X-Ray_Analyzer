import os
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import pad_sequences
import pickle

from flask import Flask, render_template, request, jsonify, abort

app = Flask(__name__)

# Constants
UPLOADS_PATH = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_LENGTH = 170

# Load the tokenizer and texts
tokenizer = Tokenizer()
texts = pickle.load(open('tokens.pkl', 'rb'))
tokenizer.fit_on_texts(texts)

# Load the models
model = load_model('best_model.h5')
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def predict_caption(image_path):
    # Process the image for prediction
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, *image.shape))
    image = preprocess_input(image)

    # Extract features using VGG16 model
    feature = vgg_model.predict(image, verbose=0)

    # Generate caption
    in_text = 'startseq'
    for i in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        yhat = np.argmax(model.predict([feature, sequence], verbose=0))
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word

    # Return the caption
    return " ".join(in_text.split(" ")[1:-1])

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Endpoint for home page
@app.route('/')
def home():
    return render_template('./index.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'image' not in request.files:
        abort(400, {'error': 'No file uploaded'})
    image = request.files['image']

    # Check if the file is an allowed type/extension
    if image.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        abort(400, {'error': 'Invalid file type'})

    # Save the file to the uploads folder
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    save_path = os.path.join(UPLOADS_PATH, image.filename)
    image.save(save_path)

    try:
        caption = predict_caption(save_path)
    except Exception as e:
        os.remove(save_path)
        abort(500, {'error': str(e)})

    # Remove the uploaded file
    os.remove(save_path)

    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
