from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import contextlib

app = Flask(__name__)

# Load the model
model = load_model('finetuned_butterflies_vgg16_model_ft.h5')

# Define the list of class names
class_names = ["Danaus_Plexippus", "Heliconius_Charitonius", "Heliconius_Erato", "Junonia_Coenia", "Lycaena_Phlaeas", "Lycaena_Phlaeas", "Papilio_Cresphontes", "Pieris_Rapae", "Vanessa_Atalanta", "Vanessa_Atalanta"]

# Function to predict and visualize single image
def predict_and_visualize_single_image(image_path, model, class_names):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    probability = prediction[0][predicted_index]
    return predicted_class, probability, image[0]

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join('static', secure_filename(f.filename))
        f.save(filepath)
        predicted_class, probability, predicted_image = predict_and_visualize_single_image(filepath, model, class_names)
        # Remove the uploaded image after prediction
        os.remove(filepath)
        # Save the predicted image
        output_path = os.path.join('static', 'output.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(predicted_image * 255, cv2.COLOR_RGB2BGR))
        prediction_text = f'Predicted butterfly class is "{predicted_class}" with probability {probability:.4f}'
        image_path = url_for('static', filename='output.jpg')
        return render_template('index.html', prediction_text=prediction_text, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)