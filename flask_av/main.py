from flask import Flask, redirect, request, jsonify
from keras import models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
import os

app = Flask(__name__)
model = None



@app.route('/')
def index():
    return redirect('/static/index.html')


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        #print(name)
        #name = name.encode("UTF-8")
        #name = name.encode("Shift_JIS")


        font = ImageFont.truetype(
            "/Users/ishikawaryuuichi/Library/ヒラギノ角ゴシック W4.ttc",
                          10)
        print(draw.textsize(name))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255),font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    # img = Image.open("knn_examples/train/"+name+"/"+name+".jpg")
    # img.show()
    #pil_image.show()
    pil_image.save("new.jpg")

@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        print("test")
        with open("trained_knn_model100.clf", 'rb') as f:
            knn_clf = pickle.load(f)
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg','JPEG', quality=100, optimize=True)
        print(img)


        X_img = face_recognition.load_image_file('test.jpg')
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)


        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=10)
        are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(X_face_locations))]

        pred = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

        print(pred[0][0])
        data = dict(pred=pred[0][0], confidence="wow")

        show_prediction_labels_on_image(os.path.join("test.jpg"), pred)


        # Predict classes and remove classifications that aren't within the threshold
        return jsonify(data)
        # img = np.asarray(img) / 255.
        # img = np.expand_dims(img, axis=0)
        # pred = model.predict(img)

        # players = [
        #     'Lebron James',
        #     'Stephen Curry',
        #     'Kevin Durant',
        # ]

        confidence = str(round(max(pred[0]), 3))
        pred = players[np.argmax(pred)]

        data = dict(pred=pred, confidence=confidence)
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('new.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    #os.remove('new.jpg')
    #load_model()
    # model._make_predict_function()
    app.run(debug=False, port=5000)
