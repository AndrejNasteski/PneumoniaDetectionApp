from flask import Flask, flash, request, redirect, url_for, Response, render_template, session
from flaskr.auth import login_required
from io import BytesIO
import os
import sqlite3
import base64
import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from werkzeug.security import generate_password_hash

from . import db
from . import auth
 

UPLOAD_FOLDER = r"flaskr\temp\uploads"
FILE_EXTENSIONS = ["jpg","jpeg","png"]
IMAGE_SIZE = 150
DISPLAY_IMAGE_SIZE_X = 1024
DISPLAY_IMAGE_SIZE_Y = 576 
MODEL_THRESHOLD = 0.93  # highest model accuracy
MODEL_PATH = r"flaskr\temp\model"
VAL_PATH = r"flaskr\temp\val"

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)
    app.register_blueprint(auth.bp)


    @app.route('/')
    def home():
        return render_template('index.html')
    

    @app.route('/upload', methods=['POST','GET'])
    @login_required
    def upload():
        if request.method == 'GET':
            session['prediction'] = None
            return render_template('upload.html')
        elif request.method == 'POST':
            session['prediction'] = None
            file_obj = request.files['file']
            if file_obj.filename == "":
                flash("No image selected for upload.")
                return redirect(url_for("upload"))
            user_id = session.get('user_id')
            if user_id is None:
                return redirect(url_for('auth.login'))
            uploaded_file_extension = file_obj.filename.split(".")[1]
            if (uploaded_file_extension.lower() in FILE_EXTENSIONS):
                if uploaded_file_extension.lower() == 'jpg':
                    file_obj.filename = file_obj.filename.split(".")[0] + ".jpeg"
                    uploaded_file_extension = 'jpeg'
                
                data = file_obj.read()
                file_bytes = np.frombuffer(BytesIO(data).getvalue(), dtype=np.uint8)
                img2 = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.resize(img2, (DISPLAY_IMAGE_SIZE_X, DISPLAY_IMAGE_SIZE_Y))
                blob = cv2.imencode('.' + uploaded_file_extension.lower(), img2)[1].tobytes()
                try:
                    conn = db.get_db()
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO images (author_id, image_data) VALUES (?, ?) ",
                                    (user_id, blob))
                    image_db_id = cursor.execute("SELECT id FROM images ORDER BY id DESC LIMIT 1"
                                                 ).fetchone()
                    session['image_db_id'] = image_db_id[0]
                    conn.commit()
                    conn.close()
                except sqlite3.Error as e:
                    flash(f"{e}")
                    return render_template('upload.html')
            else:
                flash('Only images accepted!')
                return render_template('upload.html')
        return redirect(url_for("preview_image"))

    
    @app.route('/preview', methods=['GET'])
    @login_required
    def preview_image():
        if request.method == 'GET':
            image_db_id = session.get('image_db_id')
            conn = db.get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_db_id,))
            record = cursor.fetchone()
            conn.commit()
            conn.close()
            image = record[3]
            dataurl = 'data:image/png;base64,' + base64.b64encode(BytesIO(image).getvalue()).decode('ascii')
            return render_template('preview.html', response = dataurl)
        return redirect(url_for("home"))
    

    @app.route('/classify') 
    def classify(): # x -> 1 == NORMAL, x -> 0 == PNEUMONIA
        model = keras.models.load_model(MODEL_PATH)

        image_db_id = session.get('image_db_id')
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_db_id,))
        record = cursor.fetchone()
        image = record[3]
        classified = record[4]
        file_bytes = np.frombuffer(BytesIO(image).getvalue(), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.array(image) / 255

        image_array = image_array[:,:,0]
        image_array = image_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        prediction = model.predict(image_array)     # prediction[0][0]
        
        if prediction <= 0.5:
            prediction_text = "PNEUMONIA"
        elif prediction > 0.5:
            prediction_text = "NORMAL"

        cursor.execute("UPDATE images SET model_label = ? WHERE id = ?", (prediction_text, image_db_id))
        conn.commit()
        conn.close()

        if classified is None:
            session['prediction'] = prediction_text
        return redirect(url_for("preview_image"))
    

    @app.route('/discard')
    def discard():
        session['prediction'] = None
        image_db_id = session['image_db_id']
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (image_db_id,))
        conn.commit()
        conn.close()
        flash('Image discarded.')
        session.pop("image_db_id")
        return render_template('upload.html')
    

    @app.route('/update_label_correct')
    def update_label_correct():
        image_db_id = session.get('image_db_id')
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_db_id,))
        record = cursor.fetchone()
        if record[4] is None:
            flash("Error while updating image class.")
            return render_template('upload.html')

        cursor.execute("UPDATE images SET user_label = ? WHERE id = ?", ("CORRECT", image_db_id))
        conn.commit()
        conn.close()
        flash("Image class updated.")
        return render_template('upload.html')
    

    @app.route('/update_label_incorrect')
    def update_label_incorrect():
        image_db_id = session.get('image_db_id')
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_db_id,))
        record = cursor.fetchone()
        if record[4] is None:
            flash("Error while updating image class.")
            return render_template('upload.html')
        
        cursor.execute("UPDATE images SET user_label = ? WHERE id = ?", ("INCORRECT", image_db_id))
        conn.commit()
        conn.close()
        flash("Image class updated.")
        return render_template('upload.html')


    @app.route('/retrain')
    def retrain():
        x_val = []
        y_val = []
        for i in range(1,8):
            p = os.path.join(VAL_PATH, r"NORMAL\NORMAL (" + str(i) +").jpeg")
            img_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
            x_val.append(resized_arr)
            y_val.append(1)         # 0 - PNEUMONIA, 1 - NORMAL

        for i in range(1,8): 
            p = os.path.join(VAL_PATH, r"PNEUMONIA\PNEUMONIA (" + str(i) +").jpeg")
            img_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
            x_val.append(resized_arr)
            y_val.append(0)         # 0 - PNEUMONIA, 1 - NORMAL

        x_val = np.array(x_val) / 255
        x_val = x_val.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        y_val = np.array(y_val)

        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images")
        record = cursor.fetchall()
        data = []
        labels = []
        for row in record:
            image = row[3]
            file_bytes = np.frombuffer(BytesIO(image).getvalue(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image_array = np.array(image) / 255
            image_array = image_array[:,:,0]
            
            if row[4] is not None and row[5] is None:      # model classified image
                if row[4] == "NORMAL":
                    labels.append(1)
                elif row[4] == "PNEUMONIA":
                    labels.append(0)
                elif row[4] == None:
                    continue
                else:
                    flash("Error in database labels")
                    return render_template('upload.html')
                data.append(image_array)
            elif row[4] is not None:                   # user classified image
                if row[5] == "CORRECT":
                    if row[4] == "NORMAL":
                        labels.append(1)
                    elif row[4] == "PNEUMONIA":
                        labels.append(0)
                elif row[5] == "INCORRECT":
                    if row[4] == "NORMAL":
                        labels.append(0)            # reversed
                    elif row[4] == "PNEUMONIA":
                        labels.append(1)            # reversed
                elif row[5] == None:
                    continue
                else:
                    flash("Error in database labels")
                    return render_template('upload.html')
                data.append(image_array)

        data = np.array(data)
        data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        datagen.fit(data)

        model = keras.models.load_model(MODEL_PATH)
        metrics = ['accuracy',
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall')]
        model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = metrics)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

        model.fit(datagen.flow(data, labels, batch_size = 8),
                    epochs = 10,
                    validation_data = (x_val, y_val),
                    callbacks = [learning_rate_reduction])
        
        results = model.evaluate(x_val, y_val)
        if results[1] > MODEL_THRESHOLD:
            model.save(r"flaskr\temp\Saved_model")
            print("Model results:" , results)
            flash("Model re-trained on new data.")
        else:
            flash("Re-trained model discarded.")
        return render_template('index.html')

    return app