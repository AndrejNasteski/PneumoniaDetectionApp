# PneumoniaDetectionApp

Web App implementation of Pneumonia Detection CNN Model.
After creating an account and signing in, the user may upload a chest x-ray image.
After the image has being classified as a normal or pneumonic x-ray image by the model, the user has an option to confirm whether the model classified the image correctly or not.
(New data entries labeled by the users is optional).
The images are stored in a database, in addition to the model predicted class and the optional user label. Users with admin privileges may choose to re-train the model using the new data entries from the database.


** NOTE: Install Tensorflow and Keras in virtual environment before running.