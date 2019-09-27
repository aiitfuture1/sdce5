###### Part 1 (Import libraries) ######
# Matrix math
import numpy as np
# Preprocess image
import cv2
# Load our saved model
from keras.models import load_model
# Real time server
import socketio
# Concurrent networking & web server gateway interface
import eventlet
# Web framework
from flask import Flask
# Decoding camera images
import base64
# Input output
from io import BytesIO
# Image manipulation
from PIL import Image

###### Part 2 (Initialize the server and the web app) ######
# Initialize our server
socket = socketio.Server()
# Our flask web app
app = Flask(__name__) 


###### Part 3 (Preprocess function) ######
def preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

###### Part 4 (Send control to the car) ######
# Send steering angle and throttle to the car
def send_control(steering_angle, throttle):
    socket.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

###### Part 5 (Connect) ######
@socket.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

###### Part 6 (Predict steering angle and calculate throttle) ######
# Car speed limit
speed_limit = 10
@socket.on('telemetry')
def telemetry(sid, data):
	# The current speed of the car
    speed = float(data['speed'])
    # The current image from the center camera of the car
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # Convert from PIL image to numpy array
    image = np.asarray(image)
    # Apply the preprocessing function
    image = preprocess(image)
    # The model expects 4D array
    image = np.array([image])
    # Predict the steering angle for the image
    steering_angle = float(model.predict(image))
    # Calculate throttle
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

###### Part 7 (Load and deploy) ######
# Load model and deploy to the server
if __name__ == '__main__':
	# Load our saved model
    model = load_model('model.h5')
    # Wrap Flask application with socket
    app = socketio.Middleware(socket, app)
    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)