FRONTEND PROGRAM

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
%matplotlib inline
# load model
model = load_model("/content/best_model (1).h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haar
cascade_frontalface_default.xml')
# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
 """
 Params:
 js_reply: JavaScript object containing image from webcam
 Returns:
 img: OpenCV BGR image
 """
 # decode base64 image
 image_bytes = b64decode(js_reply.split(',')[1])
 # convert bytes to numpy array
 jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
 # decode numpy array into OpenCV BGR image
 img = cv2.imdecode(jpg_as_np, flags=1)
 return img
def bbox_to_bytes(bbox_array):
 """
 Params:
 bbox_array: Numpy array (pixels) containing rectangle to over
lay on video stream.
 Returns:
 bytes: Base64 image byte string
 """
 # convert array into PIL image
 bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
 iobuf = io.BytesIO()
 # format bbox into png for return
 bbox_PIL.save(iobuf, format='png')
 # format return string
 bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.g
etvalue()), 'utf-8')))
 return bbox_bytes
# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
count = 0
while True:
 js_reply = video_frame(label_html, bbox)
 if not js_reply:
 break
 test_img = js_to_image(js_reply["img"])
 gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
 bbox_array = np.zeros([480,640,4], dtype=np.uint8)
 faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32,
5)
 for (x, y, w, h) in faces_detected:
 bbox_array = cv2.rectangle(bbox_array, (x, y), (x + w, y + h),
(255, 0, 0), thickness=7)
 roi_gray = gray_img[y:y + w, x:x + h] # cropping region of int
erest i.e. face area from image
 roi_gray = cv2.resize(roi_gray, (224, 224))
 img_pixels = image.img_to_array(roi_gray)
 img_pixels = np.expand_dims(img_pixels, axis=0)
 img_pixels /= 255
 predictions = model.predict(img_pixels)
 # find max indexed array
 max_index = np.argmax(predictions[0])
 emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surpri
se', 'neutral')
 predicted_emotion = emotions[max_index]
 bbox_array = cv2.putText(bbox_array, predicted_emotion, (int(x)
, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 2
55
 # convert overlay of bbox into bytes
 bbox_bytes = bbox_to_bytes(bbox_array)
 # update bbox so next frame gets new overlay
 bbox = bbox_bytes