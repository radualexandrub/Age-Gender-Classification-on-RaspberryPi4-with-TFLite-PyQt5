import tensorflow as tf # version 1.14
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from time import time

string_pred_age = ['04 - 06 ani', '07 - 08 ani','09 - 11 ani','12 - 19 ani','20 - 27 ani','28 - 35 ani','36 - 45 ani','46 - 60 ani','61 - 75 ani']
string_pred_gen = ['Feminin', 'Masculin']

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

interpreter_age = tf.lite.Interpreter(model_path="AgeClass_best_06_02-16-02.tflite")
interpreter_age.allocate_tensors()

interpreter_gender = tf.lite.Interpreter(model_path="GenderClass_06_03-20-08.tflite")
interpreter_gender.allocate_tensors()

# # Get input and output tensors
input_details_age = interpreter_age.get_input_details()
output_details_age = interpreter_age.get_output_details()
input_shape_age = input_details_age[0]['shape']

input_details_gender = interpreter_gender.get_input_details()
output_details_gender = interpreter_gender.get_output_details()
input_shape_gender = input_details_gender[0]['shape']

input_im = None

webcam = cv2.VideoCapture(0)
# Setare rezolutie camera Raspberry (320x240, 640x480, 1280x720, 1920x1080-nerecomandat)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    font = cv2.FONT_HERSHEY_PLAIN
    time_start = time()
    _, frame = webcam.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)
    for x,y,w,h in faces:
        saved_image = frame          
        input_im = saved_image[y:y+h, x:x+w]
        
        if input_im is None:
            print("Nu a fost detectata nicio fata")
        else:
            input_im = cv2.resize(input_im, (224,224))
            input_im = input_im.astype('float')
            input_im = input_im / 255
            input_im = img_to_array(input_im)
            input_im = np.expand_dims(input_im, axis = 0)

            # Predict
            input_data = np.array(input_im, dtype=np.float32)
            interpreter_age.set_tensor(input_details_age[0]['index'], input_data)
            interpreter_age.invoke()
            interpreter_gender.set_tensor(input_details_gender[0]['index'], input_data)
            interpreter_gender.invoke()

            output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
            output_data_gender = interpreter_gender.get_tensor(output_details_gender[0]['index'])
            index_pred_age = int(np.argmax(output_data_age))
            index_pred_gender = int(np.argmax(output_data_gender))
            prezic_age = string_pred_age[index_pred_age]
            prezic_gender = string_pred_gen[index_pred_gender]

            cv2.putText(frame, prezic_age + ', ' + prezic_gender, (x,y), font, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

    fps = "FPS: " + str(round(1.0 / (time() - time_start), 2))      
    cv2.putText(frame, fps, (20,20), font, 1, (250,250,250), 1, cv2.LINE_AA)        
    cv2.imshow("Detecting faces...", frame)
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
