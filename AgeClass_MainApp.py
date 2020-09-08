import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QPixmap, QImage

import tensorflow as tf # version 1.15.0
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

class PicButton(QAbstractButton):
    # https://stackoverflow.com/questions/2711033/how-code-a-image-button-in-pyqt
    def __init__(self, pixmap, pixmap_hover, pixmap_pressed, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap
        self.pixmap_hover = pixmap_hover
        self.pixmap_pressed = pixmap_pressed

        self.pressed.connect(self.update)
        self.released.connect(self.update)

    def paintEvent(self, event):
        pix = self.pixmap_hover if self.underMouse() else self.pixmap
        if self.isDown():
            pix = self.pixmap_pressed

        painter = QPainter(self)
        painter.drawPixmap(event.rect(), pix)

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()

    def sizeHint(self):
        return QSize(200, 200)


class UI_MainWindow(QWidget):
    def __init__(self):
        super(UI_MainWindow, self).__init__()

        self.title = "Age Classification App"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.UI_MainWindow_InitSetup()

    def UI_MainWindow_InitSetup(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: black;")

        MainWindow_Layout = QHBoxLayout()

        # ## Adaugare Buton1 pentru Camera Live
        self.Btn_Camera = PicButton(QPixmap("./Icons/Btn_Camera.jpg"), QPixmap(
            "./Icons/Btn_Camera_hover.jpg"), QPixmap("./Icons/Btn_Camera_pressed.jpg"))
        self.Btn_Camera.clicked.connect(self.Open_UI_Camera)
        MainWindow_Layout.addWidget(self.Btn_Camera)

        # ## Adauga Spacer intre 2 butoane
        spacerItem = QSpacerItem(
            40, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)
        MainWindow_Layout.addItem(spacerItem)

        # ## Adaugare Buton2 pentru Selectare 1 Imagine
        self.Btn_SelectPic = PicButton(QPixmap("./Icons/Btn_SelectPic.jpg"), QPixmap(
            "./Icons/Btn_SelectPic_hover.jpg"), QPixmap("./Icons/Btn_SelectPic_pressed.jpg"))
        self.Btn_SelectPic.clicked.connect(self.Open_UI_SelectPic)
        MainWindow_Layout.addWidget(self.Btn_SelectPic)

        # ## Adauga Spacer intre 2 butoane
        spacerItem = QSpacerItem(
            40, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)
        MainWindow_Layout.addItem(spacerItem)

        # ## Adaugare Buton3 pentru Selectare Galerie Imagini
        self.Btn_SelectGallery = PicButton(QPixmap("./Icons/Btn_SelectGallery.jpg"), QPixmap(
            "./Icons/Btn_SelectGallery_hover.jpg"), QPixmap("./Icons/Btn_SelectGallery_pressed.jpg"))
        self.Btn_SelectGallery.clicked.connect(self.Open_UI_SelectGallery)
        MainWindow_Layout.addWidget(self.Btn_SelectGallery)

        self.setLayout(MainWindow_Layout)
        self.show()

    def Open_UI_Camera(self):
        self.ui = UI_CameraWindow()
        self.ui.show()

    def Open_UI_SelectPic(self):
        self.ui = UI_SelectPic()
        self.ui.show()

    def Open_UI_SelectGallery(self):
        self.ui = UI_SelectGallery()
        self.ui.show()


class UI_CameraWindow(QWidget):
    def __init__(self):
        # ## Partea de UI Setup
        super(UI_CameraWindow, self).__init__()

        self.setObjectName("Camera_Window")
        self.resize(700, 480)

        CameraWindow_Layout = QVBoxLayout()

        # Adauga buton deschidere camera web
        self.Btn_camera_on_off = QPushButton(self)
        self.Btn_camera_on_off.setStyleSheet("background-color: white")
        CameraWindow_Layout.addWidget(self.Btn_camera_on_off)

        # Adauga placeholder pentru a afisa continutul webcam
        self.PlaceHolder_CameraLabel = QLabel()
        CameraWindow_Layout.addWidget(self.PlaceHolder_CameraLabel)

        # Adauga placeholder pentru a afisa FPS (numar cadre pe secunda)
        self.PlaceHolder_FPS = QLabel()
        self.PlaceHolder_FPS.setStyleSheet("color: white")
        self.PlaceHolder_FPS.resize(20,100)
        CameraWindow_Layout.addWidget(self.PlaceHolder_FPS)

        self.setLayout(CameraWindow_Layout)

        self.retranslateUI()

    def retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate(
            "App", "Age Classification: Live Webcam"))
        self.setStyleSheet("background-color: black;")

        self.PlaceHolder_CameraLabel.setText(
            _translate("App", ""))
        self.Btn_camera_on_off.setText(_translate("App", "Start Camera"))

        # ## Partea de OpenCV Camera
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.Webcam_update)
        # set Btn_camera_on_off callback clicked  function
        self.Btn_camera_on_off.clicked.connect(self.controlTimer)

    def Webcam_update(self):
        time_start = time()

        _, frame = self.webcam.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)

        for x, y, w, h in faces:
            input_im = frame[y:y+h, x:x+w]

            if input_im is not None:
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

                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, prezic_age + ', ' + prezic_gender, (x,y), font, 1, (255,255,255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

        # Frame info for PyQT
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, frame_channel = frame.shape
        bytesPerLine = frame_channel * frame_width

        # create QImage from frame
        qImg = QImage(frame.data, frame_width, frame_height,
                      bytesPerLine, QImage.Format_RGB888)
        # show frame in img_label
        self.PlaceHolder_CameraLabel.setPixmap(QPixmap.fromImage(qImg))

        fps = "FPS: " + str(round(1.0 / (time() - time_start), 2))
        self.PlaceHolder_FPS.setText(fps)

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.webcam = cv2.VideoCapture(0)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 640x480, 320x240
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            #self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            #self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            # start timer
            self.timer.start(1)
            # update Btn_camera_on_off text
            self.Btn_camera_on_off.setText("Stop Camera")

        # if timer just started
        else:
            # stop timer
            self.timer.stop()
            # release video webcam
            self.webcam.release()
            # update Btn_camera_on_off text
            self.Btn_camera_on_off.setText("Start")
            

class UI_SelectPic(QWidget):
    def __init__(self):
        # ## Partea de UI Setup
        super(UI_SelectPic, self).__init__()

        self.setObjectName("SelectPic_Window")
        self.setWindowTitle("Age Classification: Select an Image")
        self.setStyleSheet("background-color: black;")
        self.resize(400, 400)

        SelectPic_Layout = QVBoxLayout()

        # Adauga buton Browse for image
        self.Btn_BrowseImage = QPushButton(self)
        self.Btn_BrowseImage.setStyleSheet("background-color: white")
        self.Btn_BrowseImage.setText("Browse an image!")
        self.Btn_BrowseImage.clicked.connect(self.browseImage)
        SelectPic_Layout.addWidget(self.Btn_BrowseImage)

        # Adauga placeholder pentru a afisa imaginea
        self.PlaceHolder_Pic = QLabel()
        SelectPic_Layout.addWidget(self.PlaceHolder_Pic)

        # Adauga placeholder label pentru a afisa probabilitatile
        self.PlaceHolder_Outputs = QLabel()
        self.PlaceHolder_Outputs.adjustSize()
        self.PlaceHolder_Outputs.setStyleSheet("color: white")
        SelectPic_Layout.addWidget(self.PlaceHolder_Outputs)

        self.setLayout(SelectPic_Layout)

    def browseImage(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, 'Open an Image', '/home/pi/Desktop/Documents', 'Image files (*.jpg *.bmp *.png)')

        # ## Procesare imagine, clasificare
        pixmap, outputs_str = classifyImage(filePath)

        # ## Display image
        if pixmap.width() > 500 or pixmap.height() > 500:
            pixmap = pixmap.scaled(
                500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.PlaceHolder_Pic.setPixmap(QPixmap(pixmap))
        self.PlaceHolder_Outputs.setText(outputs_str)


class UI_SelectGallery(QWidget):
    def __init__(self):
        # ## Partea de UI Setup
        super(UI_SelectGallery, self).__init__()

        self.setObjectName("SelectGallery_Window")
        self.setWindowTitle("Age Classification: Select a Folder with Images")
        self.setStyleSheet("background-color: black;")
        self.resize(400, 400)

        self.folderPath = None
        self.imageNames = []
        self.imageNames_gen = None
        self.image_nr = 1

        SelectPic_Layout = QVBoxLayout()

        # Adauga buton Browse for image
        self.Btn_SelectFolder = QPushButton(self)
        self.Btn_SelectFolder.setStyleSheet("background-color: white")
        self.Btn_SelectFolder.setText("Select a folder with images!")
        self.Btn_SelectFolder.clicked.connect(self.GetFolderPath)
        SelectPic_Layout.addWidget(self.Btn_SelectFolder)

        # Adauga placeholder pentru a afisa imaginea
        self.PlaceHolder_Pic2 = QLabel()
        SelectPic_Layout.addWidget(self.PlaceHolder_Pic2)

        # Adauga buton pentru afisare Next Image
        self.Btn_NextImage = QPushButton(self)
        self.Btn_NextImage.setStyleSheet("background-color: white")
        self.Btn_NextImage.setText("Next image")
        self.Btn_NextImage.clicked.connect(self.NextImage)
        SelectPic_Layout.addWidget(self.Btn_NextImage)

        # Adauga label pentru afisare nr imagine
        self.PlaceHolder_Label = QLabel(
            "Momentan niciun folder nu este selectat!")
        self.PlaceHolder_Label.setStyleSheet("color:white")
        self.PlaceHolder_Label.adjustSize()
        SelectPic_Layout.addWidget(self.PlaceHolder_Label)

        self.setLayout(SelectPic_Layout)

    def GetFolderPath(self):
        self.image_nr = 1
        self.imageNames = []
        self.folderPath = QFileDialog.getExistingDirectory(
            self, "Select Directory", '/home/pi/Desktop/Documents')

        class Found(Exception):
            pass
        try:
            for dirname, dirnames, filenames in os.walk(self.folderPath):
                for filename in filenames:
                    # Verifica daca folderul contine imagini
                    if filename[-4:] not in ['.jpg', '.png', '.bmp']:
                        raise Found
                    self.imageNames.append(filename)
        except Found:
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setText("Eroare:")
            error_msg.setInformativeText(
                'Directoriul nu contine imagini!')
            error_msg.setWindowTitle("Error")
            error_msg.exec_()

        # print(self.imageNames)
        self.imageNames_gen = (_ for _ in self.imageNames)  # Creez generator

    def NextImage(self):
        if self.imageNames_gen is not None:
            self.imageName = next(self.imageNames_gen, None)
        else:
            self.PlaceHolder_Label.setText(
                "Momentan niciun folder nu este selectat!!!".upper())
            return

        # Aici depinde de sistemul de operare, Windows (\), Linux (/), deci normalizam path-ul
        if self.imageName is not None:
            full_path = os.path.join(os.path.normpath(
                self.folderPath), self.imageName)

            # ## Procesare imagine, clasificare
            pixmap, outputs_str = classifyImage(full_path)

            # ## Display image
            if pixmap.width() > 500 or pixmap.height() > 500:
                pixmap = pixmap.scaled(
                    500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.PlaceHolder_Pic2.setPixmap(QPixmap(pixmap))

            self.PlaceHolder_Label.setText(
                "Imaginea {} din {}\n{}".format(self.image_nr, len(self.imageNames), outputs_str))
            self.image_nr += 1
        else:
            self.PlaceHolder_Label.setText(
                "S-au afisat toate cele {} imagini!\nSelectati un folder nou sau inchideti!".format(len(self.imageNames)))

def classifyImage(filePath):
    frame = cv2.imread(filePath)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectie fata (preluare coordonate) cu HaarCascade:
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.01, minNeighbors=5)
    for x, y, w, h in faces:
        input_im = frame[y:y+h, x:x+h]
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

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, "{}, {}".format(prezic_age, prezic_gender), (x,y), font, 2, (255,255,255), 1, cv2.LINE_AA)

        # Creeaza o lista cu grad apartenenta la fiecare clasa [%]
        outputs_prob = output_data_age.tolist()[0]
        outputs_prob = [[string_pred_age[i]+':', round(outputs_prob[i],4)] for i in range(len(outputs_prob))]
        outputs_prob.sort(key=lambda c: -c[1])
        outputs_str = prezic_gender + '\n'
        for i in range(len(outputs_prob)):
            outputs_str += " ".join(str(_) for _ in outputs_prob[i]) + '\n'
        # print(outputs_str)
    
    # Frame info for PyQT
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, frame_channel = frame.shape
    bytesPerLine = frame_channel * frame_width

    # create QImage from frame
    qImg = QImage(frame.data, frame_width, frame_height,
                    bytesPerLine, QImage.Format_RGB888)
    return qImg, outputs_str

def main():
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    main = UI_MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
