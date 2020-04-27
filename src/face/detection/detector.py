import cv2
import numpy as np

from settings import FRONT_FACE_DETECTION_MODEL, PROTO_PATH, MODEL_PATH, CONFIDENCE_THRESH


class FaceDetectorHaar:

    def __init__(self):

        self.front_detector = cv2.CascadeClassifier(FRONT_FACE_DETECTION_MODEL)

    @staticmethod
    def reshape_faces(faces):
        new_faces = []
        for face in faces:
            x, y, w, h = face
            if w == 0 or h == 0:
                continue
            new_faces.append([x, y, x + w, y + h])

        return new_faces

    def detect_face(self, frame):

        faces = self.front_detector.detectMultiScale(frame, 1.1, 5)
        rearranged_faces = self.reshape_faces(faces=faces)

        return rearranged_faces


class FaceDetectorRes10:

    def __init__(self):

        self.face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

    def detect_face(self, frame):

        faces = []
        h, w = frame.shape[:2]
        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                           swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        self.face_detector.setInput(image_blob)
        detections = self.face_detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > CONFIDENCE_THRESH:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # extract the face ROI
                face = frame[start_y:end_y, start_x:end_x]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 5 or fH < 5:
                    continue

                faces.append([start_x, start_y, end_x, end_y])

        return faces


if __name__ == '__main__':
    FaceDetectorHaar().detect_face(frame=cv2.imread(""))
