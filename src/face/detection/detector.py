import cv2

from settings import FRONT_FACE_DETECTION_MODEL


class FaceDetector:

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


if __name__ == '__main__':
    FaceDetector().detect_face(frame=cv2.imread(""))
