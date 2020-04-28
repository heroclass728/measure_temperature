import face_recognition
import cv2
import collections
import time

from src.face.detection.detector import FaceDetectorHaar
from settings import UNDETECTED_THRESH


class FaceMatcher:

    def __init__(self):
        self.face_detector = FaceDetectorHaar()

    def recognize_face(self, face_attributes, detect_frame, show_frame, w_ratio, h_ratio, face_id):

        st_time = time.time()
        new_faces = self.face_detector.detect_face(frame=detect_frame)
        print("detection time:", time.time() - st_time)

        st_time = time.time()
        encodings = face_recognition.face_encodings(detect_frame, new_faces, model="small")
        print("recognition time:", time.time() - st_time)

        if not face_attributes.keys():

            for i, encoding in enumerate(encodings):
                face_attributes, face_id = self.insert_new_info(attributes=face_attributes, encoding=encoding,
                                                                idx=face_id, face_box=new_faces[i], new_ret=True)

        else:

            for i, encoding in enumerate(encodings):

                encoding_ret = False
                for fid in face_attributes.keys():
                    st_time = time.time()
                    match = face_recognition.compare_faces([face_attributes[fid]["encoding"]], encoding)
                    print("match time", time.time() - st_time)

                    if match:
                        encoding_ret = True
                        face_attributes, _ = self.insert_new_info(attributes=face_attributes, encoding=encoding,
                                                                  idx=fid, face_box=new_faces[i])

                if not encoding_ret:
                    face_attributes, face_id = self.insert_new_info(attributes=face_attributes, encoding=encoding,
                                                                    idx=face_id, face_box=new_faces[i], new_ret=True)

        for fid in face_attributes.keys():
            st_time = time.time()
            detected_faces = face_recognition.compare_faces(encodings, face_attributes[fid]["encoding"])
            print("all match time", time.time() - st_time)
            if True not in detected_faces:
                face_attributes[fid]["undetected"] += 1

        del_ids = []
        for fid in face_attributes.keys():

            if face_attributes[fid]["undetected"] >= UNDETECTED_THRESH:
                del_ids.append(fid)

        for del_id in del_ids:
            face_attributes.pop(del_id)

        for fid in face_attributes.keys():
            left, top, right, bottom = face_attributes[fid]["face"]
            cv2.rectangle(show_frame, (int(w_ratio * left), int(h_ratio * top)),
                                       (int(w_ratio * right), int(h_ratio * bottom)), (0, 0, 255), 2)

        return face_attributes, face_id, show_frame

    @staticmethod
    def insert_new_info(attributes, encoding, idx, face_box, new_ret=False):

        x_center = int((face_box[0] + face_box[2]) / 2)
        y_center = int((face_box[1] + face_box[3]) / 2)
        if new_ret:
            temp_dict = collections.defaultdict()
            temp_dict["face"] = face_box
            temp_dict["encoding"] = encoding
            temp_dict["undetected"] = 0
            temp_dict["centers"] = [[x_center, y_center]]
            attributes[idx] = temp_dict
            idx += 1
        else:
            attributes[idx]["face"] = face_box
            attributes[idx]["encoding"] = encoding
            attributes[idx]["undetected"] = 0
            attributes[idx]["centers"].append([x_center, y_center])

        return attributes, idx


if __name__ == '__main__':

    FaceMatcher().recognize_face(face_attributes={}, face_id=0, w_ratio=0, h_ratio=0, detect_frame=cv2.imread(""),
                                 show_frame=cv2.imread(""))
