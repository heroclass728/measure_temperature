import imutils
import time
import cv2
import numpy as np

from imutils.video import VideoStream
# from imutils.video import FPS
from src.pylepton.lepton import Lepton
from src.face.tracking.tracker import create_face_tracker, track_faces
from settings import LOCAL, VIDEO_PATH, BASE_LINE, DETECT_RESIZED, TRACK_QUALITY, POSITIVE_DIRECTION, \
    NEGATIVE_DIRECTION, FACE_TRACK_CYCLE, SHOW_RESIZED, DEVICE


class PersonCounterTemperature:

    def __init__(self):

        self.face_trackers = {}
        self.face_attributes = {}
        self.face_id = 1
        self.base_line_axis = None
        self.base_value = 0
        self.positives = 0
        self.negatives = 0
        self.show_height = None
        self.show_width = None
        self.w_ratio = None
        self.h_ratio = None
        self.lepton_buf = np.zeros((80, 60, 1), dtype=np.uint16)
        self.last_nr = 0

    @staticmethod
    def __init_base_line_axis(width, height, w_ratio, h_ratio):

        x_interval = width * (BASE_LINE[2] - BASE_LINE[0])
        y_interval = height * (BASE_LINE[3] - BASE_LINE[1])
        if x_interval > y_interval:
            axis = 1
            base_value = int(0.5 * height * (BASE_LINE[3] + BASE_LINE[1]) / h_ratio)
        else:
            axis = 0
            base_value = int(0.5 * width * (BASE_LINE[2] + BASE_LINE[0]) / w_ratio)

        return axis, base_value

    def __init_lepton(self):

        with Lepton(DEVICE) as lep:
            _, nr = lep.capture(self.lepton_buf)

        return nr

    def calculate_temperature(self, temp_frame, w_ratio, h_ratio):

        self.lepton_buf = np.flip(self.lepton_buf, 1)
        # print(np.min(lepton_buf), np.max(lepton_buf))
        array = ((self.lepton_buf.copy() * 0.0439 - 321) * 12.5 - 287)
        cv2.imshow("lepton", self.lepton_buf)
        cv2.waitKey()
        array = array.astype(np.uint8)
        array = cv2.resize(array, (640, 480))
        array = cv2.applyColorMap(array, cv2.COLORMAP_JET)

        for fid in self.face_attributes.keys():

            face_left = self.face_attributes[fid]["face"][0]
            face_top = self.face_attributes[fid]["face"][1]
            face_right = self.face_attributes[fid]["face"][2]
            face_bottom = self.face_attributes[fid]["face"][3]

            face_left_real = int(2 * face_left * self.lepton_buf.shape[1] / array.shape[1])
            face_top_real = int(2 * face_top * self.lepton_buf.shape[0] / array.shape[0])
            face_right_real = int(2 * face_right * self.lepton_buf.shape[1] / array.shape[1])
            face_bottom_real = int(2 * face_bottom * self.lepton_buf.shape[0] / array.shape[0])
            temp_array = self.lepton_buf[face_top_real: face_bottom_real, face_left_real: face_right_real, :]
            if temp_array.size == 0:
                temp_val = 0
            else:
                temp_val = '{:.2f}'.format(np.max(temp_array) * 0.0439 - 321)

            cv2.rectangle(array, (int(w_ratio * face_left), int(h_ratio * face_top)),
                          (int(w_ratio * face_right), int(h_ratio * face_bottom)), (0, 0, 255), 2)

            cv2.putText(temp_frame, temp_val, (int(self.w_ratio * face_left) + 3, int(self.h_ratio * face_top) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return temp_frame, array

    def count_person(self):

        for fid in self.face_trackers.keys():
            # if not self.person_attributes[fid]["counted"]:
            init_pos_axis = self.face_attributes[fid]["centers"][0][self.base_line_axis]
            current_pos = self.face_attributes[fid]["centers"][-1]
            current_pos_axis = current_pos[self.base_line_axis]
            y = [c[self.base_line_axis] for c in self.face_attributes[fid]["centers"]]
            direction = current_pos_axis - np.mean(y)

            if direction < 0 and current_pos_axis < self.base_value < init_pos_axis:
                self.negatives += 1
                self.face_attributes[fid]["centers"] = []
                self.face_attributes[fid]["centers"].append(current_pos)

            elif direction > 0 and current_pos_axis > self.base_value > init_pos_axis:
                self.positives += 1
                self.face_attributes[fid]["centers"] = []
                self.face_attributes[fid]["centers"].append(current_pos)

        return

    def main(self):

        if LOCAL:
            cap = cv2.VideoCapture(VIDEO_PATH)
        else:
            cap = VideoStream(usePiCamera=True).start()
            time.sleep(2.0)
        cnt = 0

        while True:

            if LOCAL:
                _, frame = cap.read()
            else:
                frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame = imutils.rotate(frame, -90)

            resized_image = cv2.resize(frame, (DETECT_RESIZED, DETECT_RESIZED))
            show_img = cv2.resize(frame, (SHOW_RESIZED[0], SHOW_RESIZED[1]))
            if self.show_height is None:
                self.show_height, self.show_width = show_img.shape[:2]
                # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                self.w_ratio = self.show_width / DETECT_RESIZED
                self.h_ratio = self.show_height / DETECT_RESIZED

            if self.base_line_axis is None:
                self.base_line_axis, self.base_value = self.__init_base_line_axis(width=self.show_width,
                                                                                  height=self.show_height,
                                                                                  w_ratio=self.w_ratio,
                                                                                  h_ratio=self.h_ratio)

            cv2.line(show_img, (int(BASE_LINE[0] * self.show_width), int(BASE_LINE[1] * self.show_height)),
                     (int(BASE_LINE[2] * self.show_width), int(BASE_LINE[3] * self.show_height)), (0, 255, 0), 5)

            fids_to_delete = []
            for fid in self.face_trackers.keys():
                tracking_quality = self.face_trackers[fid].update(resized_image)

                # If the tracking quality is good enough, we must delete this tracker
                if tracking_quality < TRACK_QUALITY:
                    fids_to_delete.append(fid)

            for fid in fids_to_delete:
                print("Removing fid " + str(fid) + " from list of trackers")
                self.face_trackers.pop(fid, None)
                self.face_attributes.pop(fid, None)

            # result_img = show_img

            if cnt % FACE_TRACK_CYCLE == 0:
                self.face_trackers, self.face_attributes, self.face_id, result_img = \
                    create_face_tracker(detect_img=resized_image, trackers=self.face_trackers,
                                          attributes=self.face_attributes, face_id=self.face_id,
                                          show_img=show_img, w_ratio=self.w_ratio, h_ratio=self.h_ratio)
            else:
                st_time = time.time()
                result_img, self.face_attributes = track_faces(face_frame=show_img, w_ratio=self.w_ratio,
                                                                 h_ratio=self.h_ratio, trackers=self.face_trackers,
                                                                 attributes=self.face_attributes)
                print("tracking time:", time.time() - st_time)

            cnt += 1
            self.count_person()

            if not LOCAL:
                if self.last_nr == self.__init_lepton():
                    continue
                else:
                    self.last_nr = self.__init_lepton()

            temp_img, thermal_array = self.calculate_temperature(temp_frame=result_img, w_ratio=self.w_ratio,
                                                                 h_ratio=self.h_ratio)

            cv2.putText(temp_img, "{} : {}".format(POSITIVE_DIRECTION, self.positives), (10, self.show_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(temp_img, "{} : {}".format(NEGATIVE_DIRECTION, self.negatives), (10, self.show_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            final_frame = np.concatenate((thermal_array, temp_img), axis=1)

            cv2.imshow("image", final_frame)
            # time.sleep(0.05)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # press q to quit
                break
        # kill open cv things
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    PersonCounterTemperature().main()
