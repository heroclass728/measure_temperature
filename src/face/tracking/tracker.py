import cv2
import dlib
import collections
import time
import numpy as np

from src.face.detection.detector import FaceDetectorHaar, FaceDetectorRes10
from src.filter.tracker_filter import filter_undetected_trackers
from src.filter.nms import non_max_suppression_slow
from settings import MARGIN, MODEL_HAAR

if MODEL_HAAR:
    face_detector = FaceDetectorHaar()
else:
    face_detector = FaceDetectorRes10()


def track_faces(face_frame, trackers, attributes, w_ratio, h_ratio):

    all_track_rects = []
    all_track_keys = []

    for fid in trackers.keys():
        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        t_top = int(tracked_position.top())
        t_right = int(tracked_position.right())
        t_bottom = int(tracked_position.bottom())
        all_track_rects.append([t_left, t_top, t_right, t_bottom])
        all_track_keys.append(fid)

    filter_ids = non_max_suppression_slow(boxes=np.array(all_track_rects), keys=all_track_keys)

    for idx in filter_ids:
        attributes.pop(idx)
        trackers.pop(idx)

    for fid in trackers.keys():

        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        if t_left < 0:
            t_left = 0
        t_top = int(tracked_position.top())
        if t_top < 0:
            t_top = 0
        t_right = int(tracked_position.right())
        if t_right > face_frame.shape[1]:
            t_right = face_frame.shape[1]
        t_bottom = int(tracked_position.bottom())
        if t_bottom > face_frame.shape[0]:
            t_bottom = face_frame.shape[0]

        t_center_x = int(0.5 * (t_left + t_right))
        t_center_y = int(0.5 * (t_top + t_bottom))

        attributes[fid]["centers"].append([t_center_x, t_center_y])
        attributes[fid]["face"] = [t_left, t_top, t_right, t_bottom]
        cv2.rectangle(face_frame, (int(w_ratio * t_left), int(h_ratio * t_top)),
                      (int(w_ratio * t_right), int(h_ratio * t_bottom)), (0, 0, 255), 3)

    return face_frame, attributes


def create_face_tracker(detect_img, show_img, trackers, attributes, face_id, w_ratio, h_ratio):

    st_time = time.time()
    face_coordinates = face_detector.detect_face(frame=detect_img)
    print(time.time() - st_time)

    detected_centers = []

    for coordinates in face_coordinates:

        left, top, right, bottom = coordinates
        x_bar = 0.5 * (left + right)
        y_bar = 0.5 * (top + bottom)
        detected_centers.append([left, top, right, bottom])

        matched_fid = None

        for fid in trackers.keys():

            tracked_position = trackers[fid].get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())

            # calculate the center point
            t_x_bar = 0.5 * (t_left + t_right)
            t_y_bar = 0.5 * (t_top + t_bottom)

            # check if the center point of the face is within the rectangle of a tracker region.
            # Also, the center point of the tracker region must be within the region detected as a face.
            # If both of these conditions hold we have a match

            if t_left <= x_bar <= t_right and t_top <= y_bar <= t_bottom and left <= t_x_bar <= right \
                    and top <= t_y_bar <= bottom:
                matched_fid = fid
                trackers.pop(fid)
                tracker = dlib.correlation_tracker()
                tracker.start_track(detect_img, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                               bottom + MARGIN))
                trackers[matched_fid] = tracker
                attributes[matched_fid]["undetected"] = 0
                cv2.rectangle(show_img, (int(w_ratio * t_left), int(h_ratio * t_top)),
                              (int(w_ratio * t_right), int(h_ratio * t_bottom)), (0, 0, 255), 3)

        # If no matched fid, then we have to create a new tracker
        if matched_fid is None:
            print("Creating new tracker " + str(face_id))
            # Create and store the tracker
            tracker = dlib.correlation_tracker()
            tracker.start_track(show_img, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                           bottom + MARGIN))
            trackers[face_id] = tracker

            temp_dict = collections.defaultdict()
            temp_dict["id"] = str(face_id)
            temp_dict["centers"] = [[x_bar, y_bar]]
            temp_dict["face"] = [left, top, right, bottom]
            temp_dict["undetected"] = 0
            attributes[face_id] = temp_dict
            cv2.rectangle(show_img, (int(w_ratio * left), int(h_ratio * top)),
                                        (int(w_ratio * right), int(h_ratio * bottom)), (0, 0, 255), 3)

            face_id += 1

    trackers, attributes = filter_undetected_trackers(trackers=trackers, attributes=attributes,
                                                      detected_rects=detected_centers)

    return trackers, attributes, face_id, show_img


if __name__ == '__main__':

    track_faces(face_frame=cv2.imread(""), trackers={}, attributes={}, w_ratio=0, h_ratio=0)
