import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
PROTO_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'deploy.prototxt')
FRONT_FACE_DETECTION_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'haarcascade_frontalface_default.xml')
SIDE_FACE_DETECTION_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'haarcascade_profileface.xml')
VIDEO_PATH = ""

CONFIDENCE_THRESH = 0.2
OVERLAP_THRESH = 0.7
MARGIN = 0
DETECT_RESIZED = 300
SHOW_RESIZED = [640, 480]
TRACK_QUALITY = 2
FACE_TRACK_CYCLE = 20
UNDETECTED_THRESH = 40
POSITIVE_DIRECTION = "Right"
NEGATIVE_DIRECTION = "Left"

# The ratio of the positions of of two points in base line corresponding to width, height.
# base line = [x1, y1, x2, y2].
# , where x1 =  screen width * BASE_LINE[0], y1 = screen height * BASE_LINE[1], x2 = width * BASE_LINE[2],
# y2 = height * BASE_LINE[3]

BASE_LINE = [0.5, 0, 0.5, 1]
THERMAL_COEFF_1 = 0.0439
THERMAL_COEFF_2 = -321
DEVICE = "/dev/spidev0.0"

MODEL_HAAR = True
LOCAL = False
