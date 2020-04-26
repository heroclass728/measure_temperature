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
UNDETECTED_THRESH = 3
POSITIVE_DIRECTION = "Right"
NEGATIVE_DIRECTION = "Left"

BASE_LINE = [0.5, 0, 0.5, 1]
DEVICE = "/dev/spidev0.0"

LOCAL = False
