from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2

from src.pylepton.lepton import Lepton


def read_flir():
    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()
    device = "/dev/spidev0.0"
    with Lepton(device) as lep:
        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.rotate(frame, -90)

            # update the FPS counter
            fps.update()

            # show the output frame
            # vis = cv2.flip( vis, 0)
            # vis = cv2.flip( vis, 1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':

    read_flir()
