import cv2
import config


class CameraCapture:
    """Thin wrapper around cv2.VideoCapture for camera management."""

    def __init__(self, cameraIndex=None):
        self.cameraIndex = cameraIndex if cameraIndex is not None else config.CameraIndex
        self.capture = None

    def start(self):
        """Open the camera. Returns True if successful."""
        try:
            self.capture = cv2.VideoCapture(self.cameraIndex)
            if not self.capture.isOpened():
                print(f"[CameraCapture] Failed to open camera {self.cameraIndex}")
                self.capture = None
                return False
            return True
        except Exception as e:
            print(f"[CameraCapture] Error starting camera: {e}")
            self.capture = None
            return False

    def readFrame(self):
        """
        Read a single frame from the camera.

        Returns:
            BGR numpy array, or None if read failed.
        """
        if self.capture is None:
            return None
        try:
            ret, frame = self.capture.read()
            if ret:
                return frame
            return None
        except Exception as e:
            print(f"[CameraCapture] Error reading frame: {e}")
            return None

    def stop(self):
        """Release the camera."""
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception as e:
                print(f"[CameraCapture] Error stopping camera: {e}")
            finally:
                self.capture = None

    def isOpened(self):
        """Check if camera is currently open."""
        return self.capture is not None and self.capture.isOpened()
