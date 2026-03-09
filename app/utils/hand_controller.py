import cv2
import mediapipe as mp
import os


class HandController:

    def __init__(self):
        self.cap      = cv2.VideoCapture(0)
        self.detector = None          # set before creation so close() is always safe

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path   = os.path.join(project_root, "hand_landmarker.task")

        BaseOptions        = mp.tasks.BaseOptions
        HandLandmarker     = mp.tasks.vision.HandLandmarker
        HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode

        options = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1
        )

        self.detector = HandLandmarker.create_from_options(options)
        self.frame_id = 0

    def read(self):
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            results = self.detector.detect_for_video(mp_image, self.frame_id)
            self.frame_id += 1

            if results.hand_landmarks:
                tip = results.hand_landmarks[0][8]   # index finger tip
                return tip.x, tip.y

            return None
        except Exception as e:
            # Graceful degradation: webcam error doesn't crash simulation
            print(f"[HandController] Read warning (webcam may be disconnected): {e}")
            return None

    def close(self):
        # FIX: detector must be closed BEFORE cap is released
        try:
            if self.detector is not None:
                self.detector.close()
                self.detector = None
        except Exception as e:
            print(f"[HandController] detector close warning: {e}")

        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"[HandController] cap release warning: {e}")