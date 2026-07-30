import os
import cv2
import torch
import pathlib
import numpy as np
import mediapipe as mp
from torchvision import transforms
from mediapipe.tasks.python import vision

class FaceProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        current_file_path = pathlib.Path(__file__).resolve()
        self.root_dir = current_file_path.parent.parent
        self.face_landmarker = self.init_face_landmarker(self.root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float32 in [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def init_face_landmarker(self, root_dir):
        model_path = os.path.join(root_dir, 'models', 'face_landmarker.task')
        
        if not os.path.exists(model_path): # Download Face Landmarker model if not present
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                model_path
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        return FaceLandmarker.create_from_options(options)

    def get_alignment_matrix(self, landmarks, image_size=(224, 224), left_eye_pos=(0.35, 0.4), right_eye_pos=(0.65, 0.4)):
        """
        Compute affine transform that aligns eye corners to canonical positions.
        landmarks are normalized [0,1], so convert to pixel coordinates first.
        image_size = (width, height)
        """
        w, h = image_size

        # MediaPipe normalized landmarks -> pixel coordinates
        l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h], dtype=np.float32)
        r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h], dtype=np.float32)

        dX = r_eye[0] - l_eye[0]
        dY = r_eye[1] - l_eye[1]
        angle = np.degrees(np.arctan2(dY, dX))

        current_dist = np.hypot(dX, dY)
        desired_dist = (right_eye_pos[0] - left_eye_pos[0]) * w

        if current_dist < 1e-6:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        scale = desired_dist / current_dist
        eye_center = ((l_eye[0] + r_eye[0]) * 0.5, (l_eye[1] + r_eye[1]) * 0.5)

        M = cv2.getRotationMatrix2D(eye_center, angle, scale)

        # Move eyes to desired canonical positions
        t_x = w * 0.5
        t_y = h * left_eye_pos[1]
        M[0, 2] += (t_x - eye_center[0])
        M[1, 2] += (t_y - eye_center[1])

        return M.astype(np.float32)

    def align_face_frame(self, frame_rgb):
        """
        frame_rgb: HxWx3 RGB uint8/float
        Returns aligned RGB image.
        """
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {frame_rgb.shape}")

        # Normalize dtype for MediaPipe
        if frame_rgb.dtype != np.uint8:
            if frame_rgb.max() <= 1.0:
                frame_rgb = (np.clip(frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        h, w, _ = frame_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return cv2.resize(frame_rgb, self.target_size)

        M = self.get_alignment_matrix(result.face_landmarks[0], image_size=(w, h))
        aligned_rgb = cv2.warpAffine(frame_rgb, M, self.target_size, flags=cv2.INTER_CUBIC)
        return aligned_rgb
    
    def extract_image(self, image_path):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        aligned_rgb = self.align_face_frame(rgb)          # HWC
        image_tensor = self.transform(aligned_rgb)        # CHW
        return image_tensor.unsqueeze(0)                  # [1, C, H, W]
    
    def extract_frames(self, video_path, num_of_frames=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < num_of_frames:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            frame_resized = cv2.resize(frame_rgb, (224, 224)) # resize image to (224x224)
            aligned_rgb = self.align_face_frame(frame_resized)
            aligned_frame_tensor = self.transform(aligned_rgb)
            frames.append(aligned_frame_tensor)
        cap.release()

        if len(frames) == 0:
            return torch.empty((0, 3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
        
        return torch.stack(frames) # [T, C, H, W]
    
    def get_face_count(self, rgb: np.ndarray) -> int:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.face_landmarker.detect(mp_image)
        return len(res.face_landmarks) if res and res.face_landmarks else 0

    def extract_image_metadata(self, image_path: str, file_ext: str) -> tuple[dict, float]:
        image_exts = {".jpg", ".jpeg", ".png"}
        
        bgr = cv2.imread(image_path)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        faces = self.get_face_count(rgb)
        
        meta = {
            "Resolution": f"{w} x {h}",
            "Duration": "N/A (image)",
            "FPS": "N/A",
            "Codec": file_ext.replace(".", "").upper(),
            "Faces found": str(faces),
        }
        
        return meta, 0.0

    def decode_fourcc(self, v: float) -> str:
        try:
            x = int(v)
            return "".join([chr((x >> 8 * i) & 0xFF) for i in range(4)]).strip() or "Unknown"
        except Exception:
            return "Unknown"

    def extract_video_metadata(self, video_path: str, file_ext: str) -> tuple[dict, float]:
        cap = cv2.VideoCapture(video_path)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        fourcc = self.decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        duration = (nframes / fps) if fps > 1e-6 else 0.0

        faces = 0
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.get_face_count(rgb)
        cap.release()

        meta = {
            "Resolution": f"{w} x {h}",
            "Duration": f"{duration:.2f}s",
            "FPS": f"{fps:.2f}" if fps > 0 else "Unknown",
            "Codec": fourcc,
            "Faces found": str(faces),
        }
        return meta, duration