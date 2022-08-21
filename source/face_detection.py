from retinaface import RetinaFace
import torch
from torchvision.utils import save_image
from facenet_pytorch import MTCNN
import numpy as np


class FaceDetector:

    # first call extract_face
    def __init__(self, minimum_confidence, post_process=False, output_size=250, save_path_prefixes=None):
        self.minimum_confidence = minimum_confidence
        self.save_path_prefixes = save_path_prefixes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector_model = MTCNN(
            keep_all=True,
            device=device,
            post_process=post_process,
            image_size=output_size,
            margin=output_size // 10, )

        self.images_faces_bounding_boxes = None
        self.images_faces_facial_keypoints = None
        self.images_faces_scores = None
        self.images_faces = None

    def compute(self, input_batch_images):
        images_faces_bounding_boxes, images_faces_scores, images_faces_facial_keypoints = self.detector_model.detect(input_batch_images, landmarks=True)

        images_faces = self.detector_model.extract(input_batch_images, images_faces_bounding_boxes, self.save_path_prefixes)
        images_faces = [faces[faces_scores >= self.minimum_confidence] for faces_scores, faces in zip(images_faces_scores, images_faces) if faces_scores[0] is not None]
        images_faces_bounding_boxes = [faces_bounding_boxes[faces_scores >= self.minimum_confidence] for faces_scores, faces_bounding_boxes in zip(images_faces_scores, images_faces_bounding_boxes) if faces_scores[0] is not None]
        images_faces_facial_keypoints = [faces_facial_keypoints[faces_scores >= self.minimum_confidence] for faces_scores, faces_facial_keypoints in zip(images_faces_scores, images_faces_facial_keypoints) if faces_scores[0] is not None]
        images_faces_scores = [faces_scores[faces_scores >= self.minimum_confidence] for faces_scores in images_faces_scores if faces_scores[0] is not None]

        images_faces = [image_faces.permute(0, 2, 3, 1).numpy().astype(np.uint8) for image_faces in images_faces]
        # images_faces = [image_faces / 255.0 for image_faces in images_faces]
        images_faces_bounding_boxes = [faces_bounding_boxes.round().astype(np.int16) for faces_bounding_boxes in images_faces_bounding_boxes]
        images_faces_facial_keypoints = [faces_facial_keypoints.round().astype(np.int16) for faces_facial_keypoints in images_faces_facial_keypoints]

        self.images_faces = images_faces
        self.images_faces_bounding_boxes, self.images_faces_scores, self.images_faces_facial_keypoints = images_faces_bounding_boxes, images_faces_scores, images_faces_facial_keypoints

        return self

    def get_bounding_boxes(self, ):
        return self.images_faces_bounding_boxes

    def get_facial_keypoints(self, ):
        return self.images_faces_facial_keypoints

    def get_images_faces(self, ):
        return self.images_faces

    def get_images_faces_scores(self, ):
        return self.images_faces_scores

    def get_eyes_coordinates(self, ):
        images_faces_eyes_coordinates = []
        for facial_keypoints in self.images_faces_facial_keypoints:
            image_faces_eyes_coordinates = facial_keypoints[:, [0, 1], :]
            images_faces_eyes_coordinates.append(image_faces_eyes_coordinates)

        return images_faces_eyes_coordinates
