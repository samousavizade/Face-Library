from face_detection import FaceDetection
from face_alignment import FaceAlignment
from face_normalization import FaceNormalizer
from face_emotion_recognition import FaceEmotionRecognizer
import numpy as np
import pickle


class EmotionRepresentationExtractor:
    def __init__(self, ):
        self.face_detection_model: FaceDetection = None
        self.face_alignment_model: FaceAlignment = None
        self.face_normalizer_model: FaceNormalizer = None
        self.face_emotion_recognition_model: FaceEmotionRecognizer = None

        self.faces = None
        self.normalized_rotated_faces = None
        self.rotated_faces = None
        self.rotation_angles = None
        self.rotation_directions = None

    def set_face_detection_model(self, face_detection_model):
        self.face_detection_model = face_detection_model
        return self

    def set_face_alignment_model(self, face_alignment_model):
        self.face_alignment_model = face_alignment_model
        return self

    def set_face_normalizer_model(self, face_normalizer_model):
        self.face_normalizer_model = face_normalizer_model
        return self

    def set_face_emotion_recognition_model(self, face_emotion_recognition_model):
        self.face_emotion_recognition_model = face_emotion_recognition_model
        return self

    def extract_representation(self, input_image):
        faces, detected_faces_information = self.face_detection_model.extract_faces(input_image, return_detections_information=True)
        rotation_angles, rotation_directions = self.face_alignment_model.compute_alignment_rotation_(self.face_detection_model.get_eyes_coordinates())
        rotated_faces = self.face_alignment_model.apply_rotation_on_images(faces, rotation_angles)
        normalized_rotated_faces = self.face_normalizer_model.normalize_faces_image(rotated_faces)

        normalized_rotated_faces_255 = [(image * 255).astype(np.uint8) for image in normalized_rotated_faces]

        representations = self.face_emotion_recognition_model.extract_representations_from_faces(normalized_rotated_faces_255)
        predictions, scores = self.face_emotion_recognition_model.predict_emotions_from_representations(representations)

        self.faces = faces
        self.rotation_angles, self.rotation_directions = rotation_angles, rotation_directions
        self.rotated_faces = rotated_faces
        self.normalized_rotated_faces = normalized_rotated_faces_255

        return predictions, scores, representations

    def get_rotations_information(self):
        return self.rotation_angles, self.rotation_directions

    def get_faces(self):
        return self.faces

    def get_rotated_faces(self):
        return self.rotated_faces

    def get_normalized_rotated_faces(self):
        return self.normalized_rotated_faces

    def clear(self):
        self.faces = None
        self.normalized_rotated_faces = None
        self.rotated_faces = None
        self.rotation_angles = None
        self.rotation_directions = None

    def store_embeddings(self, file, embeddings):
        with open(file, "wb") as file_out:
            pickle.dump({'embeddings': embeddings}, file_out, protocol=pickle.HIGHEST_PROTOCOL)

    def load_embeddings(self, file):
        with open(file, "rb") as file_in:
            stored_data = pickle.load(file_in)
            stored_embeddings = stored_data['embeddings']

        return stored_embeddings
