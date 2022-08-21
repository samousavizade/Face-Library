from source.face_detection import FaceDetector
from source.face_alignment import FaceAligner
from source.face_normalization import FaceNormalizer
from source.face_emotion_recognition import FaceEmotionRecognizer
import pickle


class EmotionRepresentationExtractor:
    def __init__(self, ):
        self.face_detection_model: FaceDetector = None
        self.face_alignment_model: FaceAligner = None
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

    def extract_representation(self, input_images):
        images_faces = self.face_detection_model.compute(input_images, ).get_images_faces()
        aligned_images_faces = self.face_alignment_model.align_images_faces(images_faces, self.face_detection_model.get_eyes_coordinates())
        normalized_aligned_images_faces = self.face_normalizer_model.normalize_images_faces(aligned_images_faces)
        representations = self.face_emotion_recognition_model.extract_representations_from_images_faces(normalized_aligned_images_faces)

        return representations

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
