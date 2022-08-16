import mediapipe
import numpy as np
import pandas as pd
import cv2


class FaceNormalizer:

    def __init__(self):
        self.mp_face_mesh = mediapipe.solutions.face_mesh
        face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)

        mp_face_mesh = mediapipe.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        self.routes_idx = self.initialize__()

    def initialize__(self):
        df = pd.DataFrame(list(self.mp_face_mesh.FACEMESH_FACE_OVAL), columns=["p1", "p2"])
        routes_idx = []

        p1 = df.iloc[0]["p1"]
        p2 = df.iloc[0]["p2"]

        for i in range(0, df.shape[0]):
            obj = df[df["p1"] == p2]
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]

            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)

        return routes_idx

    def get_landmarks__(self, input_image: np.ndarray):
        if input_image.dtype == np.float:
            input_image = (input_image * 255).astype(np.uint8)

        results = self.face_mesh.process(input_image)
        landmarks = results.multi_face_landmarks[0]

        routes = []
        # for source_idx, target_idx in mp_face_mesh.FACEMESH_FACE_OVAL:
        for source_idx, target_idx in self.routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (int(input_image.shape[1] * source.x), int(input_image.shape[0] * source.y))
            relative_target = (int(input_image.shape[1] * target.x), int(input_image.shape[0] * target.y))

            # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)

            routes.append(relative_source)
            routes.append(relative_target)

        return routes

    @staticmethod
    def normalize_with_landmark_points__(input_image, landmarks):
        mask = np.zeros((input_image.shape[0], input_image.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(landmarks), 1)
        mask = mask.astype(bool)

        out = np.zeros_like(input_image)
        out[mask] = input_image[mask]
        return out

    def normalize_faces_image(self, input_images):
        normalized_faces_images = [self.normalize_with_landmark_points__(input_image, self.get_landmarks__(input_image)) for input_image in input_images]
        return normalized_faces_images