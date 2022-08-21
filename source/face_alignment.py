import numpy as np
from scipy.spatial.distance import euclidean
from skimage.transform import rotate
import math


class FaceAligner:

    def __init__(self, ):
        pass

    @staticmethod
    def __apply_rotation_on_image_faces__(image_faces, angles):
        rotated_image_faces = [rotate(face, angle) for face, angle in zip(image_faces, angles)]
        return rotated_image_faces

    @staticmethod
    def __compute_alignment_rotation_of_image_faces__(eyes_coordinates):
        angles, directions = [], []
        for left_eye_coordinate, right_eye_coordinate in eyes_coordinates:

            left_eye_x, left_eye_y = left_eye_coordinate
            right_eye_x, right_eye_y = right_eye_coordinate

            triangle_vertex = (right_eye_x, left_eye_y) if left_eye_y > right_eye_y else (left_eye_x, right_eye_y)
            direction = -1 if left_eye_y > right_eye_y else 1  # rotate clockwise else counter-clockwise

            # compute length of triangle edges
            a = euclidean(left_eye_coordinate, triangle_vertex)
            b = euclidean(right_eye_coordinate, triangle_vertex)
            c = euclidean(right_eye_coordinate, left_eye_coordinate)

            # cosine rule
            if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
                cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
                angle = np.arccos(cos_a)  # angle in radian
                angle = (angle * 180) / math.pi  # radian to degree
            else:
                angle = 0

            angle = angle - 90 if direction == -1 else angle

            angles.append(angle)
            directions.append(direction)

        return angles, directions

    def __align_image_faces__(self, input_image_faces, input_image_faces_eyes_coordinates):
        angles, directions = self.__compute_alignment_rotation_of_image_faces__(input_image_faces_eyes_coordinates)
        rotated_image_faces = self.__apply_rotation_on_image_faces__(input_image_faces, angles)
        return rotated_image_faces

    def align_images_faces(self, input_images_faces, input_images_faces_eyes_coordinates):
        return [self.__align_image_faces__(faces, faces_eyes_coordinates) for faces, faces_eyes_coordinates in zip(input_images_faces, input_images_faces_eyes_coordinates)]
