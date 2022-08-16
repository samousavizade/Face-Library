import os
from PIL import Image
import torch
from torchvision import transforms
import urllib


def get_model_path(model_name):
    model_file = model_name + '.pt'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotions')
    # cache_dir = "emotion_models"
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, model_file)
    if not os.path.isfile(fpath):
        print(f"{model_file} not exists")
        url = 'https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/' + model_file + '?raw=true'
        print('Downloading', model_name, 'from', url)
        urllib.request.urlretrieve(url, fpath)

    return fpath


class FaceEmotionRecognizer:
    # supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b0_8_best_vgaf', device="cpu"):
        self.device = "cuda:0" if torch.cuda.is_available() and device == "cuda:0" else "cpu"
        self.is_mtl = '_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
        else:
            self.idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        self.img_size = 224 if '_b0_' in model_name else 260
        self.test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        path = get_model_path(model_name)

        model = torch.load(path)
        model = model.to(device)

        if isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.data
            self.classifier_bias = model.classifier[0].bias.data
        else:
            self.classifier_weights = model.classifier.weight.data
            self.classifier_bias = model.classifier.bias.data

        model.classifier = torch.nn.Identity()
        self.model = model.eval()
        print(path, self.test_transforms)

    def compute_probability(self, features):
        return torch.matmul(features, self.classifier_weights.T) + self.classifier_bias

    def extract_representations_from_faces(self, input_faces):
        faces = [self.test_transforms(Image.fromarray(face)) for face in input_faces]
        features = self.model(torch.stack(faces, dim=0).to(self.device))
        return features

    def predict_emotions_from_representations(self, representations, logits=True, return_features=True):
        scores = self.compute_probability(representations)
        if self.is_mtl:
            predictions_indices = torch.argmax(scores[:, :-2], dim=1)

        else:
            predictions_indices = torch.argmax(scores, dim=1)

        if self.is_mtl:
            x = scores[:, :-2]

        else:
            x = scores
        pred = torch.argmax(x[0])

        if not logits:
            e_x = torch.exp(x - torch.max(x, dim=1)[:, None])
            e_x = e_x / e_x.sum(dim=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_class[pred.item()] for pred in (predictions_indices)], scores
