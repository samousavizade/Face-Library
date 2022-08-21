from setuptools import setup, find_packages

long_description = "Python high level tools library to use in face detection, alignment, recognition, emotion, sentiment, normalization and etc."

setup(
    name="pyfacelib",
    version="0.0.2",
    description="Python High Level Face Tools Library",
    long_description=long_description + "\n\n" + open("CHANGELOG.txt").read(),
    keywords=[
        "Detection",
        "Recognition",
        "Alignment",
        "Normalization",
        "Emotion",
        "Sentiment",
        "Face Detection",
        "Face Recognition",
        "Face Alignment",
        "Face Normalization",
        "Face Emotion",
        "Face Sentiment",
    ],
    author="SAlirezaMousavizade",
    author_email="s.a.mosavizade@gmail.com",
    packages=find_packages(),
)
