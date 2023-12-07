import os
import shutil
import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


def most_frequent(List):
    return max(set(List), key=List.count)


def get_most_frequent_emotion(dict_):

    emotions = []
    for frame_nmr in dict_.keys():
        for face_nmr in dict_[frame_nmr].keys():
            emotions.append(dict_[frame_nmr][face_nmr]['emotion'])

    return most_frequent(emotions)


def process():

    # parameters for loading data and images
    image_path = sys.argv[1]
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    frames_dir = './.tmp'
    if image_path[-3:] in ['jpg', 'png']:
        images_list = [image_path]
    else:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.mkdir(frames_dir)
        os.system('ffmpeg -i {} {}/$frame_%010d.jpg'.format(image_path, frames_dir))
        images_list = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))]

    output = {}
    for image_path_, image_path in enumerate(images_list):
        # loading images
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)

        tmp = {}
        for face_coordinates_, face_coordinates in enumerate(faces):

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]

            tmp[face_coordinates_] = {'emotion': emotion_text, 'score': np.amax(emotion_classifier.predict(gray_face))}

        output[image_path_] = tmp

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    return output, get_most_frequent_emotion(output)


if __name__ == "__main__":
    output, most_frequent_emotion = process()

    for key in output.keys():
        print(output[key])

    print(most_frequent_emotion)
