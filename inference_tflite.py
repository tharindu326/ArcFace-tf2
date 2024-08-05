import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from face_detection import mtcnn_detector

class Engine:
    @staticmethod
    def turn_rgb(images):
        b, g, r = tf.split(images, 3, axis=-1)
        images = tf.concat([r, g, b], -1)
        return images

    @staticmethod
    def set_face(face):
        face = tf.image.resize(face, (112, 112), method="nearest")
        return (tf.cast(face, tf.float32) - 127.5) / 128.

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.detector = mtcnn_detector.Engine()

    def get_output(self, images):
        self.interpreter.set_tensor(self.input_details[0]['index'], images)
        self.interpreter.invoke()
        return tf.nn.l2_normalize(self.interpreter.get_tensor(self.output_details[0]['index']), axis=1)

    def __call__(self, main_path1: str, main_path2: str, th: float = 1.0):
        image = self.detector.load_image(main_path1)
        faces = self.detector.get_faces_from_image(image)
        boxes1 = self.detector.get_boxes_from_faces(faces)
        face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes1))
        face_frames = np.array([self.set_face(n) for n in face_frames], dtype=np.float32)
        image1 = image.copy()
        output1 = self.get_output(face_frames)[0]

        image = self.detector.load_image(main_path2)
        faces = self.detector.get_faces_from_image(image)
        boxes2 = self.detector.get_boxes_from_faces(faces)
        face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes2))
        face_frames = np.array([self.set_face(n) for n in face_frames], dtype=np.float32)
        image2 = image.copy()
        output2 = self.get_output(face_frames)[0]

        dist = tf.norm(output1 - output2)
        color = (0, 0, 255)
        status = "different persons"
        if dist < th:
            color = (0, 255, 0)
            status = "same person"

        print(f"Distance --> {dist}, Those images belong to {status}.")
        image1 = self.detector.draw_faces_on_image(image1, boxes1, color=color)
        image2 = self.detector.draw_faces_on_image(image2, boxes2, color=color)

        self.display_image(image1, title="Image 1")
        self.display_image(image2, title="Image 2")
  
    def display_image(self, image, title=None):
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
        

if __name__ == '__main__':
	e = Engine("/gdrive/My Drive/Arcface/model_140000.tflite")
	e("data/t2.jpg", "data/t4.jpg", th=1.14)  # give two image paths and threshold

