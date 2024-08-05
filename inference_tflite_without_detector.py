import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Engine:
    @staticmethod
    def preprocess_image(image):
        """Resize and normalize the image."""
        image = tf.image.resize(image, (112, 112), method="nearest")
        return (tf.cast(image, tf.float32) - 127.5) / 128.

    def __init__(self, model_path: str):
        """Initialize the TFLite interpreter."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_output(self, image):
        """Run inference on the provided image."""
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        return tf.nn.l2_normalize(self.interpreter.get_tensor(self.output_details[0]['index']), axis=1)

    def __call__(self, image_path1: str, image_path2: str, th: float = 1.0):
        """Load images, preprocess, perform inference, and compare outputs."""
        image1 = self.load_and_preprocess_image(image_path1)
        image2 = self.load_and_preprocess_image(image_path2)
        
        output1 = self.get_output(image1[np.newaxis, ...])
        output2 = self.get_output(image2[np.newaxis, ...])

        dist = tf.norm(output1 - output2)
        status = "same person" if dist < th else "different persons"
        print(f"Distance: {dist}, Result: {status}")
        
        self.display_image(image1, title="Image 1")
        self.display_image(image2, title="Image 2")

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess the image from the path."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = self.preprocess_image(image)
        return image

    def display_image(self, image, title=None):
        """Display the image using Matplotlib."""
        plt.figure(figsize=(8, 8))
        plt.imshow(image.numpy() * 0.5 + 0.5)  # Rescale the image for display
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
        

if __name__ == '__main__':
	e = Engine("/gdrive/My Drive/Arcface/model_140000.tflite")
	e("data/t2_face.jpg", "data/t4_face.jpg", th=1.14)  # give two image paths and threshold

