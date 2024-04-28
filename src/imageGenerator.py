import os

import cv2
import numpy as np


class ImageGenerator:
    """
    Generate images from bytes files
    """

    def __init__(self, bytes_dir, graph_dir):
        """
        Initialize the image generator
        """
        # input path
        self.BYTES_DIR = bytes_dir
        # output path
        self.GRAPH_DIR = graph_dir

    def generate_graph(self, input_folder_name):
        """
        Generate images from bytes files
        """
        input_dir_path = os.path.join(self.BYTES_DIR, input_folder_name)
        output_dir_path = os.path.join(self.GRAPH_DIR, input_folder_name)
        os.makedirs(output_dir_path, exist_ok=True)
        file_list = os.listdir(input_dir_path)
        for input_file_name in file_list:
            self._process_file(input_dir_path, output_dir_path, input_file_name)

    def _process_file(self, input_dir_path, output_dir_path, input_file_name):
        """
        Process a file to generate an image
        """
        input_file_path = os.path.join(input_dir_path, input_file_name)
        input_file_name = os.path.splitext(input_file_name)[0]
        output_file_path = os.path.join(output_dir_path, f"{input_file_name}.png")
        with open(input_file_path, "rb") as f:
            data = f.read()
        # read the bytes file with uint8
        arr = np.frombuffer(data, dtype=np.uint8)
        # reshape the array to 28x28
        image = arr.reshape(28, 28)
        # convert to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # save the image
        cv2.imwrite(output_file_path, image)
