import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Utilities:
    def __init__(self, max_pointcloud_length=8196, max_perimeter_length = 21):
        self.max_pointcloud_length = max_pointcloud_length
        self.max_perimeter_length = max_perimeter_length

    def pad_label(self, label, max_length):
        """
        Pads or trims the perimeter label to a fixed length of points (values for x and y coordinates).
        """
        result = NotImplementedError
        max_values = max_length * 2

        if len(label) < max_values:
            # Padding with zeros
            result = np.pad(label, (0, max_values - len(label)), mode='constant')
        else:
            logging.info(f"Truncating label from {len(label)} to {max_values}")
            result = label[:max_values]

        return result

    def preprocess_image(self, image_path):
        logging.info(f"Preprocessing image: {image_path}")

        image = Image.open(image_path)
        image = image.resize((224, 224))  # Resize image
        image = np.array(image) / 255.0  # Normalize image

        logging.info(f"Image processed with shape: {image.shape}")
        return image

    def preprocess_pointcloud(self, dsm_path, max_points):
        """
        Preprocess point cloud data by loading, padding, or trimming to a fixed number of points.
        """
        logging.info(f"Preprocessing point cloud: {dsm_path}")

        with open(dsm_path, 'r') as f:
            pointcloud = json.load(f)
        
        # Convert point cloud to a NumPy array
        pointcloud_np = np.array(pointcloud)
        
        # Trim or pad the point cloud
        if len(pointcloud_np) > max_points:
            pointcloud_np = pointcloud_np[:max_points]
        elif len(pointcloud_np) < max_points:
            padding = np.zeros((max_points - len(pointcloud_np), pointcloud_np.shape[1]))
            pointcloud_np = np.vstack((pointcloud_np, padding))
        
        logging.info(f"Point cloud processed with shape: {pointcloud_np.shape}")
        return pointcloud_np

    def process_directory(self, directory):
        logging.info(f"Processing directory: {directory}")

        image_path = os.path.join(directory, "google.jpg")
        dsm_path = os.path.join(directory, "dsm.json")
        planes_path = os.path.join(directory, "planes.json")

        # Preprocess image and point cloud
        image = self.preprocess_image(image_path)
        pointcloud = self.preprocess_pointcloud(dsm_path, self.max_pointcloud_length)

        # Load and preprocess planes (labels)
        with open(planes_path, 'r') as f:
            planes_data = json.load(f)
        
        if planes_data == []: 
            logging.info("Empty planes file found {}".format(planes_path))
            labels = None
        else:
            # Extract relevant information for labels
            azimuth = planes_data[0]['azimuth']
            height = planes_data[0]['height']
            tilt = planes_data[0]['tilt']
            perimeter = [coord for point in planes_data[0]['perimeter'] for coord in (point['x'], point['y'])]

            # Ensure the perimeter is padded to a fixed length
            perimeter = self.pad_label(perimeter, max_length=self.max_perimeter_length)

            labels = [azimuth, height, tilt, perimeter]

            logging.info(f"Processed directory with azimuth: {azimuth}, height: {height}, tilt: {tilt}, perimeter size: {len(perimeter)}")
        
        return image, pointcloud, labels