
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseCNNModel:
    def __init__(self, max_pointcloud_length=8196, augment=False, augmentation_count=2):
        self.max_pointcloud_length = max_pointcloud_length
        self.augment = augment
        self.augment_count = augmentation_count
        # Set seeds for repeatability
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Initialize the ImageDataGenerator for conservative augmentations
        self.datagen = ImageDataGenerator(
            rotation_range=10,        # Conservative rotation (0 to 10 degrees)
            width_shift_range=0.05,   # Small shift horizontally
            height_shift_range=0.05,  # Small shift vertically
            brightness_range=[0.9, 1.1],  # Slight brightness adjustments
            horizontal_flip=True      # Allow horizontal flips (no vertical flips)
        )
        self.datagen.seed = 42

    def _normalize_data(self, images, pointclouds, labels, resnet50=True):
        """
        Normalizes the images using preprocess_input for ResNet50, standardizes the point clouds, 
        and normalizes the labels (azimuth, height, tilt, perimeter).
        """

        if (resnet50):
            # Normalize images using preprocess_input (for ResNet50)
            images = tf.keras.applications.resnet50.preprocess_input(np.array(images).astype(np.float32))
        else:
            images = tf.keras.applications.efficientnet.preprocess_input(np.array(images).astype(np.float32))

        # Standardize point clouds
        pointclouds = np.array(pointclouds).astype(np.float32)
        pointclouds = (pointclouds - np.mean(pointclouds, axis=0)) / np.std(pointclouds, axis=0)

        # Normalize labels (azimuth, height, tilt, perimeter)
        azimuth_labels = np.array(labels[0]).astype(np.float32) / 360.0   # Normalize azimuth to [0, 1]
        tilt_labels = np.array(labels[2]).astype(np.float32) / 90.0       # Normalize tilt to [0, 1]
        
        height_labels = np.array(labels[1]).astype(np.float32)
        height_labels = (height_labels - np.mean(height_labels)) / np.std(height_labels)  # Standardize height

        perimeter_labels = np.array(labels[3]).astype(np.float32)
        perimeter_labels = (perimeter_labels - np.mean(perimeter_labels)) / np.std(perimeter_labels)  # Standardize perimeter

        return images, pointclouds, azimuth_labels, height_labels, tilt_labels, perimeter_labels
    
    def _augment_image(self, image):
        """
        Applies conservative augmentation to a single image.
        """
        # Reshape image to add a batch dimension (required by ImageDataGenerator)
        image = np.expand_dims(image, axis=0)
        
        # Apply augmentation
        aug_iter = self.datagen.flow(image, batch_size=1)
        
        # Get one augmented image from the iterator
        augmented_image = next(aug_iter)[0]  # Remove the batch dimension
        
        return augmented_image
    
    def _augment_training_data(self, images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train, augment_count):
        """
        Augment the training images and replicate the associated point clouds and labels.
        The function returns NumPy arrays of augmented data.
        """
        logging.info(f"Starting data augmentation for {len(images_train)} training images with {augment_count} augmentations each.")
        
        augmented_images = []
        augmented_pointclouds = []
        augmented_azimuth = []
        augmented_height = []
        augmented_tilt = []
        augmented_perimeter = []

        for idx, (image, pointcloud, azimuth, height, tilt, perimeter) in enumerate(zip(images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train)):
            logging.info(f"Augmenting image {idx + 1}/{len(images_train)}")

            # Apply augmentation on the image
            for augment_idx in range(augment_count):
                augmented_image = self._augment_image(image)  # Augment the image
                augmented_images.append(augmented_image)

                # Replicate the associated pointcloud and labels
                augmented_pointclouds.append(pointcloud)
                augmented_azimuth.append(azimuth)
                augmented_height.append(height)
                augmented_tilt.append(tilt)
                augmented_perimeter.append(perimeter)

                logging.info(f" - Augmented image {idx + 1}, version {augment_idx + 1}/{augment_count}")

        logging.info(f"Data augmentation complete. {len(augmented_images)} images generated.")
        
        # Convert all augmented data to NumPy arrays before returning
        return (np.array(augmented_images), np.array(augmented_pointclouds), 
                np.array(augmented_azimuth), np.array(augmented_height), 
                np.array(augmented_tilt), np.array(augmented_perimeter))