import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base_cnn_model import BaseCNNModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleRoofCNNModel(BaseCNNModel):
    def __init__(self, max_pointcloud_length=8196, augment=False, augmentation_count=2):
        # Call the base class constructor and pass the required parameters
        super().__init__(max_pointcloud_length, augment, augmentation_count)

        self.model = self._build_model()

    def _build_model(self):
        # Image input branch using ResNet50 (pretrained)
        image_input = layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='image_input')
        base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
        base_output = base_model(image_input)  # Aerial image passed through ResNet50
        image_x = layers.GlobalAveragePooling2D()(base_output)  # Pooling to condense image features

        # Point cloud input branch
        pointcloud_input = layers.Input(shape=(self.max_pointcloud_length, 3), dtype=tf.float32, name='pointcloud_input')
        pointcloud_x = layers.Conv1D(64, kernel_size=1, activation='relu')(pointcloud_input)
        pointcloud_x = layers.BatchNormalization()(pointcloud_x)
        pointcloud_x = layers.Conv1D(128, kernel_size=1, activation='relu')(pointcloud_x)
        pointcloud_x = layers.GlobalAveragePooling1D()(pointcloud_x)  # Pooling to condense point cloud features

        # Combine features from both branches
        combined = layers.Concatenate()([image_x, pointcloud_x])
        
        # Dense layers to process the combined features
        combined_x = layers.Dense(64, activation='relu')(combined)
        #combined_x = layers.Dense(32, activation='relu')(combined_x)
        
        # Output layers with increased L2 regularization for early epoch stabilization
        azimuth_output = layers.Dense(1, name='azimuth', kernel_regularizer=tf.keras.regularizers.l2(0.05))(combined_x)
        height_output = layers.Dense(1, name='height', kernel_regularizer=tf.keras.regularizers.l2(0.05))(combined_x)
        tilt_output = layers.Dense(1, name='tilt', kernel_regularizer=tf.keras.regularizers.l2(0.03))(combined_x)
        perimeter_output = layers.Dense(1, name='perimeter', kernel_regularizer=tf.keras.regularizers.l2(0.1))(combined_x)

        # Define the final model with both inputs and four outputs
        model = tf.keras.Model(inputs=[image_input, pointcloud_input], 
                               outputs=[azimuth_output, height_output, tilt_output, perimeter_output])
        return model

    def train(self, images, pointclouds, labels, checkpoint_dir='checkpoints', title=None):
        # Normalize images, point clouds, and labels using the custom normalize_data function
        images, pointclouds, azimuth_labels, height_labels, tilt_labels, perimeter_labels = self._normalize_data(images, pointclouds, labels, resnet50=True)
        
        # Split the data into training and validation sets
        images_train, images_val, pointclouds_train, pointclouds_val, azimuth_train, azimuth_val, height_train, height_val, tilt_train, tilt_val, perimeter_train, perimeter_val = train_test_split(
            images, pointclouds, azimuth_labels, height_labels, tilt_labels, perimeter_labels, test_size=0.2, random_state=42
        )
        
        # If augmentation is enabled, augment the training data
        if self.augment:
            images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train = self._augment_training_data(
                images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train, self.augment_count)
            
        loss_weights = {'azimuth': 0.2, 'height': 1.5, 'tilt': 1.5, 'perimeter': 0.3}

        # Compile the model with multiple outputs
        self.model.compile(optimizer='adam', 
                           loss={'azimuth': 'mse', 'height': 'mse', 'tilt': 'mse', 'perimeter': 'mse'}, 
                           loss_weights=loss_weights)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define the checkpoint callback to save models during training
        checkpoint_path = os.path.join(checkpoint_dir, 'simple_roof_cnn_epoch_{epoch:02d}_loss_{loss:.4f}.h5')
        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            monitor='val_loss',  # Track validation loss for model checkpointing
            verbose=1, 
            save_best_only=True,  # Save the best models based on validation loss
            mode='auto'
        )

        # Define callbacks: early stopping and learning rate reduction
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        
        # Train the model
        history = self.model.fit(
            [images_train, pointclouds_train], 
            [azimuth_train, height_train, tilt_train, perimeter_train],
            validation_data=([images_val, pointclouds_val], [azimuth_val, height_val, tilt_val, perimeter_val]),
            epochs=60, batch_size=32, 
            callbacks=[early_stopping, reduce_lr]
            # callbacks=[checkpoint,early_stopping, reduce_lr]
        )
        
        return history

