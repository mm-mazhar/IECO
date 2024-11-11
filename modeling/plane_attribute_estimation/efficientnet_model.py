import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from visualization import Visualization
from base_cnn_model import BaseCNNModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EfficientNetRoofCNNModel(BaseCNNModel):
    def __init__(self, max_pointcloud_length=8196, augment=False, augmentation_count=2):
        # Call the base class constructor and pass the required parameters
        super().__init__(max_pointcloud_length, augment, augmentation_count)

        self.model = self._build_model()

    def _build_model(self):
        # Define image input
        image_input = Input(shape=(224, 224, 3), name='image_input')

        # EfficientNetB0 Backbone
        efficient_net = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)

        # Flatten output of EfficientNet
        image_features = Flatten()(efficient_net.output)

        # Define point cloud input
        pointcloud_input = Input(shape=(self.max_pointcloud_length, 3), name='pointcloud_input')

        # Process point cloud input (you can adjust this part based on how you process the point cloud)
        pointcloud_features = Dense(128, activation='relu')(pointcloud_input)
        pointcloud_features = layers.BatchNormalization()(pointcloud_features) 
        pointcloud_features = Flatten()(pointcloud_features)

        # Concatenate image and point cloud features
        combined_features = Concatenate()([image_features, pointcloud_features])
       # Dense layers to process the combined features
        combined_x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.02))(combined_features)
        combined_x = layers.BatchNormalization()(combined_x)  # Add BatchNormalization
        combined_x = tf.keras.layers.Activation('relu')(combined_x)

        # Output layers
        azimuth_output = layers.Dense(64, activation='relu')(combined_x)
        azimuth_output = layers.Dense(1, name='azimuth', kernel_regularizer=tf.keras.regularizers.l2(0.001))(azimuth_output)

        height_output = layers.Dense(1, name='height', kernel_regularizer=tf.keras.regularizers.l2(0.05))(combined_x)
        tilt_output = layers.Dense(1, name='tilt', kernel_regularizer=tf.keras.regularizers.l2(0.03))(combined_x) 
        perimeter_output = layers.Dense(1, name='perimeter', kernel_regularizer=tf.keras.regularizers.l2(0.2))(combined_x)

        #tilt_output = layers.Dense(1, name='tilt', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined_x)
        #perimeter_output = layers.Dense(1, name='perimeter', kernel_regularizer=tf.keras.regularizers.l2(0.1))(combined_x)

        # Combine the outputs
        model = Model(inputs=[image_input, pointcloud_input], outputs=[azimuth_output, tilt_output, height_output, perimeter_output])

        return model

    def train(self, images, pointclouds, labels, checkpoint_dir='checkpoints'):
        # Normalize images, point clouds, and labels using the custom normalize_data function
        images, pointclouds, azimuth_labels, height_labels, tilt_labels, perimeter_labels = self._normalize_data(images, pointclouds, labels, resnet50=False)
        
        # Split the data into training and validation sets
        images_train, images_val, pointclouds_train, pointclouds_val, azimuth_train, azimuth_val, height_train, height_val, tilt_train, tilt_val, perimeter_train, perimeter_val = train_test_split(
            images, pointclouds, azimuth_labels, height_labels, tilt_labels, perimeter_labels, test_size=0.2, random_state=42
        )
        
        # If augmentation is enabled, augment the training data
        if self.augment:
            images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train = self._augment_training_data(
                images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train, self.augment_count)
            
        #loss_weights = {'azimuth': 0.2, 'height': 1.5, 'tilt': 1.9, 'perimeter': 0.3}
        loss_weights = {'azimuth': 0.2, 'height': 1.5, 'tilt': 2.0, 'perimeter': 0.7}

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        self.model.compile(optimizer=optimizer, 
                           loss={'azimuth': 'mse', 'height': 'mse', 'tilt': 'mse', 'perimeter': 'mse'}, 
                           loss_weights=loss_weights)

         # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define the checkpoint callback to save models during training
        checkpoint_path = os.path.join(checkpoint_dir, 'efficient_roof_cnn_epoch_{epoch:02d}_loss_{loss:.4f}.h5')
        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            monitor='val_loss',  # Track validation loss for model checkpointing
            verbose=1, 
            save_best_only=True,  # Save the best models based on validation loss
            mode='auto'
        )

        # Define callbacks: early stopping and learning rate reduction
        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
        #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-4, verbose=1)
        early_stopping_tilt = tf.keras.callbacks.EarlyStopping(monitor='val_tilt_loss', patience=8, restore_best_weights=True, verbose=1)
        early_stopping_perimeter = tf.keras.callbacks.EarlyStopping(monitor='val_perimeter_loss', patience=8, restore_best_weights=True, verbose=1)

        reduce_lr_tilt = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_tilt_loss', factor=0.2, patience=3, min_lr=1e-4, verbose=1)
        reduce_lr_perimeter = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_perimeter_loss', factor=0.2, patience=3, min_lr=1e-4, verbose=1)

        
        # Train the model
        history = self.model.fit(
            [images_train, pointclouds_train], 
            [azimuth_train, height_train, tilt_train, perimeter_train],
            validation_data=([images_val, pointclouds_val], [azimuth_val, height_val, tilt_val, perimeter_val]),
            epochs=80, batch_size=32, 
            callbacks=[early_stopping_tilt, reduce_lr_tilt, early_stopping_perimeter, reduce_lr_perimeter]
            #callbacks=[checkpoint,early_stopping, reduce_lr]
        )
        
        return history

