import os
import logging
from simple_model import SimpleRoofCNNModel
from efficientnet_model import EfficientNetRoofCNNModel
from utils import Utilities
from visualization import Visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(root_dir, efficient, subset_size, augment, both):
    logging.info("Starting Roof CNN model training process")

    visualization = Visualization()

    models = {"ResNet50": SimpleRoofCNNModel(augment=augment)}

    if (both):
        models["EfficientNet"] = EfficientNetRoofCNNModel(augment=augment)
    elif (efficient):
        models = {"EfficientNet": EfficientNetRoofCNNModel(augment=augment)}

    utilities = Utilities()

    images = []
    pointclouds = []
    azimuth_labels = []
    height_labels = []
    tilt_labels = []
    perimeter_labels = []
    
    subdirs = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]
    if subset_size:
        subdirs = subdirs[:subset_size]
    logging.info(f"Processing {len(subdirs)} directories")

    for subdir in subdirs:
        logging.info(f"Processing directory: {subdir}")

        image, pointcloud, label = utilities.process_directory(subdir)

        # If we do not get a label then skip the file
        if (label):
            images.append(image)
            pointclouds.append(pointcloud)
            azimuth_labels.append(label[0])
            height_labels.append(label[1])
            tilt_labels.append(label[2])
            perimeter_labels.append(label[3])

    # Now pass the separate label arrays to the model's training function
    labels = [azimuth_labels, height_labels, tilt_labels, perimeter_labels]

    results = {}
    for key, model in models.items():
        history = model.train(images, pointclouds, labels)
        results[key] = history

    for key, history in results.items():
        # Visualize training performance
        if (augment):
            title = "{} Augmented Training and Validation Losses for All Components".format(key)
        else:
            title = "{} Non-Augmented Training and Validation Losses for All Components".format(key)
            
        visualization.plot_component_losses(history, title=title)

    logging.info("Completed processing")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Roof 3D Reconstruction Model")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subdirectories with google.jpg, dsm.json, and planes.json.")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size to limit the number of directories processed.")
    parser.add_argument('--augment', action='store_true', help='Augment the training dataset by default its False')

    # Create a mutually exclusive group for --both and --efficient
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--efficient', action='store_true', help='Run the efficientnet model by default its False. Note if this is set --both cannot be set')
    group.add_argument('--both', action='store_true', help='Run both models, note if this is set --efficientnet cannot be set')
    
    args = parser.parse_args()
    main(args.root_dir, args.efficient, args.subset_size, args.augment, args.both)
