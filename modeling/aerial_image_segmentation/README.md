# Aerial Image Segmentation using SAMv2
This project uses the Segment Anything Model (SAM) v2 for automated roof segmentation.

## Steps
- Download Model Checkpoints: Download pre-trained SAMv2 model checkpoints to use as the starting point for fine-tuning.  
- Prepare Dataset: Generate masks for each image to be used in model training.  
- Data Processing: Resize images and masks to a standard size and extract key points within each roof segment.  
- Fine-Tune the Model: Train the model on the prepared dataset, saving fine-tuned weights periodically.  
- Inference and Segmentation: Load a test image, use key points for mask prediction, and generate a segmentation map by combining predicted masks.  

## Output

We achieved a training accuracy of **0.57** IoU after training for **1000** steps.  
**Inference Output**
![inference](https://github.com/OmdenaAI/IECO/blob/main/modeling/aerial_image_segmentation/inference.png)
