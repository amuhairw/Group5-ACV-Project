# GESCAM: Gaze Estimation for Classroom Attention Measurement

This repository contains code for training and evaluating gaze estimation models on the GESCAM dataset. The system detects where people in classroom environments are looking based on their head position and appearance.
Overview
The GESCAM system uses a multi-stream architecture to predict gaze targets in classroom settings. The model takes as input:

1. The full classroom image
2. A crop of each person's head
3. A binary mask showing the head's location in the image

From these inputs, the model predicts:

1. A heatmap showing where the person is looking
2. A probability indicating whether the person is looking inside or outside the frame

<video autoplay loop muted>
  <source src="visualization.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Dataset
The GESCAM dataset consists of annotated classroom images with:

Person bounding boxes
Gaze direction polylines
Frame metadata

The dataset loader processes these annotations to create training samples by matching each person to their corresponding gaze target.
File Structure

gescam_customized_dataset.py: Dataset loader for GESCAM annotations
ms_gescam_size_fix.py: MS-GESCAM model architecture
gescam_kaggle_script.py: Training script for Kaggle environment
gescam_validation.py: Validation and visualization script

Model Architecture
The MS-GESCAM model uses a multi-stream architecture:

Head Pathway:

Processes the head crop using a ResNet18 backbone
Extracts features representing head pose and orientation


Scene Pathway:

Processes the full scene image plus head position mask
Identifies potential gaze targets in the environment


Attention Mechanism:

Uses head features to generate an attention map
Focuses the model on relevant regions of the scene


Fusion and Decoding:

Combines attended scene features with head features
Decodes into a heatmap representing gaze location
Predicts whether the person is looking inside or outside the frame



Usage
Installation
bashCopy# Clone repository
git clone https://github.com/yourusername/gescam.git
cd gescam

# Install dependencies
pip install torch torchvision opencv-python matplotlib tqdm scikit-learn
Data Preparation

Organize your dataset:
Copydata/
├── images/
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
└── annotations.xml

The XML annotations should follow the GESCAM format with person bounding boxes and line of sight polylines.

Training
pythonCopypython gescam_kaggle_script.py
This will:

Load and preprocess the dataset
Train the MS-GESCAM model
Save checkpoints and visualizations
Evaluate the model on a validation set

Validation
pythonCopypython gescam_validation.py
This will:

Load a trained model
Evaluate performance using metrics like AUC, distance error, and angular error
Generate visualizations of model predictions
Create an attention heatmap video showing where everyone is looking

Results
Current validation metrics:

AUC: 0.6197 ± 0.1300
Distance Error: 0.4264 ± 0.2001
Angular Error: 0.01° ± 0.01°
In-frame Accuracy: 1.0000

For comparison, state-of-the-art results from the papers:

AUC of 0.921 on GazeFollow dataset
AUC of 0.860 on VideoAttentionTarget dataset
AUC of 0.943 on the full GESCAM dataset

Visualization
The system generates multiple types of visualizations:

Individual predictions:

Scene image with person bounding box
Predicted and ground truth gaze heatmaps
Heatmap overlays on the original image
Error heatmap showing prediction differences


Attention heatmap video:

Combined heatmap showing where everyone in the classroom is looking
Helps identify attention hotspots and patterns



Future Work

Performance improvements:

Train for more epochs
Use more data or better augmentation
Experiment with different architectures


Analysis tools:

Quantify classroom attention patterns
Identify areas of high/low student engagement
Track attention over time


Applications:

Real-time classroom attention monitoring
Educational research tools
Feedback systems for teachers



References

Chong, E., Wang, Y., Ruiz, N., & Rehg, J. M. (2020). Detecting Attended Visual Targets in Video. CVPR 2020.
Mathew, A. M., Khan, A. A., Khalid, T., & Souissi, R. (2024). GESCAM: A Dataset and Method on Gaze Estimation for Classroom Attention Measurement. CVPR Workshop 2024.

License
[Specify your license information here]
Acknowledgments

The GESCAM dataset creators
PyTorch team for the deep learning framework
[Add any other acknowledgments]