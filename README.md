# Medical posture detection model
Medical posture detection model: a TensorFlow end-to-end ML pipeline for 5-class human posture recognition (left/right/supine/prone/oob). Implements transfer learning on MobileNetV2 &amp; ResNet50, with data augmentation, model training, evaluation and visual inference. Reproducible code.

## Table of Contents
- [Key Features](#key-features)
- [Dataset Structure](#dataset-structure)
- [End-to-End ML Pipeline](#end-to-end-ml-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Model Evaluation](#model-evaluation)
- [Results Visualization](#results-visualization)
- [Project File Structure](#project-file-structure)
- [Reproducibility](#reproducibility)
- [Future Enhancements](#future-enhancements)

## Key Features
- 5-class human posture classification: `left`, `right`, `supine`, `prone`, `oob` (out of bed)
- Transfer learning with **ImageNet pre-trained** MobileNetV2 and ResNet50 for feature extraction
- Targeted data augmentation for training set to prevent overfitting and boost generalization
- End-to-end pipeline integrating data processing, model building, training and inference
- Comprehensive model evaluation with quantitative metrics and qualitative visualization
- Side-by-side performance comparison of MobileNetV2 and ResNet50
- Lightweight MobileNetV2 implementation optimized for edge device deployment
- Fully reproducible results with fixed random seeds and standardized hyperparameters
- Visual single-image inference with confidence scores for intuitive validation

## Dataset Structure
The dataset is split into **TRAIN** and **VALIDATION** subsets with identical directory structures for all 5 posture classes. All images are resized to 224x224 (RGB channels) for model ingestion.

### Directory Layout
```
data/
├── TRAIN/
│   ├── left/
│   ├── right/
│   ├── supine/
│   ├── prone/
│   └── oob/
└── VALIDATION/
    ├── left/
    ├── right/
    ├── supine/
    ├── prone/
    └── oob/
```

### File Naming Convention
All image files follow a consistent naming pattern for easy class identification:
```
<posture_class>_<unique_numeric_id>.jpg
# Example: left_15527489.jpg, oob_98765432.jpg, supine_12345678.jpg
```

## End-to-End ML Pipeline
The pipeline follows standard computer vision best practices and is split into four core stages, with consistent input/output standards for seamless integration.

### 1. Data Processing
All raw image data is preprocessed to ensure numerical stability and model compatibility—**augmentation is only applied to the training set** for unbiased validation.
- **Rescaling**: Normalize pixel values from `[0,255]` to `[0,1]` using a scaling factor of `1/255`
- **Data Augmentation**: Random transformations for training set:
  - Rotation (±20°)
  - Horizontal/vertical translation (±20%)
  - Shear transformation (±20%)
  - Zoom in/out (±20%)
  - Horizontal & vertical flips
  - Filled missing pixels with `nearest` fill mode
- **Batching**: Batches of 32 samples (balances GPU memory usage and training efficiency)
- **Label Encoding**: One-hot categorical encoding for multi-class classification tasks
- **Validation Set**: No augmentation, only rescaling—ensures pure, objective performance evaluation

### 2. Model Selection
Two state-of-the-art CNN architectures are implemented and compared, both using transfer learning with frozen base layers to retain general visual features from ImageNet.
- **MobileNetV2**: Lightweight, depth-wise separable convolution—prioritizes speed and low computation for edge deployment
- **ResNet50**: Deep residual network with skip connections—prioritizes complex feature extraction for high accuracy

### 3. Model Training
Unified training setup for both models to ensure a **fair performance comparison**, with adaptive optimization and multi-class loss function.

### 4. Model Evaluation & Inference
Combines quantitative statistical metrics and qualitative visual validation to assess model performance across all posture classes.

## Installation
### Prerequisites
- Python 3.8+
- TensorFlow 2.10+ (Keras integrated)
- Basic ML/visualization Python libraries

### Install Dependencies
Install all required packages via `pip` (run in terminal/command prompt):
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Requirements File
Create a `requirements.txt` file in the project root with the following content for one-click installation:
```txt
tensorflow>=2.10.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```
Install from the requirements file:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/Mia-Yiming/Medical-posture-detection-model.git
cd Medical-posture-detection-model
```

### 2. Prepare the Dataset
- Create a `data/` folder in the project root directory
- Organize your posture images into the [directory layout](#directory-layout) (TRAIN/VALIDATION with 5 class subfolders)
- Ensure all images are in JPG format and follow the [file naming convention](#file-naming-convention)

### 3. Run the Full Pipeline
Execute the main Python script to start data processing, model training, evaluation and visualization:
```bash
python main.py
```

### 4. View Results
- Real-time training logs (loss/accuracy per epoch) are printed to the console
- Classification reports (precision/recall/F1-score) for both models are printed to the console
- All visualizations (confusion matrices, training curves, inference plots) open interactively
- No additional setup required—all outputs are generated automatically

## Model Architecture
Both models use a **two-component design**: a **frozen pre-trained base model** for feature extraction and a **custom top classifier** for posture prediction. Input shape for all models is fixed at `(224, 224, 3)` (height, width, RGB channels).

### MobileNetV2 (Edge-Optimized)
Lightweight architecture with minimal computation—ideal for edge device deployment:
```
Input (224x224x3) → MobileNetV2 (frozen, ImageNet pre-trained) → Global Average Pooling → Dense(128, ReLU) → Softmax(5) → Output (5-class probability vector)
```

### ResNet50 (Accuracy-Optimized)
Deeper architecture with residual skip connections—ideal for complex feature extraction:
```
Input (224x224x3) → ResNet50 (frozen, ImageNet pre-trained) → Global Average Pooling → Dense(256, ReLU) → Softmax(5) → Output (5-class probability vector)
```

**Key Design Choice**: Base model layers are frozen to avoid overfitting and leverage pre-trained general visual features—only the custom top classifier layers are trained on the posture dataset.

## Training Configuration
A unified training setup is used for both MobileNetV2 and ResNet50 to eliminate experimental bias and enable direct performance comparison. All hyperparameters are fixed as constants in the code for easy modification.

### Core Training Hyperparameters
| Parameter               | Value                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| Optimizer               | Adam (adaptive learning rate—no manual tuning required)               |
| Loss Function           | Categorical Crossentropy (standard for multi-class classification)    |
| Training Epochs         | 10 (trade-off between model convergence and overfitting)              |
| Batch Size              | 32                                                                    |
| Input Image Size        | (224, 224)                                                            |
| Pixel Rescaling Factor  | 1/255                                                                 |
| Random Seed             | 42 (for full reproducibility)                                         |
| Class Mode              | Categorical (one-hot encoded labels)                                  |
| Shuffling               | Enabled (training set) / Disabled (validation set)                    |

### Monitored Metrics
- Training Loss & Accuracy (per epoch)
- Validation Loss & Accuracy (per epoch)
- Real-time loss/accuracy updates printed to the console

## Model Evaluation
Comprehensive **quantitative** and **qualitative** evaluation is implemented to assess model performance—metrics are calculated per class and overall to identify weak points in posture classification.

### Quantitative Evaluation
1. **Classification Report**
   - Per-class precision, recall, and F1-score
   - Per-class sample support (number of images)
   - Overall model accuracy
   - Zero-division handling for rare classes
2. **Confusion Matrix**
   - Quantifies class-wise misclassifications
   - Count of true vs. predicted labels for all 5 posture classes
   - Normalized and annotated for easy interpretation

### Qualitative Evaluation
1. **Overfitting Detection**
   - Comparison of training/validation loss/accuracy trends
   - Large gaps between training and validation metrics indicate overfitting
2. **Single-Image Inference**
   - Predicted posture label + confidence score (0-1)
   - True posture label for direct comparison
   - Side-by-side prediction from MobileNetV2 and ResNet50
   - Visual validation of model predictions on raw images

## Results Visualization
All visualizations are generated with **Matplotlib** and **Seaborn**—interactive plots open automatically after model training and evaluation, with clear labels and titles for easy interpretation.

### 1. Confusion Matrix Heatmap
- One heatmap for MobileNetV2 and one for ResNet50
- Annotated with raw sample counts
- Blue color palette for better readability
- X-axis: Predicted Pose | Y-axis: True Pose

### 2. Training Curve Comparison
- 2x2 subplot for side-by-side loss/accuracy comparison
  - Top-left: Training/Validation Loss (MobileNetV2 + ResNet50)
  - Top-right: Training/Validation Accuracy (MobileNetV2 + ResNet50)
- Dashed lines for ResNet50, solid lines for MobileNetV2
- Epochs (x-axis) vs. Metric Value (y-axis)
- Legend for clear model identification

### 3. Single-Image Inference Plots
- 3 random validation images selected for visual inference (fixed seed for reproducibility)
- Original image with no modifications
- True posture label displayed at the top
- Predicted label + confidence score for both models (text boxes with white background)
- Consistent layout for all inference plots

## Project File Structure
Minimal, clean file structure for easy navigation and maintenance—all core logic is contained in a single main script for quick execution.
```
Medical-posture-detection-model/
├── data/                  # Dataset root (TRAIN/VALIDATION subfolders)
├── main.py                # Full end-to-end ML pipeline (all core logic)
├── README.md              # Project documentation (this file)
└── requirements.txt       # One-click dependency installation
```

## Reproducibility
All random processes are **seeded** to ensure **identical results on every run**—no randomness in data processing, model training or inference.
- TensorFlow random seed
- NumPy random seed
- Python `random` module seed
- Data generator shuffle seed (fixed at 42)
- Disabled shuffling for validation set (consistent sample order)
- All hyperparameters fixed as constants in the code
- Same training/validation split for all model runs

## Future Enhancements
This project provides a robust baseline for medical posture detection and can be extended with the following improvements to boost performance and real-world applicability:
- **Model Fine-Tuning**: Unfreeze top layers of pre-trained base models and train with a small learning rate for dataset-specific feature adaptation
- **Hyperparameter Tuning**: Optimize batch size, epochs, learning rate and dense layer units via GridSearchCV/RandomizedSearchCV
- **Test Set Integration**: Add a held-out test set for final, unbiased performance assessment
- **TensorFlow Lite Conversion**: Quantize MobileNetV2 for edge device deployment (embedded medical devices, IoT)
- **Class Imbalance Handling**: Apply SMOTE/undersampling if the dataset has uneven class distribution
- **Advanced Augmentation**: Add random brightness/contrast, noise injection and random cropping for better generalization
- **Ensemble Learning**: Combine predictions from MobileNetV2 and ResNet50 for higher overall accuracy
- **Model Checkpointing**: Save the best model (by validation accuracy) during training to avoid retraining
- **TensorBoard Integration**: Interactive training visualization and metric tracking
- **Batch Inference**: Add support for bulk image prediction and result export (CSV/JSON)
- **Error Analysis**: Deep dive into misclassified images to identify dataset/model weaknesses
- **Data Preprocessing**: Add noise reduction and image normalization for low-quality input images

## Acknowledgements
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning framework for model building and training
- [ImageNet](https://www.image-net.org/) - Pre-trained weights for transfer learning feature extraction
- [Scikit-learn](https://scikit-learn.org/) - Classification metrics and confusion matrix generation
- [Matplotlib/Seaborn](https://matplotlib.org/) - Interactive data visualization and plot generation
- [NumPy](https://numpy.org/) - Numerical computing and array manipulation for image processing
