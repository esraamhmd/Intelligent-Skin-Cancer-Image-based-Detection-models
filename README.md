# Intelligent Skin Cancer Image-based Detection Models

A comprehensive machine learning and deep learning project implementing state-of-the-art models for automated skin cancer detection and classification from dermoscopic images. This project combines traditional machine learning approaches with advanced deep learning techniques to assist healthcare professionals in early diagnosis and improve patient outcomes.

## 🎯 Project Overview

Skin cancer is one of the most common types of cancer worldwide, with melanoma being responsible for approximately 75% of skin cancer deaths despite being the least common type. Early detection significantly improves treatment outcomes, with survival rates reaching up to 98% when caught in early stages. This project implements both machine learning and deep learning frameworks to create intelligent diagnostic tools that can classify skin lesions with high accuracy.

## ✨ Key Features

- **Dual Framework Approach**: Implementation of both traditional machine learning and deep learning methodologies
- **Feature Fusion**: Novel approach combining features from multiple CNN architectures (MobileNetV3, ResNet50, DenseNet121)
- **Ensemble Learning**: Advanced ensemble methods combining predictions from multiple models
- **Multi-class & Binary Classification**: Support for both detailed lesion type classification and benign/malignant detection
- **Comprehensive Preprocessing**: Advanced image preprocessing pipeline with multiple augmentation techniques
- **Benchmark Datasets**: Evaluation on HAM10000 and ISIC datasets with detailed performance analysis
- **High Performance**: Achieved 96.1% accuracy on multi-class classification and 93.56% on binary classification

## 📊 Dataset Information

### HAM10000 Dataset
- **Total Images**: 10,015 high-quality dermoscopic images
- **Source**: ViDIR Group, Department of Dermatology, Medical University of Vienna
- **Collection**: Images from Austria and Australia ensuring diversity

**Dataset Link**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

| Lesion Type | Abbreviation | Number of Images | Description |
|-------------|--------------|------------------|-------------|
| Melanoma | MEL | 1,113 | A highly malignant skin cancer |
| Melanocytic Nevi | NV | 6,705 | Benign pigmented moles |
| Basal Cell Carcinoma | BCC | 514 | A common type of skin cancer |
| Actinic Keratosis | AK | 327 | Precancerous lesions caused by sun exposure |
| Benign Keratosis | BKL | 1,099 | Includes seborrheic keratosis and lichen planus-like keratosis |
| Vascular Lesions | VASC | 142 | Includes hemangiomas and pyogenic granulomas |
| Dermatofibroma | DF | 115 | A rare benign fibrous lesion |


### ISIC Archive Dataset
- **Total Images**: 28,000 high-quality dermoscopic images
- **Source**: International Skin Imaging Collaboration (ISIC)
- **Collection**: Multiple institutions and research organizations worldwide
- **Balanced Distribution**: 4,000 images per class for 7 lesion types

**Dataset Link**: [ISIC 2019 Balanced Dataset on Kaggle](https://www.kaggle.com/datasets/shadmansobhan/isic-2019-balanced-dataset-4000-images-per-class)

## 🏗️ Framework Architectures

### Machine Learning Framework

The machine learning approach implements a sophisticated feature extraction and fusion pipeline:

**Pipeline Stages:**
1. **Dataset Input**: HAM10000 and ISIC datasets
2. **Data Augmentation**: Rotation, vertical flip, horizontal flip, affine transformation
3. **Preprocessing**: Resizing to standard dimensions and normalization
4. **Feature Extraction Fusion**: 
   - MobileNetV3: 1,280-dimensional features
   - ResNet50: 2,048-dimensional features  
   - DenseNet121: 1,024-dimensional features
   - **Combined**: 4,352-dimensional fused feature vector
5. **Machine Learning Classifiers**: Random Forest (RF) and Support Vector Machine (SVM)
  
<div align="center">
  <img src="https://github.com/esraamhmd/Intelligent-Skin-Cancer-Image-based-Detection-models/blob/main/mlframework.png" width="900" alt="Deep learning framework"/>
  </div>

### Deep Learning Framework

The deep learning approach uses an ensemble of three powerful CNN architectures:

**Pipeline Stages:**
1. **Dataset Input**: HAM10000 and ISIC datasets
2. **Data Augmentation**: Advanced augmentation techniques
3. **Preprocessing**: Resizing and normalization
4. **Ensemble Deep Learning**: 
   - MobileNetV3: Optimized for efficiency
   - ResNet50: Deep feature learning with residual connections
   - DenseNet121: Dense connectivity for feature reuse
5. **Ensemble Prediction**: Voting/averaging of predictions

<div align="center">
  <img src="https://github.com/esraamhmd/Intelligent-Skin-Cancer-Image-based-Detection-models/blob/main/dlframework.png" width="900" alt="Deep learning framework"/>
  </div>

## 🔧 Data Preprocessing Pipeline

### Data Augmentation Techniques
1. **Rotation**: Random rotation to handle different orientations
2. **Horizontal Flip**: Mirror images to increase data variability
3. **Vertical Flip**: Vertical mirroring for comprehensive augmentation
4. **Affine Transformation**: Shape and position modifications

### Preprocessing Steps
1. **Normalization**: Pixel value scaling to [0,1] range
2. **Resizing**: Standardization to consistent dimensions (224×224 or 299×299)
3. **Quality Enhancement**: CLAHE and contrast adjustments

## 🤖 Model Architectures

### Deep Learning Models

#### 1. MobileNetV3
- **Parameters**: 3.2M
- **Model Size**: 12.3 MB
- **Optimization**: Neural Architecture Search (NAS) + manual refinements
- **Features**: Inverted residual blocks, squeeze-and-excitation blocks
- **Use Case**: Mobile and edge device deployment

#### 2. ResNet50
- **Parameters**: 23.5M
- **Model Size**: 89.7 MB
- **Core Innovation**: Residual connections solving vanishing gradient problem
- **Architecture**: 50 layers with skip connections
- **Strengths**: Deep feature learning, proven medical imaging performance

#### 3. DenseNet121
- **Parameters**: 7.0M
- **Core Innovation**: Dense connectivity where each layer connects to all preceding layers
- **Benefits**: Maximum information flow, feature reuse, gradient flow optimization
- **Medical Imaging**: Excellent for fine-grained texture analysis

### Machine Learning Classifiers

#### Support Vector Machine (SVM)
```python
# Hyperparameters
kernel: 'rbf' (Radial Basis Function)
C: 1.0 (Regularization parameter)
gamma: 'scale'
probability: True
max_iter: 400
```

#### Random Forest (RF)
```python
# Hyperparameters
n_estimators: 100
max_depth: 10
min_samples_split: 10
min_samples_leaf: 4
bootstrap: True
max_features: 'sqrt'
```

## 📈 Experimental Results

### Deep Learning Results

#### HAM10000 Dataset Performance

**Multi-class Classification (96.1% Accuracy):**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| AKIEC | 96.20% | 99.60% | 97.90% |
| BCC | 97.20% | 98.60% | 97.90% |
| BKL | 94.60% | 89.70% | 92.10% |
| DF | 99.10% | 100% | 99.60% |
| MEL | 94.00% | 90.60% | 92.30% |
| NV | 91.90% | 94.50% | 93.20% |
| VASC | 99.90% | 100% | 100% |

**Binary Classification (93.56% Accuracy):**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 90.90% | 96.80% | 93.80% |
| Malignant | 96.60% | 90.30% | 93.30% |

#### ISIC Dataset Performance

**Multi-class Classification (81.2% Accuracy):**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| AK | 77.90% | 85.20% | 81.40% |
| BCC | 77.90% | 76.20% | 77.00% |
| BKL | 71.50% | 68.20% | 69.80% |
| DF | 88.90% | 98.80% | 93.60% |
| MEL | 77.70% | 65.00% | 70.80% |
| NV | 76.10% | 76.00% | 76.10% |
| VASC | 96.40% | 99.20% | 97.80% |

**Binary Classification (91.4% Accuracy):**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 88.30% | 95.50% | 91.80% |
| Malignant | 95.10% | 87.30% | 91.10% |

### Machine Learning Results

#### HAM10000 Dataset

**Random Forest Binary Classification:**
- **Accuracy**: 89.43%
- **Benign**: Precision 83.89%, Recall 97.61%, F1-Score 90.23%
- **Malignant**: Precision 97.14%, Recall 81.24%, F1-Score 88.48%

**SVM Binary Classification:**
- **Accuracy**: 80.17%
- **Benign**: Precision 87.77%, Recall 70.13%, F1-Score 77.96%
- **Malignant**: Precision 75.11%, Recall 90.22%, F1-Score 81.97%

#### ISIC Dataset

**Random Forest Binary Classification:**
- **Accuracy**: 83.45%
- **Benign**: Precision 78.13%, Recall 92.90%, F1-Score 84.88%
- **Malignant**: Precision 91.24%, Recall 74.00%, F1-Score 81.72%

### Comprehensive Performance Comparison

| Dataset | Classification Type | Machine Learning Best | Deep Learning Ensemble | Improvement |
|---------|-------------------|---------------------|----------------------|-------------|
| HAM10000 | Binary | 89.43% (RF) | 93.56% | +4.13% |
| HAM10000 | Multi-class | 76.57% (SVM) | 96.10% | +19.53% |
| ISIC | Binary | 83.45% (RF) | 91.40% | +7.95% |
| ISIC | Multi-class | 71.83% (SVM) | 81.20% | +9.37% |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (for GPU acceleration)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/esraamhmd/Intelligent-Skin-Cancer-Image-based-Detection-models.git
cd Intelligent-Skin-Cancer-Image-based-Detection-models

# Create virtual environment
python -m venv skin_cancer_env
source skin_cancer_env/bin/activate  # On Windows: skin_cancer_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```bash
torch>=1.10.0
torchvision>=0.11.0
tensorflow>=2.6.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
albumentations>=1.1.0
```

## 💻 Usage

### Training Machine Learning Models
```bash
# Train with feature fusion approach
python train_ml.py --dataset HAM10000 --classifier RF --feature_fusion True

# Train SVM classifier
python train_ml.py --dataset ISIC --classifier SVM --feature_fusion True
```

### Training Deep Learning Models
```bash
# Train individual models
python train_dl.py --model mobilenetv3 --dataset HAM10000 --epochs 50
python train_dl.py --model resnet50 --dataset ISIC --epochs 50
python train_dl.py --model densenet121 --dataset HAM10000 --epochs 50

# Train ensemble model
python train_ensemble.py --models mobilenetv3,resnet50,densenet121 --dataset HAM10000
```

### Evaluation
```bash
# Evaluate machine learning models
python evaluate_ml.py --model_path checkpoints/rf_fusion_model.pkl --test_data data/test

# Evaluate deep learning models
python evaluate_dl.py --model_path checkpoints/ensemble_model.pth --test_data data/test
```

### Making Predictions
```bash
# Single image prediction
python predict.py --image path/to/lesion.jpg --model_type ensemble --model_path checkpoints/best_ensemble.pth

# Batch prediction
python batch_predict.py --input_dir images/ --output_file predictions.csv --model_type ensemble
```

## 🔬 Technical Innovation

### Feature Fusion Approach
The machine learning framework implements a novel feature fusion technique:
- **Multi-scale Feature Extraction**: Combines features from three different CNN architectures
- **Dimensional Fusion**: Creates a comprehensive 4,352-dimensional feature vector
- **Complementary Strengths**: Leverages MobileNetV3's efficiency, ResNet50's depth, and DenseNet121's dense connectivity

### Ensemble Deep Learning
The deep learning framework uses advanced ensemble methods:
- **Soft Voting**: Averages probability distributions from multiple models
- **Model Diversity**: Combines architectures with different strengths and characteristics
- **Improved Generalization**: Reduces overfitting and improves robustness

## 📊 Performance Analysis

### Key Findings

1. **Deep Learning Superiority**: Ensemble deep learning models consistently outperform traditional machine learning approaches across all datasets and classification tasks.

2. **Feature Fusion Effectiveness**: The feature fusion approach in machine learning significantly improves performance compared to single-model approaches.

3. **Dataset Dependency**: Performance varies between datasets, with HAM10000 showing higher accuracy due to better class balance.

4. **Binary vs Multi-class**: Binary classification generally achieves higher accuracy than multi-class classification across all approaches.

### Clinical Relevance

- **High Sensitivity**: The ensemble model achieves 96.8% recall for benign cases and 90.3% for malignant cases on HAM10000
- **Low False Positives**: High precision scores reduce unnecessary biopsies and patient anxiety
- **Multi-class Capability**: Detailed lesion type classification aids in treatment planning
- **Robust Performance**: Consistent results across different datasets indicate good generalization

## 🏥 Medical Impact

### Diagnostic Support Features
- **Screening Tool**: Assists dermatologists in identifying suspicious lesions
- **Triage System**: Prioritizes cases requiring immediate attention based on malignancy probability
- **Educational Aid**: Provides detailed classification results for training purposes
- **Telemedicine**: Enables remote skin cancer screening in underserved areas

### Performance vs Clinical Standards
| Method | Sensitivity | Specificity | Accuracy |
|--------|-------------|-------------|----------|
| Dermatologists (Literature) | 89% | 64% | 75-84% |
| Our Ensemble Model | 96.8% | 90.3% | 93.56% |
| Traditional ML (Best) | 97.6% | 81.2% | 89.43% |

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team and Acknowledgments

### Memebers
**Esraa Mahmoud** - Model Developer
- GitHub: [@esraamhmd](https://github.com/esraamhmd)

### Acknowledgments
- **International Skin Imaging Collaboration (ISIC)** for providing the ISIC Archive dataset
- **Medical University of Vienna (ViDIR Group)** for the HAM10000 dataset
- **Medical Advisory Board** for clinical expertise and validation
- **Open Source Community** for deep learning frameworks and tools

## ⚠️ Important Disclaimers

### Medical Disclaimer
**This software is intended for research and educational purposes only.**

- 🔴 **Not for Clinical Diagnosis**: This tool is NOT a substitute for professional medical diagnosis
- 🔴 **No Medical Advice**: Does not provide medical advice or treatment recommendations  
- 🔴 **Consult Professionals**: Always consult qualified dermatologists for medical concerns
- 🔴 **Emergency Cases**: Seek immediate medical attention for urgent skin conditions

### Technical Disclaimer
- Model performance may vary on images different from training data
- Regular model updates and retraining are recommended for optimal performance
- Ensure proper image quality and lighting conditions for best results
- Consider model limitations and potential biases in clinical deployment scenarios

## 📊 Detailed Results Summary

### Overall Framework Performance

| Framework | Dataset | Binary Accuracy | Multi-class Accuracy | Best Performance Metric |
|-----------|---------|----------------|-------------------|----------------------|
| **Deep Learning Ensemble** | HAM10000 | **93.56%** | **96.10%** | 100% F1-Score (VASC) |
| **Deep Learning Ensemble** | ISIC | **91.40%** | **81.20%** | 97.8% F1-Score (VASC) |
| **Machine Learning (RF)** | HAM10000 | 89.43% | 66.70% | 90.23% F1-Score (Benign) |
| **Machine Learning (SVM)** | HAM10000 | 80.17% | 76.57% | 99.65% F1-Score (VASC) |

### Hardware Requirements

**Minimum System Requirements:**
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 20GB free space
- GPU: Optional for CPU-only training

**Recommended System Requirements:**
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB+
- GPU: NVIDIA RTX 3070 or better (8GB+ VRAM)
- Storage: 100GB+ SSD storage

---

**⭐ If you find this research helpful, please consider starring the repository and citing our work in your research!**
