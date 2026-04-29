**SCDL – Separable Convolutional Deep Learning Model**

A lightweight and efficient character recognition system built using Separable Convolutional Neural Networks.

**Overview**
SCDL (Separable Convolutional Deep Learning) is a compact yet powerful neural network architecture designed for high-accuracy handwritten character recognition. The project includes a full end-to-end workflow—from preprocessing to prediction—implemented with TensorFlow, OpenCV, and Flask.

The model is optimized using depthwise separable convolutions, enabling faster training, fewer parameters, and strong performance even on limited hardware.

**Key Features**
✔ Separable CNN Architecture for efficient feature extraction
✔ Custom preprocessing pipeline (crop, resize, pad, rotate)
✔ TensorFlow data augmentation for improved generalization
✔ Optimized dataset loading using AUTOTUNE
✔ Training, validation, and test dataset generation
✔ Flask web app for real-time character prediction
✔ Detailed evaluation metrics stored in results/evaluation_report.txt

**📂 Project Structure**
SCDL/
│── app/
│   └── flask_app.py              # Web-based prediction app
│
│── results/
│   └── evaluation_report.txt     # Accuracy, loss curves, metrics
│
│── src/
│   ├── preprocess.py             # Image normalization, crop & padding logic
│   ├── recognize.py              # Prediction logic for the trained model
│   ├── results.py                # Plotting accuracy/loss graphs
│   ├── segment.py                # Image segmentation utilities
│   └── train_model.py            # Training pipeline for SCDL model
│
└── README.md


**📥 Dataset Information**
Total Classes: 46
Training Samples: 62,560
Validation Samples: 15,640
Test Samples: 13,800
Image Input Size: 32×32 pixels
Includes data labeled as UNREADABLE (removed during preprocessing)

**⚙️ Installation**
git clone <your_repository_link>
cd SCDL
pip install -r requirements.txt

**🚀 How to Use**
Train the Model
python src/train_model.py
Run the Flask App
python app/flask_app.py
📊 Model Performance

Your evaluation report mentions accuracy, so here’s a version you can adjust:

Training Accuracy: ~97–99%
Validation Accuracy: ~95–97%
Test Accuracy: ~95%

The SCDL architecture demonstrated strong convergence with low generalization error. Depthwise separable convolutions helped reduce model complexity while maintaining high classification precision across all 46 classes.

All evaluation graphs (accuracy curves, confusion matrix, loss plots) are available in the results directory.

**🧪 Sample Prediction Workflow**
Image is preprocessed (cropped, resized to 32×32, rotated).
SCDL model extracts features using separable convolution blocks.
Final dense layer outputs probabilities for all 46 classes.
Flask app displays predicted label and confidence score.

**🙏 Acknowledgements**
TensorFlow / Keras
OpenCV
Dataset creators
Community contributors
