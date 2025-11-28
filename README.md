# Crop Disease Detection ML

A machine learning system for detecting diseases in crop leaves using deep learning. This project provides both a command-line interface for training and testing models, as well as a REST API for real-time inference.

## 🌾 Overview

This project identifies 52 different crop diseases across multiple crops including apples, tomatoes, grapes, corn, beans, and more. It uses transfer learning with MobileNetV2 for efficient and accurate classification.

## Features

- **52 Disease Classes**: Detects diseases across 9 crop types
- **Transfer Learning**: Built on MobileNetV2 for optimal performance
- **REST API**: FastAPI-based API for easy integration
- **Interactive Documentation**: Swagger UI for testing
- **Automated Testing**: Verification scripts included

## 📋 Supported Crops and Diseases

- **Apple**: Brown spot, Black spot, Normal leaf
- **Apricot**: Blight leaf disease, Shot hole, Normal
- **Bean**: Fungal leaf disease, Bean rust, Shot hole, Normal leaf
- **Cherry**: Leaf scorch, Brown spot, Purple leaf spot, Shot hole disease, Normal
- **Corn**: Fungal leaf, Gray leaf spot, Holcus leaf spot, Normal
- **Fig**: Blight leaf disease, Brown spot, Rust leaf, Normal
- **Grape**: Anthracnose, Brown spot, Downy mildew, Mites, Powdery mildew, Shot hole, Normal
- **Pear**: Black spot, Fire blight, Normal
- **Tomato**: Bacterial spot, Early blight, Late blight, Fusarium wilt, Verticillium wilt, Spider mites, Leaf curl, Leaf miner, Leaf mold, Septoria leaf, Healthy
- **Walnut**: Anthracnose, Blotch, Shot hole, Leaf gall mite, Normal
- **Lokat**: Leaf spot, Normal
- **Persimmons**: Brown spot

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd bkk_crop_disease_detect_ml

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python src/train.py --data_dir dataset/train --epochs 10 --batch_size 32
```

**Options:**
- `--data_dir`: Path to training dataset (default: `dataset/train`)
- `--epochs`: Number of training epochs (default: `5`)
- `--batch_size`: Batch size (default: `32`)
- `--image_size`: Input image size (default: `224`)
- `--output`: Model save path (default: `models/saved_model`)
- `--fine_tune`: Number of layers to fine-tune (default: `0`)

### Fine-Tuning the Model (Recommended for Better Accuracy)

Fine-tuning unfreezes the last layers of the pre-trained MobileNetV2 model, allowing it to adapt specifically to crop disease features. This significantly improves accuracy.

#### Step 1: Initial Training (if not done)
```bash
python src/train.py --epochs 10 --batch_size 32
```

#### Step 2: Fine-Tune the Model
```bash
python src/train.py --fine_tune 50 --epochs 10
```

This will:
- Load your existing trained model
- Unfreeze the last 50 layers of MobileNetV2
- Continue training with a lower learning rate (1e-5)
- Save the improved model back to `models/saved_model`

#### Fine-Tuning Tips

**Recommended configurations:**

**Quick fine-tuning** (faster, good results):
```bash
python src/train.py --fine_tune 30 --epochs 5
```

**Aggressive fine-tuning** (best accuracy, takes longer):
```bash
python src/train.py --fine_tune 100 --epochs 15
```

**With larger images** (more details, requires more GPU memory):
```bash
python src/train.py --fine_tune 50 --epochs 10 --image_size 384
```

**Best Practices:**
- Always do initial training first (without `--fine_tune`)
- Use smaller learning rates for fine-tuning (automatically set to 1e-5)
- Fine-tuning typically requires fewer epochs (5-15)
- Monitor validation accuracy to avoid overfitting
- More layers fine-tuned = better accuracy but slower training

### Testing the Model

```bash
python src/test.py --data_dir dataset/test --model_path models/saved_model
```


## 🌐 API Usage

### Start the API Server

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Interactive Documentation

Visit `http://127.0.0.1:8000/docs` for Swagger UI where you can test the API directly.

### API Endpoints

#### Health Check
```bash
GET /
```

**Response:**
```json
{
  "message": "Crop Disease Detection API is running"
}
```

#### Predict Disease
```bash
POST /predict
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (JPEG, PNG, etc.)

**Example using cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/leaf_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Crop analyzed successfully",
  "data": {
    "crop": "Apple",
    "disease": "black_spot",
    "confidence": "95.32%"
  }
}
```

### Example in Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("leaf_image.jpg", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Crop: {result['data']['crop']}")
print(f"Disease: {result['data']['disease']}")
print(f"Confidence: {result['data']['confidence']}")
```

## 📁 Project Structure

```
bkk_crop_disease_detect_ml/
├── api/
│   └── main.py              # FastAPI application
├── dataset/
│   ├── train/               # Training images organized by class
│   └── test/                # Test images organized by class
├── models/
│   └── saved_model/         # Trained model files
├── src/
│   ├── train.py             # Training script
│   └── test.py              # Testing script
├── notebooks/               # Jupyter notebooks for exploration
├── verify_api.py            # API verification script
├── requirements.txt         # Python dependencies
└── README.md
```

## 🧪 Verification

Run the automated verification script to test the API:

```bash
python verify_api.py
```

This script will:
1. Start the API server
2. Perform a health check
3. Send a test image for prediction
4. Verify the response format
5. Shut down the server

## 🛠️ Technologies Used

- **TensorFlow 2.12.0**: Deep learning framework
- **MobileNetV2**: Pre-trained model for transfer learning
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **NumPy**: Numerical computing
- **OpenCV**: Image processing
- **Pillow**: Image handling

## 📊 Model Architecture

The model uses transfer learning with the following architecture:
1. **Base Model**: MobileNetV2 (pre-trained on ImageNet)
2. **Preprocessing**: Built-in MobileNetV2 preprocessing
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Dropout (0.2)**: Prevents overfitting
5. **Dense Layer**: 52 classes with softmax activation

## 🎯 Performance

The model achieves high accuracy on the test dataset. You can evaluate it using:

```bash
python src/test.py
```

## 🔧 Advanced Usage

### Fine-tuning

To improve accuracy, you can fine-tune the last layers of the base model:

```bash
python src/train.py --fine_tune 20 --epochs 10
```

This will unfreeze the last 20 layers of MobileNetV2 for fine-tuning.

### Custom Dataset

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── Class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── Class2/
│       └── image1.jpg
└── test/
    ├── Class1/
    └── Class2/
```

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: Ensure you have sufficient training data for each class to achieve good model performance. The current model is trained on a curated dataset of crop disease images.
