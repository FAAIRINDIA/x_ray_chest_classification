# Chest X-Ray Multi-Label Classification with DenseNet121

## 📈 Project Overview

This project focuses on developing an AI model capable of analyzing chest X-ray images and predicting the presence or absence of **14 different medical conditions**. The model uses **DenseNet121** as the backbone, trained on a custom dataset where each image can have multiple labels. Additionally, **Grad-CAM** is used to visualize the regions of the X-ray image that the model focused on while making predictions. A **Flask-based microsite** enables users to upload chest X-ray images and receive predictions along with highlighted areas.

---

## 📁 Project Structure

```
chest-xray-classifier/
|
├── train-small.csv
├── val_data.csv
├── test_data.csv
├── images/                     # Folder containing all X-ray images
│   ├── img1.png
│   └── ...
|
├── model/
│   └── chest_xray_densenet_model.h5
|
├── app.py                      # Flask microsite
├── templates/
│   ├── index.html
│   └── result.html
|
├── gradcam.py                  # GRAD-CAM utility
├── main_train.py               # Main training and evaluation script
├── README.md
└── requirements.txt
```

---

## 🚀 Features

- ✅ Multi-label classification for 14 different conditions
- ✅ Deep learning model using **DenseNet121**
- ✅ Grad-CAM based visualization for explainability
- ✅ Flask web app to interact with the model via browser
- ✅ Train/Validation/Test split using separate CSVs
- ✅ Saves trained model for future deployment

---

## 🧰 Medical Conditions Covered

> Each X-ray may be annotated with one or more of the following 14 medical conditions (replace with actual column names):

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

---

## 📊 Dataset Description

The dataset contains:

- One `Image` column with the image filename
- One `Patientid` column (not used in training)
- 14 condition columns with binary values:
  - `1` → condition present
  - `0` → condition absent

Images are matched via filenames and are not sorted. The model is trained using paths built dynamically.

---

## 🛠️ Technologies Used

| Technology         | Purpose                         |
| ------------------ | ------------------------------- |
| Python             | Programming language            |
| TensorFlow/Keras   | Deep learning framework         |
| DenseNet121        | Base convolutional architecture |
| Pandas & NumPy     | Data manipulation               |
| Matplotlib/Seaborn | Visualization tools             |
| Grad-CAM           | Model explainability            |
| Flask              | Microsite / Web interface       |

---

## 🧪 How It Works

### 🔁 Training Phase:

- Preprocesses CSVs (excluding `Image` and `Patientid`)
- Performs data augmentation on training images
- Trains a DenseNet121-based model using `binary_crossentropy` loss
- Saves the trained model (`.h5`) file for reuse

### 🔍 Validation & Testing:

- Evaluates model accuracy and loss
- Displays predictions and confusion matrix
- Threshold of `0.5` used to classify presence of each condition

### 🔍 Grad-CAM Visualization:

- Generates heatmaps highlighting the most important image regions
- Makes model decisions interpretable for users and professionals

### 🌐 Microsite:

- Built with Flask
- Upload an image → Get predictions and Grad-CAM heatmap

---

## 👩‍💼 How to Run

### ⚙️ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/chest-xray-classifier.git
cd chest-xray-classifier
```

### 🐍 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 📈 Step 3: Train the Model

```bash
python main_train.py
```

### 🌍 Step 4: Run the Microsite

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 📸 Sample Results

| Image | Grad-CAM | Predictions                  |
| ----- | -------- | ---------------------------- |
|       |          | ['Infiltration', 'Effusion'] |
|       |          | ['Cardiomegaly']             |

---

## 📂 Files to Upload on GitHub

| File                                               | Purpose                          |
| -------------------------------------------------- | -------------------------------- |
| `train-small.csv`, `val_data.csv`, `test_data.csv` | CSV files with labels            |
| `images/`                                          | Folder with X-ray images         |
| `main_train.py`                                    | Main model training + evaluation |
| `gradcam.py`                                       | GRAD-CAM helper                  |
| `app.py`                                           | Flask microsite                  |
| `templates/`                                       | HTML files for the site          |
| `requirements.txt`                                 | List of Python dependencies      |
| `README.md`                                        | This readme you're reading now   |

---

## ✨ Future Work

- Support for multi-class segmentation of lungs
- Improved performance through fine-tuning DenseNet121
- Deploy the model via a cloud API (e.g., Streamlit or FastAPI)
- Integration with medical report generation from image

---

## 👩‍⚖️ Disclaimer

This model is a **prototype** and **not intended for clinical use**. It's designed for educational purposes only.

---

## 🧑‍🏫 About the Author

**Yash Kiran (YK)**\
Pursuing a diploma in **Artificial Intelligence and Machine Learning**\
Interested in ML/DL, Computer Vision, and Explainable AI.

