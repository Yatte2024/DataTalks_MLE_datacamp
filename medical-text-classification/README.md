# Medical Text Classification Project

## 1. Project Description

This project focuses on classifying short medical abstracts into four disease categories:

- Cancer  
- Cardiology  
- Diabetes  
- Hypertension  

The goal is to build a machine learning model that automatically predicts the category of a medical abstract.  
Machine learning is suitable for this task because medical texts contain domain-specific terminology and phrase patterns that can be learned effectively using TF-IDF features and traditional ML models.

---

## 2. Dataset

The dataset used in this project is the **Medical Abstracts Dataset for Text Classification** from Kaggle:

https://www.kaggle.com/datasets/tboyle10/medicalabstracts

Place the dataset in:

```
data/train.csv  
data/test.csv
```

Each file must contain:

- `text` — the medical abstract  
- `label` — the disease category  

---

## 3. Environment Setup

### Using virtualenv

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## 4. Running the Project

### Clone the project

```bash
git clone <your-repo-url>
cd medical-text-classification
```

### Running the notebook (EDA + modeling)

Open:

```
notebook.ipynb
```

This notebook includes:

- Class distribution  
- Text length analysis  
- Unigrams and bigrams  
- Word clouds  
- t-SNE visualization  
- Model comparison  

---

## 5. Model Training (train.py)

To train the final model:

```bash
python train.py
```

The script will:

- Load training and test data  
- Extract TF-IDF features  
- Train Logistic Regression, Random Forest, and XGBoost  
- Use GridSearchCV to find the best hyperparameters  
- Evaluate the final model  
- Save:

```
models/best_model.pkl
models/tfidf.pkl
models/label_encoder.pkl
```

---

## 6. Prediction API (predict.py)

Start the web API:

```bash
python predict.py
```

The API runs at:

```
http://127.0.0.1:5000/predict
```

### Making a POST request

#### Using curl

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This therapy improves blood pressure control."}'
```

Example response:

```json
{
    "prediction": "Hypertension"
}
```

#### Using Python

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Immunotherapy is effective in metastatic melanoma."}

response = requests.post(url, json=data)
print(response.json())
```

---

## 7. Docker Deployment

### Build the Docker image

```bash
docker build -t medical-api .
```

### Run the container

```bash
docker run -p 5000:5000 medical-api
```

Access the API at:

```
http://127.0.0.1:5000/predict
```

---

## 8. Project Structure

```
medical-text-classification/
│
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── notebook.ipynb
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf.pkl
│   ├── label_encoder.pkl
│
└── Dockerfile
```

---

## 9. Cloud Deployment (Optional)

For extra credit, deployment steps may include:

### Render

1. Connect GitHub repository  
2. Create a new Web Service  
3. Set start command:

```
python predict.py
```

### Docker Hub + Cloud VM

```bash
docker build -t yourname/medical-api .
docker push yourname/medical-api
ssh <server>
docker pull yourname/medical-api
docker run -p 5000:5000 yourname/medical-api
```

---

## 10. Summary

This project demonstrates a full machine learning pipeline:

- Data exploration  
- Text processing with TF-IDF  
- Multiple ML models with hyperparameter tuning  
- Evaluation and model selection  
- Reproducible training script  
- REST API for prediction  
- Docker-based deployment  

It meets all requirements for a complete and deployable ML classification project.

