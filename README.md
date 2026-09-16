# AI Customer Intelligence Platform

An end-to-end AI/ML customer support intelligence platform demonstrating supervised machine learning, unsupervised learning, deep learning, Transformer-based NLP, Generative AI, GANs, text-to-image generation, model evaluation, hyperparameter optimization, bias/fairness analysis, and a FastAPI inference service.

---

## Business Problem

Customer support organizations receive large volumes of customer requests involving billing, payments, bookings, accounts, loyalty, and services.

This platform demonstrates how AI/ML can help automate and enrich:

- Customer ticket classification
- Customer priority assessment
- Sentiment analysis
- Customer segmentation
- Resolution-time prediction
- AI-generated customer responses

### Example

**Customer ticket**

> I was charged twice for my hotel reservation and nobody has responded for three days. Please fix this.

**AI analysis**

```text
Category                  : Billing
Priority                  : High
Sentiment                 : Negative
Customer Segment          : High Value
Predicted Resolution Time: ~20 hours
```

---

## Solution Architecture

```text
                    Customer Support Ticket
                              |
                              v
                       Data Ingestion
                         CSV / SQL
                              |
                              v
                        Data Cleaning
                           Pandas
                              |
                              v
                     Feature Engineering
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
       Supervised ML   Unsupervised ML     NLP / Deep Learning
       Classification       K-Means          Transformers
             |                |                |
             v                v                v
     Logistic Regression   Customer          DistilBERT
                          Segmentation
             |                                 |
             +----------------+----------------+
                              |
                              v
                    Customer Intelligence
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
       Resolution Model                 Generative AI
       Random Forest                    Llama 3 / Ollama
             |                                 |
             v                                 v
    Predicted Resolution Time          Customer Response
                              |
                              v
                           FastAPI
                              |
                              v
                       /classify API
```

---

## Technology Stack

### Data / Python
- Python
- Pandas
- NumPy
- SQLite
- SQLAlchemy

### Machine Learning
- Scikit-learn
- Logistic Regression
- Random Forest
- K-Means
- GridSearchCV

### Deep Learning
- PyTorch
- Neural Networks

### NLP / Transformers
- Hugging Face Transformers
- DistilBERT
- TF-IDF
- Contextual embeddings

### Generative AI
- Llama 3
- Ollama
- Prompt engineering
- Stable Diffusion
- Diffusers

### Other AI
- GAN
- Synthetic data generation

### API / Persistence / Testing
- FastAPI
- Pydantic
- Uvicorn
- Joblib
- Pytest

---

## Project Structure

```text
ai-customer-intelligence-platform/
├── api/
│   └── app.py
├── data/
│   └── tickets.csv
├── models/
│   ├── ticket_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   └── resolution_time_model.joblib
├── outputs/
│   ├── model_evaluation.csv
│   └── hotel_amenities.png
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── database.py
│   ├── ml_classifier.py
│   ├── clustering.py
│   ├── neural_network.py
│   ├── transformer.py
│   ├── transformer_classifier.py
│   ├── llama.py
│   ├── gan.py
│   ├── text_to_image.py
│   ├── model_evaluation.py
│   ├── hyperparameter_optimization.py
│   └── bias_fairness_evaluation.py
├── tests/
│   └── test_analytics.py
├── requirements.txt
└── README.md
```

---

# AI / ML Capabilities

## 1. Data Wrangling and Feature Engineering

Implemented in `src/data_pipeline.py`.

The pipeline performs:

- CSV ingestion
- Schema validation
- Duplicate removal
- Missing-value handling
- Text normalization
- Numeric type conversion
- Feature engineering

Example engineered features:

```python
clean_df["ticket_length"] = clean_df["ticket_text"].str.len()

clean_df["is_high_value_customer"] = (
    clean_df["customer_spend"] >= 3000
).astype(int)

clean_df["is_frequent_customer"] = (
    clean_df["previous_tickets"] >= 5
).astype(int)
```

Run:

```bash
python src/data_pipeline.py
```

### Customer Behavior Features

Customer behavior in this project is represented by:

```text
customer_spend
previous_tickets
is_high_value_customer
is_frequent_customer
```

These features are used for customer segmentation and behavioral-segment model evaluation.

---

## 2. SQL Analytics

Implemented in `src/database.py`.

The cleaned dataset is loaded into SQLite.

Example query:

```sql
SELECT
    category,
    COUNT(*) AS ticket_count,
    AVG(customer_spend) AS average_customer_spend,
    AVG(resolution_hours) AS average_resolution_hours
FROM customer_tickets
GROUP BY category
ORDER BY ticket_count DESC;
```

Run:

```bash
python src/database.py
```

---

## 3. Supervised Machine Learning

### Logistic Regression

Implemented in `src/ml_classifier.py`.

Pipeline:

```text
Ticket Text
    |
    v
TF-IDF
    |
    v
Logistic Regression
    |
    v
Ticket Category
```

TF-IDF configuration:

```python
TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)
```

Persisted artifacts:

```text
models/ticket_classifier.joblib
models/tfidf_vectorizer.joblib
```

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

Run:

```bash
python src/ml_classifier.py
```

---

## 4. Unsupervised Learning

### K-Means Customer Clustering

Implemented in `src/clustering.py`.

```python
KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
```

The clustering model is evaluated using the Silhouette Score.

Observed result:

```text
Silhouette Score ≈ 0.0425
```

The relatively low score indicates weak cluster separation and demonstrates why unsupervised results should be evaluated rather than assumed to be meaningful.

Run:

```bash
python src/clustering.py
```

---

## 5. Neural Network

Implemented in `src/neural_network.py`.

Architecture:

```text
TF-IDF Features
       |
       v
Linear Layer
       |
       v
128 neurons
       |
       v
ReLU
       |
       v
Dropout
       |
       v
Output Layer
       |
       v
7 Categories
```

Training:

```text
Loss: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 0.001
Epochs: 50
```

Run:

```bash
python src/neural_network.py
```

---

## 6. Hugging Face Transformer / DistilBERT

Implemented in `src/transformer.py`.

Model:

```text
distilbert-base-uncased
```

DistilBERT is used to generate contextual NLP representations.

Demonstrated embedding output:

```text
torch.Size([5, 768])
```

Run:

```bash
python src/transformer.py
```

---

## 7. Transformer Classification

Implemented in `src/transformer_classifier.py`.

A DistilBERT sequence-classification model is fine-tuned for the seven ticket categories.

Training configuration:

```text
Model: DistilBERT
Batch Size: 16
Epochs: 2
Learning Rate: 2e-5
Optimizer: AdamW
```

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score

Because the dataset is synthetic and highly structured, the classification results are unusually high.

---

## 8. Llama 3 Generative AI

Implemented in `src/llama.py`.

Llama 3 is run locally through Ollama.

Pipeline:

```text
Customer Ticket
       |
       v
ML Classification
       |
       +-- Category
       +-- Priority
       +-- Sentiment
       +-- Resolution Time
       |
       v
     Llama 3
       |
       v
Customer Response
```

Run:

```bash
python src/llama.py
```

Ollama must be running locally with the `llama3` model available.

---

## 9. GAN

Implemented in `src/gan.py`.

A PyTorch Generative Adversarial Network demonstrates synthetic customer-spend generation.

```text
Random Noise
     |
     v
Generator
     |
     v
Synthetic Customer Spend
     |
     v
Discriminator
     ^
     |
Real Customer Spend
```

The Generator attempts to produce realistic values while the Discriminator attempts to distinguish generated values from real values.

This is an educational demonstration rather than a production synthetic-data generation system.

Run:

```bash
python src/gan.py
```

---

## 10. Text-to-Image Generation

Implemented in `src/text_to_image.py`.

Technology:

```text
Stable Diffusion
Diffusers
```

Pipeline:

```text
Text Prompt
    |
    v
Stable Diffusion
    |
    v
Generated Image
```

Output:

```text
outputs/hotel_amenities.png
```

Run:

```bash
python src/text_to_image.py
```

---

# Model Evaluation

Implemented in `src/model_evaluation.py`.

| Model | Task | Evaluation |
|---|---|---|
| Logistic Regression | Classification | Accuracy / Precision / Recall / F1 |
| PyTorch Neural Network | Classification | Accuracy / Classification metrics |
| DistilBERT | Classification | Accuracy / Precision / Recall / F1 |
| K-Means | Clustering | Silhouette Score |
| Random Forest | Regression | MAE |
| Optimized Random Forest | Regression | MAE |

Observed classification accuracy on the synthetic dataset:

```text
Logistic Regression      1.00
PyTorch Neural Network   1.00
DistilBERT               1.00
```

K-Means:

```text
Silhouette Score ≈ 0.0425
```

The identical classification scores do not establish that the models have identical real-world performance. The synthetic dataset contains highly category-specific language that makes classification unusually easy.

---

# Resolution-Time Prediction

The platform predicts expected ticket resolution time using:

```text
category
priority
sentiment
customer_spend
previous_tickets
ticket_length
is_high_value_customer
is_frequent_customer
```

Model:

```text
RandomForestRegressor
```

Baseline:

```text
MAE ≈ 2.26 hours
```

---

# Hyperparameter Optimization

Implemented in `src/hyperparameter_optimization.py`.

The Random Forest model is optimized using `GridSearchCV`.

Parameters explored:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

Best configuration found:

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
```

Results:

```text
Baseline MAE:    2.26 hours
Optimized MAE:   2.19 hours
Improvement:     0.07 hours
```

This demonstrates systematic hyperparameter tuning rather than relying only on default model settings.

---

# Bias and Fairness Evaluation

Implemented in `src/bias_fairness_evaluation.py`.

The dataset does not contain protected attributes such as:

```text
race
gender
age
```

Therefore, this project does not claim to perform protected-attribute fairness analysis.

Instead, model prediction error is compared across customer behavioral segments.

## High Value vs Standard

```text
High Value MAE:  2.225 hours
Standard MAE:    2.289 hours
```

## Frequent vs Non-Frequent

```text
Frequent Customer MAE:     2.215 hours
Non-Frequent Customer MAE: 2.300 hours
```

The differences are relatively small in this synthetic dataset.

Run:

```bash
python src/bias_fairness_evaluation.py
```

### Interview Positioning

> I evaluated model error across customer behavioral segments. The MAE was similar across high-value versus standard customers and frequent versus non-frequent customers. Since the dataset doesn't contain protected attributes such as race, gender, or age, this is a behavioral-segment fairness analysis rather than a protected-attribute fairness assessment.

---

# FastAPI Inference Service

Implemented in `api/app.py`.

The API combines:

```text
ML Classification
       +
Customer Segmentation
       +
Resolution-Time Prediction
       +
Llama 3 Response Generation
```

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

## Ticket Classification

```http
POST /classify
```

Example request:

```json
{
  "ticket_text": "I was charged twice for my hotel reservation and nobody has responded for three days. Please fix this.",
  "customer_spend": 5000,
  "previous_tickets": 2
}
```

Example response structure:

```json
{
  "ticket_text": "...",
  "predicted_category": "Billing",
  "confidence": 0.75,
  "priority": "High",
  "sentiment": "Negative",
  "customer_segment": "High Value",
  "predicted_resolution_hours": 20.11,
  "generated_response": "..."
}
```

The classifier and resolution model are persisted using Joblib and loaded when FastAPI starts.

---

# Classification Metrics

For classification, the project considers:

### Accuracy

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Of the items predicted as a class, how many actually belong to that class?

### Recall

Of the actual items belonging to a class, how many were correctly identified?

### F1 Score

Harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

Accuracy alone can be misleading for imbalanced datasets, so production systems should consider precision, recall, and F1 at the class level.

---

# Regression Metric

## Mean Absolute Error

```text
MAE =
Average(|Actual - Predicted|)
```

For example:

```text
MAE = 2.19 hours
```

means predictions differ from actual resolution time by approximately 2.19 hours on average.

---

# Unsupervised Learning Metric

## Silhouette Score

The Silhouette Score measures how well observations fit within their assigned cluster compared with neighboring clusters.

Higher values generally indicate better-separated clusters.

This project obtained:

```text
≈ 0.0425
```

which indicates weak cluster separation.

---

# Responsible AI Considerations

The project demonstrates:

- Data quality validation
- Missing-value handling
- Model evaluation
- Bias/fairness analysis
- Awareness of synthetic-data limitations
- Comparison of multiple model architectures
- Avoiding unsupported claims based on synthetic benchmark results

The LLM response generation uses prompt constraints to reduce unsupported claims and prevent the model from claiming that backend actions such as refunds or escalations have already occurred.

---

# Running the Project

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Data Pipeline

```bash
python src/data_pipeline.py
```

## Train Classification Model

```bash
python src/ml_classifier.py
```

## Run Model Evaluation

```bash
python src/model_evaluation.py
```

## Run Hyperparameter Optimization

```bash
python src/hyperparameter_optimization.py
```

## Run Fairness Evaluation

```bash
python src/bias_fairness_evaluation.py
```

## Run Llama

Make sure Ollama is running with Llama 3:

```bash
ollama list
```

Then:

```bash
python src/llama.py
```

## Run FastAPI

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to test the `/classify` endpoint.

---

# Model Selection

### Logistic Regression
Fast baseline and interpretable text classification.

### Neural Network
Demonstrates nonlinear deep-learning classification.

### DistilBERT
Provides contextual NLP representations and Transformer-based classification.

### K-Means
Supports unsupervised customer segmentation.

### Random Forest
Supports nonlinear tabular prediction for resolution-time estimation.

### Llama 3
Supports natural-language generation and customer-facing responses.

---

# End-to-End Workflow

```text
Business Problem
      ↓
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
SQL Analytics
      ↓
Model Development
      ↓
Model Comparison
      ↓
Hyperparameter Optimization
      ↓
Fairness Evaluation
      ↓
Generative AI
      ↓
FastAPI Inference Interface
```

---

# Interview Summary

> I built an AI Customer Intelligence Platform for intelligent customer-support ticket processing. I used Pandas and SQL for data preparation, Logistic Regression as a classical ML baseline, K-Means for unsupervised segmentation, PyTorch for a neural-network classifier, and Hugging Face DistilBERT for Transformer-based NLP. I also integrated Llama 3 through Ollama for generative responses, demonstrated GAN-based synthetic data generation and Stable Diffusion text-to-image generation, and used Random Forest with GridSearchCV for resolution-time prediction. Finally, I evaluated the models using classification metrics, MAE and Silhouette Score, performed a behavioral-segment fairness analysis, and exposed the solution through FastAPI.

---

# Important Limitations

This project is primarily an AI/ML demonstration and interview portfolio project.

The dataset is synthetic and highly structured. Therefore:

- Classification accuracy is not representative of production performance.
- The K-Means clusters have weak separation.
- GAN output is an educational demonstration.
- Fairness analysis does not use protected attributes.
- Llama responses require additional production-grade validation and governance.
- The project is not intended to make real customer-support decisions without additional controls.

---

# Skills Demonstrated

```text
Python
Pandas
NumPy
SQL
Scikit-learn
PyTorch
Hugging Face Transformers
DistilBERT
Llama 3
Ollama
GANs
Stable Diffusion
Diffusers
NLP
Supervised Learning
Unsupervised Learning
Deep Learning
Generative AI
Feature Engineering
Model Evaluation
Hyperparameter Optimization
Responsible AI
Fairness Evaluation
FastAPI
Joblib
Pytest
```

## Project Goal

The goal of this project is to demonstrate how multiple AI/ML techniques can be combined into a coherent enterprise business solution rather than implementing isolated machine-learning examples.