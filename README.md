# E-commerce Recommendation System using Collaborative Filtering

A customer specific product recommendation system built using matrix factorization and collaborative filtering to predict customer preferences based on past purchase behaviour.

## Key Objectives

- Develop a recommendation engine for an e-commerce dataset using collaborative filtering.
- Provide customer specific product recommendations based on customer’s past purchases.
- Build a deployable API using Flask to enable real-time recommendations.

## Methodology

Matrix factorization, a popular collaborative filtering technique, to capture latent relationships between users and products is used here. This involves:

1. Encoding customer and product IDs to integer indices.
2. Training a neural network model to learn low-dimensional embeddings.
3. Predicting interaction scores based on dot product of user and product embeddings.

## Model Pipeline

### Data Preprocessing

- Load the e-commerce dataset.
- Drop missing values and convert quantity to integer.
- Encode Customer Id and Product using LabelEncoder.

### Model Definition

- A PyTorch-based matrix factorization model with embedding layers for users and products.

### Training

- Train on quantity of items purchased using Mean Squared Error loss.
- Save the model as model.pth.

### Serving via Flask API

- `/recommend` endpoint accepts a user ID and returns top-N product recommendations.

## Challenges Addressed

- More importance to frequently occurring customer-product interactions to avoid cold start issues.
- Dimensionality reduction is used to handle limited user-item interaction matrix.
- Scalable Pipeline designed using PyTorch and Flask for easy integration. 

## Results

- The model showed strong convergence during training: Loss reduced from 7122.28 to 89.53 over 10 epochs.
- Accurate and relevant product recommendations are generated for known/ repetitive customers.
- Fast inference time enables real-time responses from the Flask API.

## Impact

This model can be integrated into any e-commerce platform to:

- Enhance cutomer experience through personalization.
- Boost engagement and conversions.
- Reduce less engagement of customers by showing relevant products.

## Technology and Tools

| Category       | Tools Used                         |
|----------------|-------------------------------------|
| Language       | Python 3.10                         |
| Libraries      | PyTorch, Pandas, Scikit-learn       |
| Framework      | Flask (REST API)                    |
| Model Type     | Matrix Factorization (Embedding layers) |
| Dataset Source | E-commerce CSV dataset (Kaggle)     |

## How to Run

### Install Dependencies

```bash
pip install pandas torch flask scikit-learn
```

### Prepare Dataset

Save the dataset at required location

### Run the Application

```bash
python ecommerce.py
```

### Use the API

Send the post request to `http://127.0.0.1:5000/recommend` with the JSON payload:

```json
{
  "user_id": "10000",
  "num_recommendations": 5
}
```
