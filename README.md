# USA-based Car Price Predictor2026 

A machine learning web app that predicts used car prices based on vehicle features, limited between vehicles from 1990 up to 2021.

## Live Demo
Frontend: https://vrodawg.github.io/USA-car-price-predictor

API: https://usa-car-price-predictor.onrender.com/docs


## Tech Stack
- Python, pandas, scikit-learn (Gradient Boosting)
- FastAPI, uvicorn
- HTML, CSS, JavaScript

## Results
- Trained on 350,000+ real Craigslist vehicle listings
- 90% R² accuracy, $2,684 mean absolute error

## How to Run Locally
pip install -r requirements.txt
uvicorn app:app --reload
