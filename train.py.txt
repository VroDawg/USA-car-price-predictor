import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
df = pd.read_csv('vehicles_clean.csv')

cat_cols = ['manufacturer', 'model', 'fuel', 'title_status',
            'transmission', 'drive', 'type', 'paint_color', 'state']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['price'])
y = df['price']

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

joblib.dump(model, 'car_price_model.pkl')
print("Model saved!")