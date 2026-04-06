import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
df = pd.read_csv('vehicles_clean.csv')

cat_cols = ['manufacturer', 'model', 'fuel', 'title_status',
            'transmission', 'drive', 'type', 'paint_color', 'state']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=['price'])
y = df['price']

print("Training model...")
model = GradientBoostingRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, random_state=42)
model.fit(X, y)

joblib.dump(model, 'car_price_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
print("Done! Model and encoders saved.")