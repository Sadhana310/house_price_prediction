import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Check if dataset exists, if not create a sample dataset
dataset_file = 'Bangalore_House_Data.csv'

if not os.path.exists(dataset_file):
    print(f"Dataset file '{dataset_file}' not found. Creating a sample dataset...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    locations = ['Electronic City', 'Whitefield', 'Marathahalli', 'HSR Layout', 'Koramangala', 
                 'Indiranagar', 'JP Nagar', 'Bannerghatta Road', 'BTM Layout', 'Hebbal']
    
    data = {
        'location': np.random.choice(locations, n_samples),
        'total_sqft': np.random.uniform(500, 5000, n_samples),
        'bath': np.random.randint(1, 6, n_samples),
        'bhk': np.random.randint(1, 6, n_samples),
        'price': np.random.uniform(20, 200, n_samples)  # Price in lakhs
    }
    
    df = pd.DataFrame(data)
    df.to_csv(dataset_file, index=False)
    print(f"Sample dataset created: {dataset_file}")

# Load dataset
try:
    df = pd.read_csv(dataset_file)
    print(f"Dataset loaded successfully with {len(df)} records")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Data cleaning
df = df.dropna()
print(f"After dropping NaN values: {len(df)} records")

# Convert total_sqft to numeric, handling any non-numeric values
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df = df.dropna(subset=['total_sqft'])
print(f"After cleaning total_sqft: {len(df)} records")

# Feature engineering
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
df = df[df['bath'] < df['bhk'] + 2]  # Removing outliers
print(f"After removing outliers: {len(df)} records")

# Location encoding
location_stats = df['location'].value_counts()
df['location'] = df['location'].apply(lambda x: 'other' if location_stats[x] <= 10 else x)

# Final data
dummies = pd.get_dummies(df['location'])
# Only drop 'other' column if it exists
if 'other' in dummies.columns:
    X = pd.concat([df[['total_sqft', 'bath', 'bhk']], dummies.drop('other', axis=1)], axis=1)
else:
    X = pd.concat([df[['total_sqft', 'bath', 'bhk']], dummies], axis=1)
y = df['price'] * 100000

print(f"Final dataset shape: X={X.shape}, y={y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Model R² Score - Train: {train_score:.4f}, Test: {test_score:.4f}")

# Save model and columns
try:
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully: house_price_model.pkl")

    columns = {
        'data_columns': list(X.columns)
    }

    with open("columns.json", "w") as f:
        import json
        f.write(json.dumps(columns))
    print("Columns saved successfully: columns.json")
    
except Exception as e:
    print(f"Error saving model files: {e}")