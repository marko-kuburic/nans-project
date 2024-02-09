import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Amazon_popular_books_dataset.csv')

col_names = ['brand', 'final_price', 'rating', 'reviews_count', 'categories']
df = df[col_names]

# Fill missing values
df['final_price'] = df['final_price'].fillna(df['final_price'].median())
df['rating'] = df['rating'].str.split().str[0].astype(float)
df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Frequency encoding for categories
category_list = [category for sublist in df['categories'] for category in sublist]
category_freq = pd.Series(category_list).value_counts().to_dict()

df['categories'] = df['categories'].apply(lambda categories: sum(category_freq[cat] for cat in categories))

# Perform one-hot encoding on the 'brand' column
df['brand'] = pd.Categorical(df['brand'])
df['brand'] = df['brand'].cat.codes

# Split data into features and target
X = df.drop(columns=['rating', 'final_price'], axis=1)
y = df['rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (Coefficient of Determination): {r_squared:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the above code for model training and evaluation is already executed

# Visualization of Actual vs Predicted Ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.show()

# Visualization of Prediction Errors
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()
