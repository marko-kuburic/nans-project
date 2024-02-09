import pandas as pd
import ast
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Amazon_popular_books_dataset.csv')

col_names = ['brand', 'final_price', 'rating', 'root_bs_rank', 'reviews_count', 'categories']
df = df[col_names]

# Fill missing values with the mean for 'final_price'
df['final_price'] = df['final_price'].fillna(df['final_price'].mean())

# Convert the 'rating' column from a format like "4.6 out of 5 stars" to float
df['rating'] = df['rating'].str.split().str[0].astype(float)

# Convert string representations of lists in 'categories' column to actual lists
df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Get all unique categories in a flat list
all_categories = set(category for sublist in df['categories'] for category in sublist)

# Convert the set of all categories to a list to preserve order
all_categories_list = list(all_categories)

# Create a temporary DataFrame for category columns with default value of 0
categories_df = pd.DataFrame(0, index=df.index, columns=all_categories_list)

# Populate the temporary DataFrame; set to 1 where the book belongs to the category
for index, row in df.iterrows():
    for category in row['categories']:
        categories_df.at[index, category] = 1

# Concatenate the temporary DataFrame with the original DataFrame
df = pd.concat([df, categories_df], axis=1)

# Perform one-hot encoding on the 'brand' column
df_brand_encoded = pd.get_dummies(df['brand'], prefix='brand')
df = pd.concat([df, df_brand_encoded], axis=1)

# Drop the original 'brand' and 'categories' columns as they are now encoded
df.drop(['brand', 'categories'], axis=1, inplace=True)

# Fill missing values in the target variable with the mean
df['root_bs_rank'] = df['root_bs_rank'].fillna(df['root_bs_rank'].mean())

# Assuming 'rating' is the target variable and the rest are features
X = df.drop(columns=['rating'])
y = df['rating']

# Drop non-numeric columns from feature set X
X_numeric = X.select_dtypes(include=['number'])
X_numeric = X_numeric.fillna(X_numeric.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Add a constant to the model (important for OLS regression)
X_train_with_const = sm.add_constant(X_train)
X_test_with_const = sm.add_constant(X_test)

# Initialize and fit the OLS model
model = sm.OLS(y_train, X_train_with_const).fit()

# Make predictions on the test set
y_pred = model.predict(X_test_with_const)

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
