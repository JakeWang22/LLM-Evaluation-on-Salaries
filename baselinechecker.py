import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_df = pd.read_csv('Training Data - sampled_postings.csv')
test_input_df = pd.read_csv('Testing Data - Sheet1.csv')         
test_complete_df = pd.read_csv('Testing Data Complete - Sheet1.csv')  

train_df = train_df.dropna(subset=['max_salary'])

train_descriptions = train_df['description'].fillna('')  
train_salaries = train_df['max_salary']

test_descriptions = test_complete_df['description'].fillna('')
test_true_salaries = test_complete_df['max_salary']

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_descriptions)
X_test = vectorizer.transform(test_descriptions)

model = LinearRegression()
model.fit(X_train, train_salaries)

predictions = model.predict(X_test)

mse = mean_squared_error(test_true_salaries, predictions)
mae = mean_absolute_error(test_true_salaries, predictions)
r2 = r2_score(test_true_salaries, predictions)

print("Linear Regression Baseline Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

output_df = pd.DataFrame({
    'description': test_complete_df['description'],
    'true_max_salary': test_true_salaries,
    'predicted_max_salary': predictions
})

output_df.to_csv('linRegPred.csv', index=False)