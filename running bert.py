from sklearn.linear_model import Ridge
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os

# instructions:
# run pip install transformers datasets torch scikit-learn
# also had to run pip install --upgrade numpy<2
# then make sure the complete data file is loaded into the same folder that's opened
    # this is used to make the train/test split
# run this file but UPDATE THE PATH FOR WHERE TO SAVE THE CSV FILE
# we then used the generated file to conduct analysis on excel

# in same folder but make sure it isn't open somewhere else bc then will get permission denied error
df = pd.read_excel("Complete Data.xlsx")

descriptions = df["description"].tolist()
salaries = df["max_salary"].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    descriptions, salaries, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()  # freeze weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)

# get the embeddings for the text data using BERT's frozen weights
def embed_texts(texts):
    all_embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.pooler_output.squeeze().cpu().numpy()
            all_embeddings.append(cls_embedding)
    return np.array(all_embeddings)

X_train = embed_texts(train_texts)
X_test = embed_texts(test_texts)

# once we have the embeddings, we can use them as features for regression
# the frozen weights using BERT have some information "encoded" in them so we can try to use this to be better than just the regression model
reg = Ridge()
reg.fit(X_train, train_labels)

# Predict and evaluate
preds = reg.predict(X_test)

# saving the values to a csv file, this is where we got the values to evaluate how well the model did at predictions
# calculated the RMSE, normalized RMSE, and improvement from the median guess
df = pd.DataFrame({
    'True_Salary': test_labels,
    'Predicted_Salary': preds
})

df.to_csv(r'C:\Users\umadu\OneDrive - Georgia Institute of Technology\Documents\local georgia tech files\24-25\spring\nlp final project\bert_salary_predictions.csv', index=False)

print("Saved to bert_salary_predictions.csv")