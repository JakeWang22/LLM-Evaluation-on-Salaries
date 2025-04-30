# LLM-Evaluation-on-Salaries
The dataset from https://www.kaggle.com/datasets/arshkon/linkedin-job-postings was used which contains job descriptions and max salaries for job postings on LinkedIn. To evaluate different LLMs, we trained them using the training data file, then had the LLMs predict given the descriptions in the testing data file. We also conducted various baseline models, including a simple linear regression as well as BERT. We evaluated based on the root mean squared error (RMSE), the normalized RMSE, the RMSE improvement, as well as what percent of data points fell within different percentiles away from the actual. To get the BERT sample, instructions can be found in the running bert.py file. Similarly, instructions can also be found in the baselinechecker.py file in order to get the linear regression model. Based on our results, we found that BERT had the lowest root RMSE, Gemini had the greatest proportion of predictions fall near the actual values, and Claude showed the overall worst performance in all categories.

Execution Instructions:

  BERT - run pip install transformers datasets torch scikit-learn.
  Also had to run pip install --upgrade numpy<2.
  Then make sure the complete data file is loaded into the same folder that's opened.
  This is used to make the train/test split.
  Run this file but UPDATE THE PATH FOR WHERE TO SAVE THE CSV FILE.
  We then used the generated file to conduct analysis on excel.

  Linear Regression (baselinechecker) - Before running file, run pip install pandas scikit-learn. Make sure the kaggle dataset CSV is in the same folder and called 
  "postings.csv".

To Prompt LLMs:
Type - You are given 80 job descriptions with their maximum yearly salaries in U.S. dollars, followed by 20 new job descriptions without salaries. Learn from the first set and predict the maximum salary for each of the 20 new descriptions. Output a CSV with two columns: "description" and "predicted_max_salary," matching each new description with its predicted salary. Respond only with the completed CSV, no extra text.
Then, add the training data file and the testing data file and press enter.
