import pandas as pd

df = pd.read_csv('postings.csv')

filtered_df = df[(df['pay_period'] == 'YEARLY') & (df['location'] == 'New York, NY')]

filtered_df = filtered_df[['description', 'max_salary']]

sampled_df = filtered_df.sample(n=100, random_state=42)

sampled_df.to_csv('sampled_postings.csv', index=False)
sampled_df.to_json('sampled_postings.json', orient='records', lines=True)