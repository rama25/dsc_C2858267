import pandas as pd
import json

file_path = 'transactions/transactions.txt'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

print(df.info())

summary = df.describe(include='all').transpose()
summary['null_values'] = df.isnull().sum()
print(summary)

summary.to_csv('summary_statistics.csv', index=True)
