import pandas as pd
import numpy as np
import json

file_path = 'transactions/transactions.txt'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])

df['isReversal'] = np.where(df['transactionAmount'].shift(-1) == -df['transactionAmount'], True, False)
reversed_transactions = df[df['isReversal']]

df['time_diff'] = df['transactionDateTime'].diff().dt.total_seconds().abs()
multi_swipe = df[(df['transactionAmount'] == df['transactionAmount'].shift(-1)) & 
                 (df['merchantCategoryCode'] == df['merchantCategoryCode'].shift(-1)) & 
                 (df['time_diff'] <= 300)]

total_reversed = reversed_transactions['transactionAmount'].sum()
total_multi_swipe = multi_swipe['transactionAmount'].sum()

print(f"Total Reversed Transactions: {len(reversed_transactions)}, Total Amount: {total_reversed}")
print(f"Total Multi-Swipe Transactions: {len(multi_swipe)}, Total Amount: {total_multi_swipe}")

reversed_transactions.to_csv('reversed_transactions.csv', index=False)
multi_swipe.to_csv('multi_swipe_transactions.csv', index=False)
