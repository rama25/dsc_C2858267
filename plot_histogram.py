import pandas as pd
import matplotlib.pyplot as plt
import json

file_path = 'transactions/transactions.txt'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

# Plot histogram of transaction amounts
plt.figure(figsize=(10, 6))
plt.hist(df['transactionAmount'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Histogram of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.grid(True)


plt.savefig('transaction_amount_histogram.png')
plt.show()
