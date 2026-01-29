from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Load dataset
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Create DataFrame
df = pd.DataFrame({
    "text": data.data,
    "label": data.target,
    "category": [data.target_names[i] for i in data.target]
})

# Save as CSV
df.to_csv("news_data.csv", index=False)

print("CSV dataset created successfully!")
print(df.head())
