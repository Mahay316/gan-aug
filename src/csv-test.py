import pandas as pd

chunks = []
chunk_size = 10000

for chunk in pd.read_csv('datasets/top-1m.csv', chunksize=chunk_size):
    for row in chunk.iterrows():
        print(f"Name: {row['no']}, Age: {row['url']}")
