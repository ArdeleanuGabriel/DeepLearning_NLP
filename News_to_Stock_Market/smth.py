import pandas as pd
from keybert import KeyBERT
import os

file_path = 'df_0_news_only.parquet'
output_file = 'df_0_news_with_keywords.parquet'
temp_dir = 'temp_chunks'
os.makedirs(temp_dir, exist_ok=True)
chunk_size = 50

kw_model = KeyBERT()


def process_chunk(chunk_file, output_chunk_file):
    if os.path.exists(output_chunk_file):
        print(f"Skipping already processed chunk: {output_chunk_file}")
        return
    df_chunk = pd.read_parquet(chunk_file)
    df_chunk['key_words'] = df_chunk['text'].apply(lambda x: ', '.join(extract_keywords(x)))
    df_chunk.to_parquet(output_chunk_file, index=False)


def extract_keywords(text, top_n=10):
    if pd.isnull(text) or not isinstance(text, str):
        return []
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    return [kw[0] for kw in keywords]


def split_file(file_path, temp_dir, chunk_size):
    if all(os.path.exists(os.path.join(temp_dir, f'chunk_{i}.parquet')) for i in range(len(os.listdir(temp_dir)) // 2)):
        print("Chunks already split.")
        return [os.path.join(temp_dir, f'chunk_{i}.parquet') for i in range(len(os.listdir(temp_dir)) // 2)]

    df = pd.read_parquet(file_path)
    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    chunk_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_file = os.path.join(temp_dir, f'chunk_{i}.parquet')
        if not os.path.exists(chunk_file):
            df.iloc[start_idx:end_idx].to_parquet(chunk_file, index=False)
        chunk_files.append(chunk_file)

    return chunk_files


def merge_chunks(chunk_files, output_file):
    dfs = [pd.read_parquet(chunk_file) for chunk_file in chunk_files if '_processed' in chunk_file]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_parquet(output_file, index=False)


if __name__ == '__main__':
    chunk_files = split_file(file_path, temp_dir, chunk_size)

    # Process each chunk file
    processed_chunk_files = []
    for chunk_file in chunk_files:
        output_chunk_file = chunk_file.replace('.parquet', '_processed.parquet')
        process_chunk(chunk_file, output_chunk_file)
        processed_chunk_files.append(output_chunk_file)

    # Merge processed chunks into the final output file
    merge_chunks(processed_chunk_files, output_file)

    print("Keywords added and file saved to:", output_file)
