import pandas as pd
from pathlib import Path

def load_dataset(csv_path, text_column='text', label_column='generated'):
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Ensure specified columns exist
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns {text_column} and/or {label_column} not found in CSV")
    
    df = df[[text_column, label_column]]
    df.dropna(inplace=True)
   #df = df.head(1000)
    return df

# Optional test run when module is run directly
if __name__ == "__main__":
    path = "/Users/susmitapaudel/projects/ai-text-detector/data/raw/AI_Human.csv"
    df = load_dataset(path)
    print("First 5 rows:\n", df.head())
    print("Shape:", df.shape)
    print("Columns:", df.columns)
