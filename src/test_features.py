from features import extract_all_features, build_feature_matrix
import pandas as pd

# Example text
text_sample = "Hello world! This is a test sentence."

# Test extract_features on a single text
features = extract_all_features(text_sample)
print("Features for one text:", features)

# Test build_feature_matrix on a small dataframe
df = pd.DataFrame({
    "text": [
        "Hello world! This is a test sentence.",
        "NLTK is amazing for natural language processing.",
        "Python makes data science easier."
    ]
})

feature_matrix = build_feature_matrix(df)
print("Feature matrix shape:", feature_matrix.shape)
print("Feature matrix:\n", feature_matrix)
