import torch
import pandas as pd
import joblib
from features import extract_all_features  # your functions
from model import TextNN

# Load trained model
train_features = pd.read_csv("/Users/susmitapaudel/projects/ai-text-detector/data/raw/features_output.csv").drop(columns=['text','generated'])
input_dim = train_features.shape[1]
model = TextNN(input_dim)
model.load_state_dict(torch.load('text_feature_nn.pth', map_location='cpu'))
model.eval()

# Load scaler
scaler = joblib.load('scaler.pkl')

def predict_text(text):
    # Extract features
    features_dict = extract_all_features(text)
    features = pd.DataFrame([features_dict])
    # Ensure columns are in same order as training
    features = features[train_features.columns]
    # Scale features
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    # Predict
    with torch.no_grad():
        prob = model(features_tensor).item()
    label = "AI-generated" if prob > 0.5 else "Human-written"
    return label, prob

# Example usage
if __name__ == "__main__":
    #text = "Artificial intelligence is transforming the world."
    text = input('enter the text')
    label, prob = predict_text(text)
    print(f"Prediction: {label}, Probability: {prob:.4f}")
