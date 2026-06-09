import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def get_label(events):
    if "wicket" in events:
        return "wicket"
    elif "six" in events:
        return "six"
    elif "four" in events:
        return "four"
    elif "chase_climax" in events:
        return "tense_moment"
    else:
        # We optionally treat 'runs' (1s/2s/3s) and empty events as 'none' 
        # unless you specifically want to model normal runs.
        return "none"

def train_model(features_path="data/delivery_features.json", model_out="highlight_classifier.pkl"):
    print("Loading delivery features...")
    with open(features_path) as f:
        data = json.load(f)
        
    X = []
    y = []
    
    for item in data:
        features = item.get("feature_vector")
        if not features:
            continue
            
        label = get_label(item.get("events", []))
        
        X.append(features)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"  {cls:<15}: {cnt}")
        
    # Split into train/test (80/20)
    # Stratify ensures each class is represented proportionally in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(unique_classes) > 1 else None
    )
    
    print(f"\nTraining Random Forest model on {len(X_train)} samples...")
    # Using balanced class weights to help with class imbalance (lots of 'none', few 'six')
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save the model
    joblib.dump(clf, model_out)
    print(f"\nModel saved successfully as '{model_out}'")
    
    # Print feature importances
    importances = clf.feature_importances_
    
    # Quick explanation of top features: (CLIP is 0-511, CLIP Prompts 512-519, Optical Flow 520-522, Scoreboard 523-530, Whisper 531)
    prompt_importance = sum(importances[512:520])
    flow_importance = sum(importances[520:523])
    score_importance = sum(importances[523:531])
    whisper_importance = importances[531] if len(importances) > 531 else 0
    vision_importance = sum(importances[0:512])
    
    print("\nFeature Modality Importances:")
    print(f"  Raw Vision (CLIP):    {vision_importance:.3f}")
    print(f"  CLIP Prompts:         {prompt_importance:.3f}")
    print(f"  Optical Flow:         {flow_importance:.3f}")
    print(f"  Scoreboard State:     {score_importance:.3f}")
    print(f"  Commentary (Whisper): {whisper_importance:.3f}")

if __name__ == "__main__":
    train_model()
