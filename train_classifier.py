import json
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

def get_label(events):
    flat = " ".join(events)
    if "run-out" in flat:
        return "run-out"
    elif "wicket" in flat:
        return "wicket"
    elif "six" in flat:
        return "six"
    elif "four" in flat:
        return "four"
    elif "milestone" in flat:
        return "milestone"
    elif "win" in flat:
        return "win"
    else:
        return "none"

def train_model(features_path="data/delivery_features.json", model_out="highlight_pipeline.pkl"):
    print(f"Loading raw features from {features_path}...")
    with open(features_path) as f:
        data = json.load(f)

    X_raw = []
    y_raw = []

    for item in data:
        feat = item.get("feature_vector")
        if not feat or len(feat) < 1165:
            continue
        X_raw.append(feat)
        label = get_label(item.get("events", []))
        y_raw.append(label)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    
    if len(X_raw) == 0:
        print("No valid features found. Ensure stage2_features.py ran successfully.")
        return

    print(f"Loaded {len(X_raw)} items. Splitting features...")

    vid_feats = X_raw[:, 0:768]
    txt_feats = X_raw[:, 768:1152]
    core_feats = X_raw[:, 1152:1165]

    n_samples = vid_feats.shape[0]
    pca_vid = PCA(n_components=min(32, n_samples))
    pca_txt = PCA(n_components=min(16, n_samples))

    print(f"Fitting PCA(Video) to {pca_vid.n_components} dims...")
    vid_pca = pca_vid.fit_transform(vid_feats)
    
    print(f"Fitting PCA(Text)  to {pca_txt.n_components} dims...")
    txt_pca = pca_txt.fit_transform(txt_feats)

    X_final = np.concatenate([vid_pca, txt_pca, core_feats], axis=1)
    print(f"Final feature vector size per delivery: {X_final.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    print("\nClass Distribution:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls:<12}: {np.sum(y_encoded == i)}")

    print("\nTraining XGBoost mapping PCA + Metadata to Highlights...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42
    )

    clf.fit(X_final, y_encoded)
    
    y_pred = clf.predict(X_final)
    print("\nClassification Report (On Train Set):")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_, zero_division=0))

    pipeline = {
        "pca_vid": pca_vid,
        "pca_txt": pca_txt,
        "label_encoder": le,
        "model": clf,
        "dim_vid": 768,
        "dim_txt": 384
    }
    
    joblib.dump(pipeline, model_out)
    print(f"\nPipeline (PCA + XGBoost) saved to '{model_out}'")

if __name__ == "__main__":
    import os
    data_dir = os.environ.get("DATA_DIR", "data")
    train_model(
        features_path=f"{data_dir}/delivery_features.json",
        model_out=f"{data_dir}/highlight_pipeline.pkl"
    )
