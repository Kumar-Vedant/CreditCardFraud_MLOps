import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# load data
df = pd.read_csv("data/train.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    # handle class imbalance (scale_pos_weight = #non-fraud/#fraud)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # train model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])

    # log model parameters to MLFlow
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("eval_metric", "logloss")
    mlflow.log_param("random_state", 42)

    # log data parameters to MLFlow
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_val))
    mlflow.log_param("num_features", X_train.shape[1])
    mlflow.log_param("fraud_ratio", sum(y_train) / len(y_train))

    # log model evaluation scores to MLFlow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # log model to MLFlow
    mlflow.sklearn.log_model(model, "model")

    # save model
    with open("api/model.pkl", "wb") as f:
        pickle.dump(model, f)

print("Model trained and saved!")