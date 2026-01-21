import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import os
import mlflow
import mlflow.sklearn
import dagshub

def run_train():
    # 1. Configuração do DagsHub e MLflow
    repo_owner = "fernando-marconi"
    repo_name = "olist-delivery-prediction-mlops"
    
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    
    # Ativa o log automático de parâmetros e métricas do Scikit-Learn
    mlflow.sklearn.autolog()

    # Caminhos
    input_file = "data/processed/final_dataset.csv"
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    df = pd.read_csv(input_file)
    features = ['price', 'freight_value', 'product_weight_g', 'product_volume_cm3']
    X = df[features].fillna(0)
    y = df['is_late']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Iniciando o experimento no MLflow
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        print("Iniciando treinamento com MLflow...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        
        # Registrando uma métrica personalizada manualmente
        mlflow.log_metric("f1_score_manual", score)
        
        print(f"Modelo treinado! F1-Score: {score:.4f}")

        # Salvando o modelo
        joblib.dump(model, os.path.join(model_path, "model.pkl"))

if __name__ == "__main__":
    run_train()