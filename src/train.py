import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import os
import mlflow
import mlflow.sklearn
import dagshub

def run_train():
    # Setup DagsHub/MLflow - Credenciais do seu projeto
    repo_owner = "fernando-marconi"
    repo_name = "olist-delivery-prediction-mlops"
    
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    
    # O autolog registrará todas as tentativas da busca automática
    mlflow.sklearn.autolog()

    # Caminhos e Dados
    input_file = "data/processed/final_dataset.csv"
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    df = pd.read_csv(input_file)
    features = [
        'price', 'freight_value', 'product_weight_g', 
        'product_volume_cm3', 'is_interstate', 
        'estimated_days_to_deliver', 'purchase_day_of_week'
    ]
    X = df[features].fillna(0)
    y = df['is_late']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Definindo o "Grade" de parâmetros para testar
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }

    base_model = RandomForestClassifier(class_weight='balanced', random_state=42)

    # 2. Configurando a busca automática (focada em melhorar o F1-Score)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1 # Usa todos os núcleos do seu processador
    )

    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        print("Iniciando a busca automática pelos melhores hiperparâmetros...")
        grid_search.fit(X_train, y_train)

        # 3. Extraindo o melhor modelo encontrado
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        score = f1_score(y_test, y_pred)
        
        print(f"Melhor configuração encontrada: {grid_search.best_params_}")
        print(f"F1-Score de Teste com esta configuração: {score:.4f}")

        # Salvando o artefato final
        joblib.dump(best_model, os.path.join(model_path, "model.pkl"))

if __name__ == "__main__":
    run_train()