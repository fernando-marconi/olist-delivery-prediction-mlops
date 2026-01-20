import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
import os

def run_train():
    # Caminhos de entrada e saída
    input_file = "data/processed/final_dataset.csv"
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    # 1. Carregando os dados pré-processados
    df = pd.read_csv(input_file)

    # 2. Selecionando variáveis para o modelo (Features e Target)
    # Por enquanto, usaremos features numéricas básicas presentes no merge
    features = ['price', 'freight_value'] 
    X = df[features].fillna(0)
    y = df['is_late']

    # 3. Divisão em Treino e Teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Treinamento do Modelo
    print("Iniciando o treinamento do modelo...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Avaliação básica
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"Modelo treinado! F1-Score nos dados de teste: {score:.4f}")
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred))

    # 6. Salvando o modelo treinado (O "artefato")
    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    print(f"Modelo salvo em: {model_path}/model.pkl")

if __name__ == "__main__":
    run_train()