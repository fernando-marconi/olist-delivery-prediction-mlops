import pandas as pd
import os

def run_preprocessing():
    # Definindo caminhos
    input_file = "data/processed/consolidated_orders.csv"
    output_file = "data/processed/final_dataset.csv"
    
    # Lendo os dados que você consolidou anteriormente
    df = pd.read_csv(input_file)

    # 1. Convertendo colunas de data (que hoje são texto) para o formato de tempo
    date_columns = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # 2. Definindo o nosso ALVO (Target): O pedido atrasou?
    # Primeiro, removemos pedidos não entregues (onde a data de entrega é vazia)
    df = df.dropna(subset=['order_delivered_customer_date'])
    
    # Criamos a coluna 'is_late': 1 se a entrega real foi depois da estimada, 0 se não.
    df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

    # 3. Criando variáveis de tempo (Features)
    # Tempo total de entrega em dias
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # Salvando o dataset pronto para o treinamento
    df.to_csv(output_file, index=False)
    print(f"Sucesso! Dataset final gerado com {df.shape[0]} linhas.")
    print(f"Coluna de alvo 'is_late' criada com sucesso.")

if __name__ == "__main__":
    run_preprocessing()