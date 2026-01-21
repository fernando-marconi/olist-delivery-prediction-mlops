import pandas as pd
import os

def run_preprocessing():
    input_file = "data/processed/consolidated_orders.csv"
    output_file = "data/processed/final_dataset.csv"
    
    df = pd.read_csv(input_file)

    # 1. Converter datas
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # 2. Criar Alvo e limpar nulos
    df = df.dropna(subset=['order_delivered_customer_date'])
    df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

    # 3. Engenharia de Features Temporais
    # Prazo prometido em dias
    df['estimated_days_to_deliver'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    
    # Dia da semana (0=Segunda, 6=Domingo)
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek

    # 4. Features Geográficas e de Produto
    df['is_interstate'] = (df['customer_state'] != df['seller_state']).astype(int)
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].mean())
    df['product_volume_cm3'] = df['product_volume_cm3'].fillna(df['product_volume_cm3'].mean())

    df.to_csv(output_file, index=False)
    print(f"Sucesso! Dataset gerado com features de Prazo e Dia da Semana.")

if __name__ == "__main__":
    run_preprocessing()