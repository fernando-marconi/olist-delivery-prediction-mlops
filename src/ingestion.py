import pandas as pd
import os

def run_ingestion():
    # Caminhos das pastas
    raw_path = "data/raw"
    processed_path = "data/processed"
    
    # Cria a pasta processed se ela não existir
    os.makedirs(processed_path, exist_ok=True)

    print("Lendo os dados da Olist (Orders, Items e Products)...")
    
    # Carregando as tabelas principais
    orders = pd.read_csv(os.path.join(raw_path, 'olist_orders_dataset.csv'))
    items = pd.read_csv(os.path.join(raw_path, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(raw_path, 'olist_products_dataset.csv'))
    
    # 1. Unindo Pedidos com Itens (Base que já tínhamos)
    df = pd.merge(orders, items, on='order_id', how='inner')
    
    # 2. Unindo com os Produtos para trazer peso e dimensões
    # Usamos 'left' para garantir que não perderemos pedidos caso algum produto não esteja na lista
    df = pd.merge(df, products, on='product_id', how='left')
    
    # Salvando o arquivo consolidado
    output_file = os.path.join(processed_path, 'consolidated_orders.csv')
    df.to_csv(output_file, index=False)
    print(f"Sucesso! Dataset consolidado com {df.shape[1]} colunas e {df.shape[0]} linhas.")

if __name__ == "__main__":
    run_ingestion()