import pandas as pd
import numpy as np

# Configurações para gerar uma base de dados grande
num_products = 1000  # número de produtos
num_sales = 200  # número de vendas

# Gerar dados de produtos
products = [f'Produto_{i+1}' for i in range(num_products)]
product_types = ['Tipo_1', 'Tipo_2', 'Tipo_3']
prices = np.random.uniform(10, 100, num_products)

# Gerar vendas aleatórias
np.random.seed(42)
sales_data = {
    'produto': np.random.choice(products, num_sales),
    'codigo': np.random.randint(10000, 99999, num_sales),
    'tipo': np.random.choice(product_types, num_sales),
    'preco': np.random.choice(prices, num_sales),
    'quantidade': np.random.randint(1, 100, num_sales),
    'data': np.random.choice(pd.date_range(start='2020-01-01', end='2023-12-31'), num_sales)
}

# Criar DataFrame e salvar como CSV
df_large = pd.DataFrame(sales_data)
df_large.to_csv('vendas_grande.csv', index=False)
