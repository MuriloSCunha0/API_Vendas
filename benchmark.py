import requests
import time

# Definir a URL da API
url_predict = 'http://localhost:8000/predict'
url_predict_with_shap = 'http://localhost:8000/predict_with_shap'

# Dados de exemplo
sales_data = {
    'produto': 'Produto_1',
    'mes': 1
}

# Função para medir desempenho
def measure_performance(url, data, n_requests=100):
    total_time = 0
    total_memory = 0

    for _ in range(n_requests):
        start_time = time.time()
        response = requests.post(url, json=data)
        response_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            total_time += result['execution_time']
            total_memory += result['memory_usage']
        else:
            print(f"Request failed with status code {response.status_code}")

    avg_time = total_time / n_requests
    avg_memory = total_memory / n_requests

    return avg_time, avg_memory

# Medir desempenho sem SHAP
avg_time_no_shap, avg_memory_no_shap = measure_performance(url_predict, sales_data)

# Medir desempenho com SHAP
avg_time_with_shap, avg_memory_with_shap = measure_performance(url_predict_with_shap, sales_data)

#Save Results

with open('results.txt', 'w') as f:
    f.write(f"Tempo médio sem SHAP: {avg_time_no_shap:.4f} segundos\n")
    f.write(f"Tempo médio com SHAP: {avg_time_with_shap:.4f} segundos\n")
    f.write(f"Uso médio de memória sem SHAP: {avg_memory_no_shap:.2f} bytes\n")
    f.write(f"Uso médio de memória com SHAP: {avg_memory_with_shap:.2f} bytes\n")
    f.write(f"Ganho de tempo com SHAP: {avg_time_no_shap - avg_time_with_shap:.4f} segundos\n")
    f.write(f"Economia de memória com SHAP: {avg_memory_no_shap - avg_memory_with_shap:.2f} bytes\n")
    f.write(f"Porcentagem de economia de memória com SHAP: {100 * (avg_memory_no_shap - avg_memory_with_shap) / avg_memory_no_shap:.2f}%\n")
