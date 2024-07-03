from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import shap
import time
import tracemalloc

app = FastAPI()

# Carregar modelo, explainer e colunas de treino
model = joblib.load('modelo_vendas.pkl')
explainer = joblib.load('explainer.pkl')
X_train_columns = joblib.load('x_train_columns.pkl')

# Cache para valores SHAP
shap_cache = {}

class SalesData(BaseModel):
    produto: str
    mes: int

def get_shap_values(df):
    df_str = df.to_string()
    if df_str in shap_cache:
        return shap_cache[df_str]
    else:
        shap_values = explainer.shap_values(df, check_additivity=False)
        shap_cache[df_str] = shap_values
        return shap_values

@app.post('/predict')
async def predict(sales_data: SalesData):
    try:
        start_time = time.time()
        tracemalloc.start()
        
        df = pd.DataFrame([sales_data.dict()])
        df = pd.get_dummies(df, columns=['produto'])
        
        # Adicionar colunas faltantes do one-hot encoding
        df = df.reindex(columns=X_train_columns, fill_value=0)  # Reindexar e preencher valores ausentes

        prediction = model.predict(df)
        current, peak = tracemalloc.get_traced_memory()
        execution_time = time.time() - start_time
        tracemalloc.stop()

        return {
            'prediction': prediction[0],
            'execution_time': execution_time,
            'memory_usage': peak,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/predict_with_shap')
async def predict_with_shap(sales_data: SalesData):
    try:
        start_time = time.time()
        tracemalloc.start()

        df = pd.DataFrame([sales_data.dict()])
        df = pd.get_dummies(df, columns=['produto'])
        
        # Adicionar colunas faltantes do one-hot encoding
        df = df.reindex(columns=X_train_columns, fill_value=0)  # Reindexar e preencher valores ausentes

        prediction = model.predict(df)
        shap_values = get_shap_values(df)
        shap_values_list = shap_values.tolist()

        current, peak = tracemalloc.get_traced_memory()
        execution_time = time.time() - start_time
        tracemalloc.stop()

        return {
            'prediction': prediction[0],
            'shap_values': shap_values_list[0],
            'execution_time': execution_time,
            'memory_usage': peak,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/predict_with_shap_no_cache')
async def predict_with_shap_no_cache(sales_data: SalesData):
    try:
        start_time = time.time()
        tracemalloc.start()

        df = pd.DataFrame([sales_data.dict()])
        df = pd.get_dummies(df, columns=['produto'])
        
        # Adicionar colunas faltantes do one-hot encoding
        df = df.reindex(columns=X_train_columns, fill_value=0)  # Reindexar e preencher valores ausentes

        prediction = model.predict(df)
        # Calcular valores SHAP sem usar cache
        shap_values = explainer.shap_values(df, check_additivity=False)
        shap_values_list = shap_values.tolist()

        current, peak = tracemalloc.get_traced_memory()  # Capturar uso de memória
        execution_time = time.time() - start_time
        tracemalloc.stop()

        return {
            'prediction': prediction[0],
            'shap_values': shap_values_list[0],
            'execution_time': execution_time,
            'memory_usage': peak,  # Incluir o uso de memória na resposta
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/performance_test')
async def performance_test(sales_data: SalesData, n_requests: int = 100):
    try:
        # Medir desempenho sem SHAP
        total_time_no_shap = 0
        total_memory_no_shap = 0
        for _ in range(n_requests):
            response = await predict(sales_data)
            total_time_no_shap += response['execution_time']
            total_memory_no_shap += response['memory_usage']

        avg_time_no_shap = total_time_no_shap / n_requests
        avg_memory_no_shap = total_memory_no_shap / n_requests

        # Medir desempenho com SHAP usando cache
        total_time_with_shap = 0
        total_memory_with_shap = 0
        for _ in range(n_requests):
            response = await predict_with_shap(sales_data)
            total_time_with_shap += response['execution_time']
            total_memory_with_shap += response['memory_usage']

        avg_time_with_shap = total_time_with_shap / n_requests
        avg_memory_with_shap = total_memory_with_shap / n_requests

        # Medir desempenho com SHAP sem cache
        total_time_with_shap_no_cache = 0
        total_memory_with_shap_no_cache = 0
        for _ in range(n_requests):
            response = await predict_with_shap_no_cache(sales_data)
            total_time_with_shap_no_cache += response['execution_time']
            total_memory_with_shap_no_cache += response['memory_usage']

        avg_time_with_shap_no_cache = total_time_with_shap_no_cache / n_requests
        avg_memory_with_shap_no_cache = total_memory_with_shap_no_cache / n_requests

        # Calcular ganho de tempo e economia de memória
        time_gain_with_shap = avg_time_no_shap - avg_time_with_shap
        memory_savings_with_shap = avg_memory_no_shap - avg_memory_with_shap
        memory_savings_percentage_with_shap = 100 * memory_savings_with_shap / avg_memory_no_shap if avg_memory_no_shap > 0 else 0

        time_gain_no_cache = avg_time_no_shap - avg_time_with_shap_no_cache
        memory_savings_no_cache = avg_memory_no_shap - avg_memory_with_shap_no_cache
        memory_savings_percentage_no_cache = 100 * memory_savings_no_cache / avg_memory_no_shap if avg_memory_no_shap > 0 else 0

        # Salvar resultados em um arquivo txt
        with open('results.txt', 'w') as f:
            f.write(f"Tempo médio sem SHAP: {avg_time_no_shap:.4f} segundos\n")
            f.write(f"Tempo médio com SHAP (cache): {avg_time_with_shap:.4f} segundos\n")
            f.write(f"Tempo médio com SHAP (sem cache): {avg_time_with_shap_no_cache:.4f} segundos\n")
            f.write(f"Uso médio de memória sem SHAP: {avg_memory_no_shap:.2f} bytes\n")
            f.write(f"Uso médio de memória com SHAP (cache): {avg_memory_with_shap:.2f} bytes\n")
            f.write(f"Uso médio de memória com SHAP (sem cache): {avg_memory_with_shap_no_cache:.2f} bytes\n")
            f.write(f"Ganho de tempo com SHAP (cache): {time_gain_with_shap:.4f} segundos\n")
            f.write(f"Economia de memória com SHAP (cache): {memory_savings_with_shap:.2f} bytes\n")
            f.write(f"Porcentagem de economia de memória com SHAP (cache): {memory_savings_percentage_with_shap:.2f}%\n")
            f.write(f"Ganho de tempo com SHAP (sem cache): {time_gain_no_cache:.4f} segundos\n")
            f.write(f"Economia de memória com SHAP (sem cache): {memory_savings_no_cache:.2f} bytes\n")
            f.write(f"Porcentagem de economia de memória com SHAP (sem cache): {memory_savings_percentage_no_cache:.2f}%\n")

        return {
            'average_time_without_shap': avg_time_no_shap,
            'average_time_with_shap': avg_time_with_shap,
            'average_time_with_shap_no_cache': avg_time_with_shap_no_cache,
            'average_memory_without_shap': avg_memory_no_shap,
            'average_memory_with_shap': avg_memory_with_shap,
            'average_memory_with_shap_no_cache': avg_memory_with_shap_no_cache,
            'time_gain_with_shap': time_gain_with_shap,
            'memory_savings_with_shap': memory_savings_with_shap,
            'memory_savings_percentage_with_shap': memory_savings_percentage_with_shap,
            'time_gain_with_shap_no_cache': time_gain_no_cache,
            'memory_savings_with_shap_no_cache': memory_savings_no_cache,
            'memory_savings_percentage_with_shap_no_cache': memory_savings_percentage_no_cache
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/clear_cache')
async def clear_cache():
    shap_cache.clear()
    return {'message': 'Cache cleared'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
