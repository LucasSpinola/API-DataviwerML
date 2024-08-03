import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from fastapi import HTTPException, UploadFile, File
from typing import Optional
from fastapi.responses import FileResponse

SEED = 5
np.random.seed(SEED)
modelo: Optional[RandomForestClassifier] = None

async def train_model(file: UploadFile):
    global modelo

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo CSV: {str(e)}")

    expected_columns = ['id_user', 'class_id', 'enrollment', 'QuestionsList_list01', 'QuestionsList_list02', 
                        'QuestionsList_list03', 'submitted_list_final01', 'submitted_list_final02', 
                        'submitted_list_final03']
    if df.empty or not all(col in df.columns for col in expected_columns):
        raise HTTPException(status_code=400, detail="Arquivo CSV não contém as colunas esperadas ou está vazio")

    df = df.dropna(axis=0, how='any')

    x = df.drop(columns=['id_user', 'class_id', 'enrollment', 'QuestionsList_list01', 'QuestionsList_list02', 
                        'QuestionsList_list03', 'submitted_list_final01', 'submitted_list_final02', 
                        'submitted_list_final03'])
    y = df.iloc[:,53].values

    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

    if treino_x.empty or teste_x.empty:
        raise HTTPException(status_code=500, detail="Erro ao dividir os dados: conjuntos de treino ou teste estão vazios")

    try:
        modelo = RandomForestClassifier(random_state=SEED, bootstrap=True, max_depth=None, 
                                        min_samples_leaf=2, min_samples_split=10, n_estimators=300)
        modelo.fit(treino_x, treino_y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao treinar o modelo: {str(e)}")

    try:
        previsoes = modelo.predict(teste_x)
        acuracia = accuracy_score(teste_y, previsoes) * 100
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar predições: {str(e)}")

    df_result = pd.DataFrame({
        'Real': teste_y,
        'Predito': previsoes
    })

    output_filename = "resultados_treinamento.csv"
    df_result.to_csv(output_filename, index=False)

    return {
        "message": "Modelo treinado com sucesso!",
        "accuracy": acuracia,
        "output_file": output_filename
    }

    
async def predict(file: UploadFile):
    global modelo

    if modelo is None:
        raise HTTPException(status_code=400, detail="O modelo ainda não foi treinado.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um CSV")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler o arquivo CSV: {str(e)}")

    if 'enrollment' not in df.columns:
        raise HTTPException(status_code=400, detail="Arquivo CSV não contém a coluna 'enrollment'")

    df = df.fillna(0)
    
    x_pred = df.drop(columns=['id_user', 'class_id', 'enrollment', 'QuestionsList_list01', 'QuestionsList_list02', 
                        'QuestionsList_list03', 'submitted_list_final01', 'submitted_list_final02', 
                        'submitted_list_final03'])
    enrollment = df['enrollment']

    try:
        previsoes = modelo.predict(x_pred)
        probabilidades = modelo.predict_proba(x_pred)[:, 1] * 100
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar predições: {str(e)}")

    df_pred = pd.DataFrame({
        'enrollment': enrollment,
        'result': previsoes,
        'prob': probabilidades
    })

    output_filename = "resultados_predicao.csv"
    df_pred.to_csv(output_filename, index=False)

    return FileResponse(output_filename, media_type='text/csv', headers={"Content-Disposition": f"attachment; filename={output_filename}"})
