# League of Legends Match Predictor — PyTorch

Modelo de regresión logística implementado desde cero con PyTorch para predecir victorias en League of Legends basándose en estadísticas de partida.

## Pipeline completo

- Preprocesamiento y estandarización de features
- Entrenamiento con backpropagation manual (1000 épocas)
- Pérdida ponderada (Weighted BCE) para manejo de clases
- Regularización L2 (Ridge) con weight_decay
- Evaluación: Accuracy, ROC-AUC, Matriz de Confusión
- Hyperparameter tuning (comparación de learning rates)
- Feature importance analysis por pesos del modelo

## Resultados

| Modelo | Train Accuracy | Test Accuracy |
|--------|---------------|--------------|
| Sin L2 | 55.50% | 51.50% |
| Con L2 | 55.50% | 51.50% |
| **AUC-ROC** | — | **~0.50** |


## Stack

- Python 3.13
- PyTorch 2.8.0 (CPU)
- Scikit-learn
- Pandas
- Matplotlib

## Instalación
```bash
pip install pandas scikit-learn matplotlib
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

## Dataset

League of Legends match statistics via IBM Skills Network  
1,000 partidas | 8 features: kills, deaths, assists, gold_earned, cs, wards_placed, wards_killed, damage_dealt
