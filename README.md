# Predicción de temperatura mínima diaria con LSTM

Práctica de Inteligencia Artificial: una red neuronal LSTM entrenada con una serie de tiempo de temperaturas mínimas diarias para predecir la temperatura de los **próximos 15 días**.

[Abrir en Colab](https://colab.research.google.com/drive/1sGYZWnF5TMqH_aVQXkwS5KpsyapxSTMh#scrollTo=QqC8oupgmDBq)

## Objetivo

- Predecir la temperatura mínima diaria a partir de los **últimos 14 días**.
- Lograr un `val_loss` **menor a 0.01** usando *mean squared error* (MSE).
- Demostrar predicciones hasta **15 días a futuro**.

## Dataset

- **Archivo:** `Daily_minimum_temps.csv` (descargado de [Kaggle](https://www.kaggle.com/)).
- **Contenido:** una fecha y la temperatura mínima registrada ese día.
- **Limpieza:** el archivo original puede traer símbolos o letras en la columna de temperatura. Se eliminan los caracteres no numéricos, se convierten los valores a número y se descartan los registros que no se pueden recuperar. También se ordenan las fechas y se quitan duplicados.
- **División:** 800 registros para validación y el resto para entrenamiento.

## Metodología

1. **Limpieza** del dataset (valores inválidos, fechas, duplicados).
2. **División** en entrenamiento y validación (800 registros).
3. **Normalización** con `MinMaxScaler`, ajustado solo con los datos de entrenamiento para no filtrar información de validación.
4. **Ventanas deslizantes:** cada muestra usa 14 días como entrada y el día siguiente como objetivo.
5. **Entrenamiento** de la red LSTM con `ModelCheckpoint` (guarda el mejor modelo según `val_loss`) y `EarlyStopping`.
6. **Predicción recursiva a 15 días:** el modelo predice un día, lo agrega a la ventana y repite el proceso 15 veces.

## Arquitectura de la red

| Capa | Configuración |
|---|---|
| Entrada | Ventana de 14 días × 1 característica |
| LSTM | 64 unidades, `return_sequences=True` |
| Dropout | 0.2 |
| LSTM | 32 unidades |
| Dense | 1 neurona (temperatura del día siguiente) |

| Hiperparámetro | Valor |
|---|---|
| Función de pérdida | MSE |
| Optimizador | Adam |
| Tamaño de lote | 32 |
| Épocas máximas | 100 |
| Callbacks | `ModelCheckpoint`, `EarlyStopping` (paciencia de 10 épocas) |

## Resultados

> Completa esta sección con los valores que imprime tu notebook.

| Métrica | Valor |
|---|---|
| `val_loss` (MSE, datos escalados) | 0.00764 |
| Meta `val_loss` < 0.01 | Cumplida |
| Error aproximado en °C (RMSE) | 2.30 °C |
| Error absoluto medio a 15 días |  1.23 °C |

Gráficas generadas por el notebook:

- [curva_loss](curva_loss.png): evolución de `loss` y `val_loss` por época.
-[prediccion_15dias_vs_real](prediccion_15dias_vs_real.png): predicción de los últimos 15 días conocidos contra los valores reales.
- [prediccion_15dias](prediccion_15dias.png): predicción de los 15 días posteriores al último dato.

## Estructura del proyecto

```text
.
├── prediccion_temperatura_lstm.py   
├── Daily_minimum_temps.csv          
├── best_model_lstm.keras            
├── curva_loss.png
├── prediccion_15dias_vs_real.png
├── prediccion_15dias.png
└── README.md
```

## Cómo ejecutarlo

### En Google Colab (recomendado)

1. Abre el notebook con el botón de Colab de arriba.
2. Sube `Daily_minimum_temps.csv` al entorno de ejecución.
3. Ejecuta las celdas en orden.

### En local

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
python prediccion_temperatura_lstm.py
```

Asegúrate de que `Daily_minimum_temps.csv` esté en la misma carpeta que el script.

## Tecnologías

Python, TensorFlow/Keras, NumPy, pandas, scikit-learn y Matplotlib.

## Autor

**Pedro Rafael**
