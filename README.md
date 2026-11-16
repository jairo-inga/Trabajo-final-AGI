# FinGuard AI — Detección de Fraude Financiero y Anomalías Transaccionales

## 1. Descripción del Problema y Arquitectura
**FinGuard AI** es un sistema desacoplado de detección de fraude transaccional en tiempo real (Backend en **Python / FastAPI / Scikit-Learn** y Frontend interactivo en **HTML5 / Tailwind CSS / JS**) diseñado bajo los lineamientos del **Bloque 2** del curso de Inteligencia Artificial.

```
finguard-ai/
├── backend/
│   ├── __init__.py
│   ├── data_generator.py     # Generación de dataset realista desbalanceado (CSV)
│   ├── ml_engine.py          # Modo Base + Modelos Bloque 2 (RL, KNN, DT, MLP, DBSCAN, K-Means)
│   └── main.py               # API REST en FastAPI + servidor de archivos estáticos del frontend
├── frontend/
│   ├── index.html            # Interfaz web moderna (Dark Mode profesional con Tailwind CSS CDN)
│   └── app.js                # Consumo asíncrono (fetch) de endpoints y renderizado de tablas/gráficas
├── data/                     # Almacenamiento de transacciones.csv y artefactos .pkl
├── requirements.txt          # Dependencias exactas
└── README.md                 # Plantilla oficial de 1 página del curso
```

## 2. Dataset Transaccional Desbalanceado (`data/transacciones.csv`)
- **Volumen y Desbalance:** 5,000 transacciones sintéticas (`np.random.seed(42)`): **97% legítimas (`clase = 0`, 4,850)** y **3% fraudulentas (`clase = 1`, 150)**.
- **Variables Explicativas (8):** `monto_pen`, `hora_dia`, `distancia_km`, `intentos_previos_1h`, `es_internacional`, `dispositivo_nuevo`, `ratio_monto_promedio`, `antiguedad_cuenta_dias`.

## 3. Modelos Evaluados y Función de Costo Financiera
- **Preprocesamiento sin Fuga de Datos:** Partición estratificada 80/20 (`random_state=42`), ajuste de `StandardScaler` exclusivamente sobre entrenamiento y validación cruzada estratificada ($k=5$ folds).
- **Supervisados:** Modo Base (Regla Heurística Estática), Regresión Logística (`class_weight='balanced'`), KNN ($k=5$), Árbol de Decisión (`max_depth=6, class_weight='balanced'`), Perceptrón Multicapa (`hidden_layer_sizes=(32, 16), max_iter=300`).
- **No Supervisados:** K-Means ($k=2$) vs. DBSCAN ($\varepsilon=1.8, \text{min\_samples}=10$) evaluados con Silhouette Score y tasa de captura de outliers.
- **Simulación Financiera (S/):**
  $$\text{Beneficio\_Neto} = \text{Pérdida\_Evitada (TP)} - \text{Pérdida\_Sufrida (FN)} - \text{Costo\_Falsos\_Bloqueos (FP} \times 25)$$
