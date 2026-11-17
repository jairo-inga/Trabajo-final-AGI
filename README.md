# FinGuard AI — Sistema Inteligente de Detección de Fraude y Anomalías Transaccionales

## 1. Problema y Quién lo Sufre
Las entidades financieras, pasarelas de pago y comercios electrónicos en el Perú sufren pérdidas millonarias por fraude con tarjetas y toma de control de cuentas (*account takeover*), enfrentando un **severo desbalance de clases** (~97% transacciones legítimas vs. ~3% fraudulentas). Los analistas de riesgo y clientes sufren dos dolores opuestos: dejar pasar un fraude (**Falso Negativo, FN**) ocasiona la pérdida directa del 100% del monto transado, mientras que bloquear erróneamente a un cliente legítimo (**Falso Positivo, FP**) genera fricción comercial y un costo operativo promedio de **S/ 25.00** por reclamo/verificación.

## 2. Modo Base (Regla Heurística Estática)
Como línea base (*baseline* no-IA), el sistema implementa el motor de reglas tradicional bancario:
$$\text{Bloquear } (\hat{y}=1) \iff \text{monto\_pen} > 1500 \;\lor\; (\text{hora\_dia} \le 4 \;\land\; \text{es\_internacional} = 1)$$
Esta regla es rígida: no se adapta a fraudes evasivos nacionales menores a S/ 1,500 ni a ráfagas de micro-pruebas (*carding*), y penaliza a clientes legítimos con compras altas.

## 3. Técnicas Comparadas (Parte 2 / Bloque 2)
Sobre un dataset determinista (`np.random.seed(42)`) de **5,000 transacciones** y 8 variables (`monto_pen`, `hora_dia`, `distancia_km`, `intentos_previos_1h`, `es_internacional`, `dispositivo_nuevo`, `ratio_monto_promedio`, `antiguedad_cuenta_dias`), se aplica partición estratificada **80% train / 20% test**, escalamiento `StandardScaler` sin fuga de datos (*data leakage*) y validación cruzada estratificada ($k=5$ folds):
- **Supervisadas:** Regresión Logística (`class_weight='balanced'`), K-Nearest Neighbors (`n_neighbors=5`), Árbol de Decisión (`max_depth=6, class_weight='balanced'`), y Perceptrón Multicapa (`MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300)`).
- **No Supervisadas (Auditoría de Anomalías):** Clustering particional **K-Means ($k=2$)** vs. basado en densidad **DBSCAN ($\varepsilon=1.8, \text{min\_samples}=10$)** evaluados mediante *Silhouette Score* y tasa de captura de fraude como *outliers*.

## 4. Tabla de Resultados (Test Set: 1,000 tx | Umbral $\tau = 0.50$)
Función de impacto económico: $\text{Beneficio\_Neto} = \text{Pérdida\_Evitada (TP)} - \text{Pérdida\_Sufrida (FN)} - (\text{FP} \times 25)$.

| Técnica / Modelo | CV F1 ($k=5$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Matriz (TN / FP / FN / TP) | Beneficio Neto (S/) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modo Base (Regla Estática)** | `0.5291 ±0.082` | `0.9620` | `0.4048` | `0.5667` | `0.4722` | `0.7704` | `945 / 25 / 13 / 17` | **S/ 40,518.78** |
| **Regresión Logística** | `0.7141 ±0.044` | `0.9780` | `0.5769` | `1.0000` | `0.7317` | `0.9981` | `948 / 22 / 0 / 30` | **S/ 58,914.74** |
| **K-Nearest Neighbors ($k=5$)** | `0.8644 ±0.052` | `0.9960` | `1.0000` | `0.8667` | `0.9286` | `0.9833` | `970 / 0 / 4 / 26` | **S/ 39,558.60** |
| **Árbol de Decisión (`depth=6`)** | `0.8676 ±0.049` | `0.9900` | `0.7941` | `0.9000` | `0.8438` | `0.9471` | `963 / 7 / 3 / 27` | **S/ 46,138.60** |
| **Perceptrón Multicapa (MLP)** | `0.9235 ±0.040` | `0.9950` | `0.8788` | `0.9667` | `0.9206` | `0.9993` | `966 / 4 / 1 / 29` | **S/ 47,590.34** |

- **Clustering No Supervisado:** **K-Means ($k=2$)** alcanza un *Silhouette Score* de `0.6715` capturando el **89.33%** (`134/150`) de los fraudes en el cluster anómalo; **DBSCAN ($\varepsilon=1.8, \text{min\_samples}=10$)** obtiene un *Silhouette Score* de `0.4534` pero eleva la captura de fraudes atípicos al **94.67%** (`142/150`) aislándolos como ruido (`label = -1`).

## 5. Cómo Ejecutarlo en Local y Nube
- **Ejecución Local:**
  ```bash
  pip install -r requirements.txt
  python backend/ml_engine.py
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```
  Abrir en el navegador: `http://localhost:8000` (Documentación interactiva OpenAPI en `http://localhost:8000/docs`).
- **Despliegue en Render / Hugging Face Spaces:** Configurar como *Web Service* con **Build Command** `pip install -r requirements.txt` y **Start Command** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.

## 6. Declaración de Uso de IA
Se empleó asistencia de IA generativa como copiloto de arquitectura de software para estructurar los endpoints REST en FastAPI, maquetar la interfaz reactiva con Tailwind CSS / Chart.js y sistematizar las pruebas de validación cruzada. El diseño experimental, la formulación de costos financieros y la interpretación de resultados fueron validados íntegramente por el equipo.

## 7. Roles de Integrantes
- **Integrante 1 (Arquitecto ML & Datos):** Diseño del generador sintético desbalanceado (`data_generator.py`), preprocesamiento sin *data leakage* y validación cruzada estratificada.
- **Integrante 2 (Ingeniero de Modelos Supervisados y Clustering):** Implementación y ajuste de Regresión Logística, KNN, Árbol de Decisión, MLP, K-Means y DBSCAN (`ml_engine.py`).
- **Integrante 3 (Desarrollador Backend & API):** Construcción de servicios FastAPI (`main.py`), serialización de artefactos `.pkl` y pruebas de endpoints.
- **Integrante 4 (Desarrollador Frontend & Analítica Financiera):** Dashboard interactivo Dark Mode (`index.html`, `app.js`), simulador en vivo y análisis de beneficio neto.
