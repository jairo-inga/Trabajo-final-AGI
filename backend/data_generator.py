"""Generador de dataset transaccional sintético desbalanceado para FinGuard AI."""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "monto_pen",
    "hora_dia",
    "distancia_km",
    "intentos_previos_1h",
    "es_internacional",
    "dispositivo_nuevo",
    "ratio_monto_promedio",
    "antiguedad_cuenta_dias",
]
TARGET_COLUMN = "clase"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transacciones.csv"


def generate_transactions_dataset(
    n_samples: int = 5000,
    fraud_ratio: float = 0.03,
    seed: int = 42,
    output_path: Path | str = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Genera un dataset sintético estadísticamente coherente con 97% transacciones
    legítimas (clase=0) y 3% fraudulentas (clase=1), fijando np.random.seed(42).
    """
    np.random.seed(seed)

    n_fraud = int(round(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # --- 1. Transacciones Legítimas (97% -> 4,850 registros) ---
    # Distribución log-normal para montos cotidianos en soles, con cola pesada (>1500 S/)
    legit_monto = np.clip(np.random.lognormal(mean=4.25, sigma=0.98, size=n_legit), 5.0, 5800.0)

    # Distribución horaria diurna/vespertina predominante (menor actividad entre 00:00 y 05:00)
    hour_probs_legit = np.array([
        0.008, 0.006, 0.005, 0.005, 0.006, 0.012,
        0.025, 0.042, 0.058, 0.065, 0.068, 0.070,
        0.075, 0.072, 0.068, 0.065, 0.064, 0.065,
        0.068, 0.062, 0.045, 0.026, 0.012, 0.008,
    ])
    hour_probs_legit = hour_probs_legit / hour_probs_legit.sum()
    legit_hora = np.random.choice(np.arange(24), size=n_legit, p=hour_probs_legit)

    legit_distancia = np.clip(np.random.exponential(scale=7.5, size=n_legit), 0.1, 320.0)
    legit_intentos = np.clip(np.random.poisson(lam=0.35, size=n_legit), 0, 6)
    legit_internacional = np.random.binomial(1, 0.05, size=n_legit)
    legit_dispositivo = np.random.binomial(1, 0.07, size=n_legit)
    legit_ratio = np.clip(np.random.lognormal(mean=-0.02, sigma=0.35, size=n_legit), 0.15, 4.8)
    legit_antiguedad = np.random.randint(45, 3650, size=n_legit)

    # Pequeña fracción (2.5%) de clientes legítimos viajeros / compras atípicas (ruido realista)
    n_travelers = int(n_legit * 0.025)
    traveler_idx = np.random.choice(n_legit, size=n_travelers, replace=False)
    legit_monto[traveler_idx] = np.random.uniform(1200.0, 4200.0, size=n_travelers)
    legit_distancia[traveler_idx] = np.random.uniform(40.0, 280.0, size=n_travelers)
    legit_internacional[traveler_idx] = np.random.binomial(1, 0.45, size=n_travelers)
    legit_dispositivo[traveler_idx] = np.random.binomial(1, 0.35, size=n_travelers)
    legit_ratio[traveler_idx] = np.random.uniform(1.8, 4.5, size=n_travelers)

    # --- 2. Transacciones Fraudulentas (3% -> 150 registros) ---
    # Patrón A: Fraude de alto monto / vaciado de cuenta (85 casos)
    # Patrón B: Fraude medio evasivo (<1500 S/ para evadir regla estática) + ráfagas (45 casos)
    # Patrón C: Micro-pruebas atípicas de tarjetas (carding probing) y casos grises (20 casos)
    n_high = 85
    n_evasive = 45
    n_micro_gray = n_fraud - n_high - n_evasive  # 20

    fraud_monto = np.concatenate([
        np.random.uniform(1100.0, 6200.0, size=n_high),
        np.random.uniform(380.0, 1480.0, size=n_evasive),
        np.concatenate([
            np.random.uniform(5.0, 35.0, size=10),    # Micro-pruebas atípicas (carding probing)
            np.random.uniform(120.0, 550.0, size=10), # Casos grises de monto moderado
        ]),
    ])

    # Mayor incidencia de fraude en madrugada (00:00 a 05:00)
    hour_probs_fraud = np.array([
        0.11, 0.12, 0.13, 0.12, 0.10, 0.07,
        0.02, 0.015, 0.015, 0.015, 0.015, 0.015,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
    ])
    hour_probs_fraud = hour_probs_fraud / hour_probs_fraud.sum()
    fraud_hora = np.random.choice(np.arange(24), size=n_fraud, p=hour_probs_fraud)

    fraud_distancia = np.concatenate([
        np.clip(np.random.lognormal(mean=4.1, sigma=0.75, size=n_high + n_evasive), 12.0, 950.0),
        np.random.uniform(5.0, 140.0, size=n_micro_gray),
    ])

    fraud_intentos = np.concatenate([
        np.clip(np.random.poisson(lam=2.7, size=n_high), 0, 10),
        np.clip(np.random.poisson(lam=3.6, size=n_evasive), 1, 12),
        np.clip(np.random.poisson(lam=2.4, size=n_micro_gray), 0, 9),
    ])

    fraud_internacional = np.concatenate([
        np.random.binomial(1, 0.55, size=n_high),
        np.random.binomial(1, 0.48, size=n_evasive),
        np.random.binomial(1, 0.35, size=n_micro_gray),
    ])

    fraud_dispositivo = np.concatenate([
        np.random.binomial(1, 0.74, size=n_high + n_evasive),
        np.random.binomial(1, 0.45, size=n_micro_gray),
    ])

    fraud_ratio = np.concatenate([
        np.random.uniform(2.4, 8.8, size=n_high),
        np.random.uniform(2.1, 7.2, size=n_evasive),
        np.concatenate([
            np.random.uniform(0.05, 0.35, size=10), # Micro-pruebas tienen ratio muy bajo o atípico
            np.random.uniform(1.4, 3.8, size=10),
        ]),
    ])

    fraud_antiguedad = np.concatenate([
        np.random.randint(3, 720, size=n_high + n_evasive),
        np.random.randint(90, 2200, size=n_micro_gray),
    ])

    # --- 3. Ensamblaje y Mezcla Determinista ---
    df_legit = pd.DataFrame({
        "monto_pen": np.round(legit_monto, 2),
        "hora_dia": legit_hora.astype(int),
        "distancia_km": np.round(legit_distancia, 2),
        "intentos_previos_1h": legit_intentos.astype(int),
        "es_internacional": legit_internacional.astype(int),
        "dispositivo_nuevo": legit_dispositivo.astype(int),
        "ratio_monto_promedio": np.round(legit_ratio, 2),
        "antiguedad_cuenta_dias": legit_antiguedad.astype(int),
        "clase": np.zeros(n_legit, dtype=int),
    })

    df_fraud = pd.DataFrame({
        "monto_pen": np.round(fraud_monto, 2),
        "hora_dia": fraud_hora.astype(int),
        "distancia_km": np.round(fraud_distancia, 2),
        "intentos_previos_1h": fraud_intentos.astype(int),
        "es_internacional": fraud_internacional.astype(int),
        "dispositivo_nuevo": fraud_dispositivo.astype(int),
        "ratio_monto_promedio": np.round(fraud_ratio, 2),
        "antiguedad_cuenta_dias": fraud_antiguedad.astype(int),
        "clase": np.ones(n_fraud, dtype=int),
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    dataset = generate_transactions_dataset()
    counts = dataset["clase"].value_counts()
    print(f"Dataset generado exitosamente en: {DEFAULT_DATA_PATH}")
    print(f"Total registros: {len(dataset)} | Legítimas (0): {counts.get(0, 0)} | Fraudes (1): {counts.get(1, 0)}")
