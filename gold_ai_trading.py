from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# 1. CONFIGURACION
# =========================================================
CARPETA_ORO = Path(r"C:\Users\blink\OneDrive\Desktop\dukascopy_gold")

# Tu calendario queda exactamente con ese nombre
RUTA_CALENDARIO = Path(r"C:\Users\blink\OneDrive\Desktop\Python\economic_calendar.csv.csv")

CARPETA_SALIDA = Path(r"C:\Users\blink\Downloads\Proyecto_ORO_V2")

PASOS_TARGET = 3                 # 3 velas de 5 minutos = 15 minutos
THRESHOLD_NIVEL = 1.0           # cercanía a soporte/resistencia
VENTANA_EVENTO_MIN = 30         # ventana antes y después de evento macro

RSI_PERIOD = 14
ATR_PERIOD = 14

PROBA_LONG = 0.55               # si prob_up >= 0.55 => compra
PROBA_SHORT = 0.45              # si prob_up <= 0.45 => venta
COSTO_TRANSACCION = 0.0003      # costo simple en retorno porcentual


# =========================================================
# 2. UTILIDADES
# =========================================================
def asegurar_directorio(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def calcular_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


# =========================================================
# 3. LEER Y UNIR CSV DEL ORO
# =========================================================
def cargar_datos_oro(carpeta_oro: Path) -> pd.DataFrame:
    if not carpeta_oro.exists():
        raise FileNotFoundError(f"No existe la carpeta del oro: {carpeta_oro}")

    archivos = sorted(glob.glob(str(carpeta_oro / "xauusd_*.csv")))

    print("¿Existe la carpeta del oro?")
    print(carpeta_oro.exists())
    print("\nCantidad de archivos encontrados:", len(archivos))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos xauusd_*.csv en {carpeta_oro}")

    lista_dfs = []
    columnas_obligatorias = {"utc", "open", "high", "low", "close", "volume"}

    for archivo in archivos:
        print(archivo)
        df = pd.read_csv(archivo)
        df.columns = [c.lower().strip() for c in df.columns]

        faltantes = columnas_obligatorias - set(df.columns)
        if faltantes:
            raise ValueError(f"El archivo {archivo} no tiene columnas obligatorias: {faltantes}")

        df["utc"] = df["utc"].astype(str).str.replace(" UTC", "", regex=False).str.strip()
        df["datetime"] = pd.to_datetime(
            df["utc"],
            format="%d.%m.%Y %H:%M:%S.%f",
            errors="coerce"
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
        lista_dfs.append(df)

    df_final = pd.concat(lista_dfs, ignore_index=True)
    df_final = (
        df_final.sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    print("\nTamaño dataset unido:")
    print(df_final.shape)

    return df_final


# =========================================================
# 4. FEATURES TECNICAS MEJORADAS
# =========================================================
def crear_features_tecnicas(
    df: pd.DataFrame,
    pasos_target: int = 3,
    threshold_nivel: float = 1.0,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    df = df.copy()

    # Retornos
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["return_12"] = df["close"].pct_change(12)

    # Momentum
    df["momentum_3"] = df["close"].diff(3)
    df["momentum_6"] = df["close"].diff(6)

    # Candle structure
    df["candle_range"] = df["high"] - df["low"]
    df["candle_body"] = df["close"] - df["open"]
    df["body_abs"] = df["candle_body"].abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_to_range"] = df["body_abs"] / df["candle_range"].replace(0, np.nan)

    # Medias
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["dist_ma_5"] = df["close"] - df["ma_5"]
    df["dist_ma_10"] = df["close"] - df["ma_10"]
    df["dist_ma_20"] = df["close"] - df["ma_20"]
    df["dist_ema_9"] = df["close"] - df["ema_9"]
    df["dist_ema_21"] = df["close"] - df["ema_21"]

    # Volatilidad
    df["volatility_5"] = df["return_1"].rolling(5).std()
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    # RSI y ATR
    df["rsi_14"] = calcular_rsi(df["close"], period=rsi_period)
    df["atr_14"] = calcular_atr(df, period=atr_period)
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Volumen
    df["volume_ma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_10"].replace(0, np.nan)

    # Soportes / resistencias
    df["resistance_3"] = df["high"].rolling(3).max()
    df["support_3"] = df["low"].rolling(3).min()
    df["resistance_6"] = df["high"].rolling(6).max()
    df["support_6"] = df["low"].rolling(6).min()

    df["dist_to_resistance"] = df["resistance_3"] - df["close"]
    df["dist_to_support"] = df["close"] - df["support_3"]

    df["near_support"] = (df["dist_to_support"] <= threshold_nivel).astype(int)
    df["near_resistance"] = (df["dist_to_resistance"] <= threshold_nivel).astype(int)

    # Tiempo
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["session_asia"] = df["hour"].between(0, 7).astype(int)
    df["session_london"] = df["hour"].between(8, 12).astype(int)
    df["session_newyork"] = df["hour"].between(13, 17).astype(int)

    # Target
    df["future_close"] = df["close"].shift(-pasos_target)
    df["future_return"] = (df["future_close"] / df["close"]) - 1
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    print("\nConteo del target:")
    print(df["target"].value_counts(dropna=False))

    print("\nPorcentaje del target:")
    print(df["target"].value_counts(normalize=True, dropna=False))

    print("\nTamaño después de features técnicas:")
    print(df.shape)

    return df


# =========================================================
# 5. LEER Y LIMPIAR CALENDARIO ECONOMICO
# =========================================================
def cargar_calendario(ruta_calendar: Path) -> pd.DataFrame:
    if not ruta_calendar.exists():
        raise FileNotFoundError(f"No se encontró el calendario económico en: {ruta_calendar}")

    print(f"\nLeyendo calendario desde: {ruta_calendar}")

    # Tu archivo usa ;
    calendar_df = pd.read_csv(ruta_calendar, sep=";", encoding="utf-8-sig")

    print("\nColumnas originales detectadas:")
    print(calendar_df.columns.tolist())

    # Normalizar nombres de columnas
    calendar_df.columns = (
        pd.Index(calendar_df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Renombrar aliases
    alias_columnas = {
        "datetime_utc": "datetime_utc",
        "datetime": "datetime_utc",
        "date_time": "datetime_utc",
        "pais": "country",
        "currency": "country",
        "evento": "event",
        "importance": "impact",
        "importancia": "impact",
        "categoria": "category",
    }

    calendar_df = calendar_df.rename(
        columns={c: alias_columnas[c] for c in calendar_df.columns if c in alias_columnas}
    )

    # Si no existe category, crearla
    if "category" not in calendar_df.columns:
        calendar_df["category"] = ""

    columnas_obligatorias = {"datetime_utc", "country", "event", "impact", "category"}
    faltantes = columnas_obligatorias - set(calendar_df.columns)

    if faltantes:
        print("\nColumnas después de normalizar:")
        print(calendar_df.columns.tolist())
        raise ValueError(f"Al calendario le faltan columnas obligatorias: {faltantes}")

    # Convertir fecha
    calendar_df["datetime_utc"] = pd.to_datetime(
        calendar_df["datetime_utc"],
        errors="coerce",
        dayfirst=True
    )

    # Limpiar texto
    calendar_df["country"] = calendar_df["country"].astype(str).str.upper().str.strip()
    calendar_df["event"] = calendar_df["event"].astype(str).str.lower().str.strip()
    calendar_df["impact"] = calendar_df["impact"].astype(str).str.lower().str.strip()
    calendar_df["category"] = calendar_df["category"].astype(str).str.lower().str.strip()

    # Normalizar impacto
    calendar_df["impact"] = calendar_df["impact"].replace({
        "high": "high",
        "medium": "medium",
        "low": "low",
        "alto": "high",
        "medio": "medium",
        "media": "medium",
        "bajo": "low"
    })

    calendar_df = (
        calendar_df.dropna(subset=["datetime_utc"])
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    print("\nPrimeras filas del calendario limpio:")
    print(calendar_df.head())

    return calendar_df

# =========================================================
# 6. FEATURES MACRO MEJORADAS
# =========================================================
def crear_features_macro(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    ventana_evento_min: int = 30,
) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    calendar_df = calendar_df.copy().sort_values("datetime_utc").reset_index(drop=True)

    # Distancia al próximo y al último evento
    eventos_base = calendar_df[["datetime_utc"]].drop_duplicates().sort_values("datetime_utc")

    futuro = pd.merge_asof(
        df[["datetime"]],
        eventos_base.rename(columns={"datetime_utc": "event_time"}),
        left_on="datetime",
        right_on="event_time",
        direction="forward"
    )

    pasado = pd.merge_asof(
        df[["datetime"]],
        eventos_base.rename(columns={"datetime_utc": "event_time"}),
        left_on="datetime",
        right_on="event_time",
        direction="backward"
    )

    df["minutes_to_event"] = (
        (futuro["event_time"] - df["datetime"]).dt.total_seconds() / 60
    )
    df["minutes_since_event"] = (
        (df["datetime"] - pasado["event_time"]).dt.total_seconds() / 60
    )

    # Inicializar flags
    flags = [
        "high_impact_event",
        "medium_impact_event",
        "usd_event",
        "fed_event",
        "inflation_event",
        "employment_event",
        "growth_event",
    ]
    for col in flags:
        df[col] = 0

    # Para acelerar el cruce temporal
    time_series = df["datetime"]

    for _, evento in calendar_df.iterrows():
        inicio = evento["datetime_utc"] - pd.Timedelta(minutes=ventana_evento_min)
        fin = evento["datetime_utc"] + pd.Timedelta(minutes=ventana_evento_min)

        left = time_series.searchsorted(inicio, side="left")
        right = time_series.searchsorted(fin, side="right")

        if left >= right:
            continue

        pais = str(evento["country"]).upper()
        impacto = str(evento["impact"]).lower()
        categoria = str(evento["category"]).lower()
        nombre_evento = str(evento["event"]).lower()

        idx = slice(left, right)

        if impacto == "high":
            df.loc[idx, "high_impact_event"] = 1

        if impacto == "medium":
            df.loc[idx, "medium_impact_event"] = 1

        if pais in ["US", "USD"]:
            df.loc[idx, "usd_event"] = 1

        if "rates/fed" in categoria or any(
            x in nombre_evento for x in ["fomc", "fed", "press conference", "minutes"]
        ):
            df.loc[idx, "fed_event"] = 1

        if (
            "inflation" in categoria
            or "cpi" in nombre_evento
            or "ppi" in nombre_evento
            or "inflation" in nombre_evento
        ):
            df.loc[idx, "inflation_event"] = 1

        if (
            "employment" in categoria
            or "nfp" in nombre_evento
            or "employment" in nombre_evento
            or "payroll" in nombre_evento
        ):
            df.loc[idx, "employment_event"] = 1

        if (
            "growth" in categoria
            or "gdp" in nombre_evento
            or "pmi" in nombre_evento
        ):
            df.loc[idx, "growth_event"] = 1

    df["minutes_to_event"] = df["minutes_to_event"].fillna(9999)
    df["minutes_since_event"] = df["minutes_since_event"].fillna(9999)

    print("\nFeatures macro creadas correctamente.")
    return df


# =========================================================
# 7. PREPARAR DATASET DEL MODELO
# =========================================================
def preparar_dataset_modelo(df: pd.DataFrame):
    features = [
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "momentum_3",
        "momentum_6",
        "candle_range",
        "candle_body",
        "body_abs",
        "upper_wick",
        "lower_wick",
        "body_to_range",
        "dist_to_support",
        "dist_to_resistance",
        "near_support",
        "near_resistance",
        "dist_ma_5",
        "dist_ma_10",
        "dist_ma_20",
        "dist_ema_9",
        "dist_ema_21",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "rsi_14",
        "atr_14",
        "atr_pct",
        "volume_ratio",
        "hour",
        "minute",
        "day_of_week",
        "session_asia",
        "session_london",
        "session_newyork",
        "minutes_to_event",
        "minutes_since_event",
        "high_impact_event",
        "medium_impact_event",
        "usd_event",
        "fed_event",
        "inflation_event",
        "employment_event",
        "growth_event",
    ]

    columnas_requeridas = features + ["target", "future_return", "future_close", "close", "datetime"]
    df_model = df.dropna(subset=columnas_requeridas).copy().reset_index(drop=True)

    X = df_model[features]
    y = df_model["target"]

    print("\nTamaño dataset modelable:")
    print(df_model.shape)

    return df_model, X, y, features


# =========================================================
# 8. SPLIT TEMPORAL
# =========================================================
def split_train_test(df_model: pd.DataFrame, features: list[str], test_size: float = 0.2):
    split_idx = int(len(df_model) * (1 - test_size))

    df_train = df_model.iloc[:split_idx].copy()
    df_test = df_model.iloc[split_idx:].copy()

    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train["target"]
    y_test = df_test["target"]

    print("\nTrain:", X_train.shape)
    print("Test :", X_test.shape)

    return df_train, df_test, X_train, X_test, y_train, y_test


# =========================================================
# 9. ENTRENAR Y EVALUAR MODELOS
# =========================================================
def entrenar_modelos(X_train, X_test, y_train, y_test, features):
    modelos = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced"
                ))
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
    }

    resultados = []

    for nombre, modelo in modelos.items():
        print("\n" + "=" * 70)
        print(f"MODELO: {nombre}")
        print("=" * 70)

        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_test)[:, 1]
        else:
            proba = np.zeros(len(X_test))

        acc = accuracy_score(y_test, pred)

        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan

        cm = confusion_matrix(y_test, pred)
        report = classification_report(y_test, pred, zero_division=0)

        print("Accuracy:", acc)
        print("AUC     :", auc)
        print("\nMatriz de confusión:")
        print(cm)
        print("\nReporte de clasificación:")
        print(report)

        resultado = {
            "nombre": nombre,
            "modelo": modelo,
            "accuracy": acc,
            "auc": auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "proba_test": proba,
            "pred_test": pred,
        }

        # Importancias o coeficientes
        if nombre == "logistic_regression":
            coefs = pd.Series(
                modelo.named_steps["modelo"].coef_[0],
                index=features
            ).sort_values(key=np.abs, ascending=False)
            resultado["ranking_features"] = coefs

            print("\nTop 15 coeficientes:")
            print(coefs.head(15))

        elif nombre == "random_forest":
            importancias = pd.Series(
                modelo.feature_importances_,
                index=features
            ).sort_values(ascending=False)
            resultado["ranking_features"] = importancias

            print("\nTop 15 importancias:")
            print(importancias.head(15))

        resultados.append(resultado)

    return resultados


# =========================================================
# 10. BACKTEST BASICO
# =========================================================
def calcular_max_drawdown(curva: pd.Series) -> float:
    rolling_max = curva.cummax()
    drawdown = (curva / rolling_max) - 1
    return drawdown.min()


def backtest_senales(
    df_test: pd.DataFrame,
    probabilidades: np.ndarray,
    nombre_modelo: str,
    proba_long: float = 0.55,
    proba_short: float = 0.45,
    costo_transaccion: float = 0.0003,
):
    bt = df_test[["datetime", "close", "future_close", "future_return", "target"]].copy()
    bt["proba_up"] = probabilidades

    bt["signal"] = np.where(
        bt["proba_up"] >= proba_long,
        1,
        np.where(bt["proba_up"] <= proba_short, -1, 0)
    )

    # Retorno bruto
    bt["strategy_return_gross"] = bt["signal"] * bt["future_return"]

    # Costo por cambio de posición
    bt["prev_signal"] = bt["signal"].shift(1).fillna(0)
    bt["trade_flag"] = (bt["signal"] != bt["prev_signal"]).astype(int)

    # No cobrar costo en barras sin señal y sin cambio
    bt["strategy_return_net"] = bt["strategy_return_gross"] - (bt["trade_flag"] * costo_transaccion)

    # Curvas acumuladas
    bt["equity_market"] = (1 + bt["future_return"].fillna(0)).cumprod()
    bt["equity_strategy"] = (1 + bt["strategy_return_net"].fillna(0)).cumprod()

    # Métricas
    operaciones = bt[bt["signal"] != 0].copy()
    total_trades = len(operaciones)

    if total_trades > 0:
        hit_rate = (operaciones["strategy_return_gross"] > 0).mean()
        avg_trade = operaciones["strategy_return_net"].mean()
        profit_factor = (
            operaciones.loc[operaciones["strategy_return_net"] > 0, "strategy_return_net"].sum()
            /
            abs(operaciones.loc[operaciones["strategy_return_net"] < 0, "strategy_return_net"].sum())
            if abs(operaciones.loc[operaciones["strategy_return_net"] < 0, "strategy_return_net"].sum()) > 0
            else np.nan
        )
    else:
        hit_rate = np.nan
        avg_trade = np.nan
        profit_factor = np.nan

    total_return_strategy = bt["equity_strategy"].iloc[-1] - 1
    total_return_market = bt["equity_market"].iloc[-1] - 1
    exposure = (bt["signal"] != 0).mean()
    max_dd = calcular_max_drawdown(bt["equity_strategy"])

    resumen = {
        "modelo": nombre_modelo,
        "total_return_strategy": total_return_strategy,
        "total_return_market": total_return_market,
        "total_trades": total_trades,
        "hit_rate": hit_rate,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "max_drawdown": max_dd,
    }

    print("\n" + "-" * 70)
    print(f"BACKTEST: {nombre_modelo}")
    print("-" * 70)
    for k, v in resumen.items():
        print(f"{k}: {v}")

    return bt, resumen


# =========================================================
# 11. EXPORTAR RESULTADOS
# =========================================================
def exportar_resultados(
    df_features: pd.DataFrame,
    resumen_modelos: pd.DataFrame,
    resumen_backtest: pd.DataFrame,
    backtests: dict,
    carpeta_salida: Path,
):
    asegurar_directorio(carpeta_salida)

    ruta_features = carpeta_salida / "xauusd_features_modelo_v2.csv"
    ruta_modelos = carpeta_salida / "resumen_modelos_v2.csv"
    ruta_backtest = carpeta_salida / "resumen_backtest_v2.csv"

    df_features.to_csv(ruta_features, index=False)
    resumen_modelos.to_csv(ruta_modelos, index=False)
    resumen_backtest.to_csv(ruta_backtest, index=False)

    for nombre, df_bt in backtests.items():
        ruta_bt = carpeta_salida / f"backtest_{nombre}.csv"
        df_bt.to_csv(ruta_bt, index=False)

    print("\nArchivos exportados correctamente en:")
    print(carpeta_salida)


# =========================================================
# 12. MAIN
# =========================================================
def main():
    # 1) Leer oro
    df_oro = cargar_datos_oro(CARPETA_ORO)

    # 2) Features técnicas
    df_oro = crear_features_tecnicas(
        df_oro,
        pasos_target=PASOS_TARGET,
        threshold_nivel=THRESHOLD_NIVEL,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
    )

    # 3) Calendario macro
    calendar_df = cargar_calendario(RUTA_CALENDARIO)

    # 4) Features macro
    df_oro = crear_features_macro(
        df_oro,
        calendar_df,
        ventana_evento_min=VENTANA_EVENTO_MIN
    )

    # 5) Dataset modelable
    df_model, X, y, features = preparar_dataset_modelo(df_oro)

    # 6) Split temporal
    df_train, df_test, X_train, X_test, y_train, y_test = split_train_test(
        df_model,
        features,
        test_size=0.2
    )

    # 7) Entrenar modelos
    resultados_modelos = entrenar_modelos(X_train, X_test, y_train, y_test, features)

    # 8) Resumen modelos
    resumen_modelos = pd.DataFrame([
        {
            "modelo": r["nombre"],
            "accuracy": r["accuracy"],
            "auc": r["auc"],
        }
        for r in resultados_modelos
    ]).sort_values(["auc", "accuracy"], ascending=False).reset_index(drop=True)

    print("\nResumen comparativo modelos:")
    print(resumen_modelos)

    # 9) Backtest por modelo
    backtests = {}
    resumenes_bt = []

    for r in resultados_modelos:
        bt_df, bt_resumen = backtest_senales(
            df_test=df_test,
            probabilidades=r["proba_test"],
            nombre_modelo=r["nombre"],
            proba_long=PROBA_LONG,
            proba_short=PROBA_SHORT,
            costo_transaccion=COSTO_TRANSACCION,
        )
        backtests[r["nombre"]] = bt_df
        resumenes_bt.append(bt_resumen)

    resumen_backtest = pd.DataFrame(resumenes_bt).sort_values(
        "total_return_strategy",
        ascending=False
    ).reset_index(drop=True)

    print("\nResumen comparativo backtest:")
    print(resumen_backtest)

    # 10) Exportar todo
    exportar_resultados(
        df_features=df_model,
        resumen_modelos=resumen_modelos,
        resumen_backtest=resumen_backtest,
        backtests=backtests,
        carpeta_salida=CARPETA_SALIDA,
    )

    print("\nProyecto ORO V2 finalizado correctamente.")


if __name__ == "__main__":
    main()























