# 07_market_regime_classification.py

import pandas as pd
import numpy as np
import logging

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parámetros de Configuración ---
DATA_FILE_PATH = 'data/all_crypto_data_10_years.csv'
OUTPUT_FILE_PATH = 'data/market_regime_data.csv'
BTC_SYMBOL = 'BTCUSDT'
REGIME_MA_PERIOD = 200
LATERAL_THRESHOLD_PERCENT = 0.02  # Umbral del 2% para definir el régimen lateral

def load_and_prepare_data(file_path, btc_symbol):
    """
    Carga los datos desde el archivo CSV y prepara el DataFrame de Bitcoin.
    """
    logging.info(f"Cargando datos desde {file_path}...")
    try:
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        # Filtrar solo los datos de Bitcoin para el análisis de régimen
        df_btc = df[df['symbol'] == btc_symbol].copy()
        logging.info(f"Datos de {btc_symbol} cargados exitosamente. Rango de fechas: {df_btc.index.min()} a {df_btc.index.max()}")
        return df_btc
    except FileNotFoundError:
        logging.error(f"Error: El archivo de datos {file_path} no fue encontrado.")
        return None

def classify_market_regime(df, ma_period, lateral_threshold):
    """
    Calcula la MA200 y clasifica el régimen de mercado diario.
    """
    logging.info(f"Calculando la media móvil de {ma_period} días para la clasificación de régimen...")
    
    # Calcular la Media Móvil Simple de 200 días
    df['regime_ma'] = df['close'].rolling(window=ma_period).mean()
    
    # Calcular la diferencia porcentual entre el precio de cierre y la MA
    df['ma_diff_percent'] = (df['close'] - df['regime_ma']) / df['regime_ma']
    
    # Definir las condiciones para cada régimen
    conditions = [
        (df['ma_diff_percent'] > lateral_threshold),  # Régimen Alcista
        (df['ma_diff_percent'] < -lateral_threshold), # Régimen Bajista
    ]
    
    # Definir los resultados correspondientes a las condiciones
    outcomes = ['Alcista', 'Bajista']
    
    # Aplicar la clasificación
    # np.select es eficiente para aplicar lógica condicional en columnas
    # El valor por defecto 'Lateral' se aplica cuando ninguna condición es verdadera
    df['regime'] = np.select(conditions, outcomes, default='Lateral')
    
    # Manejar el período inicial donde la MA no está disponible (NaN)
    df['regime'] = df['regime'].replace('0', 'Indefinido') # np.select puede devolver '0' si el default no se especifica bien
    df.loc[df['regime_ma'].isna(), 'regime'] = 'Indefinido'

    logging.info("Clasificación de régimen completada.")
    logging.info(f"Distribución de regímenes:\n{df['regime'].value_counts(normalize=True)}")
    
    return df

def save_data(df, output_path):
    """
    Guarda el DataFrame con la nueva columna de régimen en un archivo CSV.
    """
    # Seleccionamos solo las columnas relevantes para mantener el archivo limpio
    # El script de backtesting necesitará unir esta información por fecha.
    df_regime = df[['regime']].copy()
    
    logging.info(f"Guardando los datos de régimen en {output_path}...")
    df_regime.to_csv(output_path)
    logging.info("Datos guardados exitosamente.")

def main():
    """
    Función principal para ejecutar el script.
    """
    logging.info("--- Iniciando Script de Clasificación de Régimen de Mercado ---")
    
    # 1. Cargar datos de BTC
    df_btc = load_and_prepare_data(DATA_FILE_PATH, BTC_SYMBOL)
    
    if df_btc is not None:
        # 2. Clasificar el régimen de mercado
        df_btc_regime = classify_market_regime(df_btc, REGIME_MA_PERIOD, LATERAL_THRESHOLD_PERCENT)
        
        # 3. Guardar los resultados
        save_data(df_btc_regime, OUTPUT_FILE_PATH)
        
        logging.info("--- Proceso completado exitosamente ---")

if __name__ == '__main__':
    main()

