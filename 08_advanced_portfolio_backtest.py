import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import ta as ta_lib # Necesitamos 'ta' para el ATR

warnings.filterwarnings('ignore', category=FutureWarning)

print("Iniciando Backtest Avanzado de Portafolio (Gestión de Riesgo y Salida Dinámica)...")

# --- 1. Carga de Datos ---
try:
    df = pd.read_csv('dataset_para_entrenamiento_multiclass.csv', index_col=['coin_id', 'timestamp'], parse_dates=True)
    df.sort_index(inplace=True)
    print("Dataset multiclase cargado.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset_para_entrenamiento_multiclass.csv'.")
    exit()

# --- 2. Configuración de la Simulación Avanzada ---
print("\nPaso 2: Configurando la simulación avanzada...")
# Gestión de Capital
INITIAL_CAPITAL = 1000.0
TRADE_SIZE = 100.0
MAX_OPEN_POSITIONS = 5 # <<-- LÍMITE DE POSICIONES

# Fricciones del Mercado
COMMISSION_FEE = 0.001
SLIPPAGE_PERCENT = 0.0005

# Parámetros de Salida Dinámica
ATR_PERIOD = 14
ATR_MULTIPLIER_TP = 3.0 # Take Profit a 3 veces el ATR
TRAILING_STOP_PERCENT = 0.10 # Trailing stop del 10% desde el máximo de la operación
MAX_HOLDING_DAYS = 21 # Salida por tiempo

# Umbral de Confianza del Modelo
PROBABILITY_THRESHOLD = 0.70

# Variables de simulación
trades = []
active_positions = {}
portfolio_value = INITIAL_CAPITAL
model = None

# --- 3. Bucle de Simulación Walk-Forward ---
unique_dates = sorted(df.index.get_level_values('timestamp').unique())
train_period_end_date = unique_dates[int(len(unique_dates) * 0.2)]
test_dates = [d for d in unique_dates if d > train_period_end_date]

print("\nPaso 3: Ejecutando el bucle de simulación...")
for current_date in tqdm(test_dates, desc="Procesando años"):
    
    # Entrenamiento del Modelo (sin cambios)
    train_df = df[df.index.get_level_values('timestamp') < current_date]
    if not train_df.empty:
        y_train = train_df['target']
        X_train = train_df.drop(columns=['target']).select_dtypes(include=np.number)
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)

    # <<-- LÓGICA DE SALIDA DINÁMICA -->>
    positions_to_close = []
    for pos, pos_data in active_positions.items():
        coin, entry_date = pos
        if (coin, current_date) in df.index:
            current_price = df.loc[(coin, current_date), 'close']
            
            # Actualizar el precio máximo y el trailing stop
            pos_data['high_price'] = max(pos_data['high_price'], current_price)
            pos_data['trailing_stop_price'] = pos_data['high_price'] * (1 - TRAILING_STOP_PERCENT)

            # Condiciones de Venta
            exit_reason = None
            if current_price >= pos_data['take_profit_price']:
                exit_reason = "Take Profit"
            elif current_price <= pos_data['trailing_stop_price']:
                exit_reason = "Trailing Stop"
            elif (current_date - entry_date).days >= MAX_HOLDING_DAYS:
                exit_reason = "Max Hold Time"

            if exit_reason:
                sell_price_slippage = current_price * (1 - SLIPPAGE_PERCENT)
                sell_fee = TRADE_SIZE * (sell_price_slippage / pos_data['entry_price']) * COMMISSION_FEE
                pnl = (sell_price_slippage - pos_data['entry_price']) / pos_data['entry_price']
                pnl_usd = pnl * TRADE_SIZE - pos_data['entry_fee'] - sell_fee
                
                portfolio_value += pnl_usd
                trades.append({'date': current_date, 'pnl_usd': pnl_usd, 'portfolio_value': portfolio_value, 'reason': exit_reason})
                positions_to_close.append(pos)

    for pos in positions_to_close: del active_positions[pos]

    # <<-- LÓGICA DE ENTRADA CON GESTIÓN DE CAPITAL -->>
    if model and len(active_positions) < MAX_OPEN_POSITIONS:
        daily_data = df[df.index.get_level_values('timestamp') == current_date]
        if not daily_data.empty:
            X_today = daily_data.drop(columns=['target']).select_dtypes(include=np.number)
            probabilities = model.predict_proba(X_today)
            daily_data['buy_probability'] = probabilities[:, 3]
            
            # Rankear las señales del día
            buy_signals = daily_data[daily_data['buy_probability'] >= PROBABILITY_THRESHOLD].sort_values('buy_probability', ascending=False)
            
            # Entrar en las mejores señales hasta llenar las posiciones disponibles
            for coin, row in buy_signals.iterrows():
                if len(active_positions) >= MAX_OPEN_POSITIONS:
                    break # Dejar de buscar si ya hemos llenado nuestras posiciones
                
                coin_id = coin[0]
                if not any(p[0] == coin_id for p in active_positions.keys()):
                    entry_price_slippage = row['close'] * (1 + SLIPPAGE_PERCENT)
                    entry_fee = TRADE_SIZE * COMMISSION_FEE
                    
                    # Calcular ATR para el Take Profit dinámico
                    atr = ta_lib.volatility.AverageTrueRange(high=row['high'], low=row['low'], close=row['close'], window=ATR_PERIOD).average_true_range().iloc[-1]
                    take_profit_price = entry_price_slippage + (atr * ATR_MULTIPLIER_TP)

                    active_positions[(coin_id, current_date)] = {
                        'entry_price': entry_price_slippage,
                        'entry_fee': entry_fee,
                        'high_price': entry_price_slippage, # El precio más alto hasta ahora es el de entrada
                        'take_profit_price': take_profit_price,
                        'trailing_stop_price': entry_price_slippage * (1 - TRAILING_STOP_PERCENT)
                    }

# --- 4. Cálculo de Métricas ---
print("\n--- RESULTADOS DEL BACKTEST AVANZADO ---")
if not trades:
    print("\nNo se realizaron operaciones.")
else:
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('date')
    final_capital = trades_df['portfolio_value'].iloc[-1]
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate = (trades_df['pnl_usd'] > 0).mean()
    num_trades = len(trades_df)
    peak = trades_df['portfolio_value'].cummax()
    drawdown = (trades_df['portfolio_value'] - peak) / peak
    max_drawdown = drawdown.min()

    print(f"Capital Inicial: €{INITIAL_CAPITAL:,.2f}")
    print(f"Capital Final: €{final_capital:,.2f}")
    print(f"Rentabilidad Total: {total_return:.2%}")
    print(f"Número de Operaciones: {num_trades}")
    print(f"Tasa de Acierto (Win Rate): {win_rate:.2%}")
    print(f"Drawdown Máximo: {max_drawdown:.2%}")
    print("\nAnálisis de Salidas:")
    print(trades_df['reason'].value_counts(normalize=True))

    plt.figure(figsize=(12, 7))
    plt.plot(trades_df['date'], trades_df['portfolio_value'], label='Estrategia Avanzada')
    plt.title('Curva de Capital (Gestión de Riesgo y Salida Dinámica)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (€)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('curva_de_capital_avanzada.png')
    print("\nGráfico guardado como 'curva_de_capital_avanzada.png'")

print("\n¡PROYECTO COMPLETADO!")
