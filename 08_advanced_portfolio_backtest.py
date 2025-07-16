import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import ta as ta_lib

warnings.filterwarnings('ignore', category=FutureWarning)

print("Iniciando Backtest Avanzado de Portafolio (V2 - ATR Corregido)...")

# --- 1. Carga de Datos ---
try:
    df = pd.read_csv('dataset_para_entrenamiento_multiclass.csv', index_col=['coin_id', 'timestamp'], parse_dates=True)
    df.sort_index(inplace=True)
    print("Dataset multiclase cargado.")
except FileNotFoundError:
    print("Error: No se encontro el archivo 'dataset_para_entrenamiento_multiclass.csv'.")
    exit()

# --- 2. PRE-CALCULO DEL ATR ---
print("\nPaso 1.5: Pre-calculando el ATR para todas las monedas...")
ATR_PERIOD = 14
df['atr'] = df.groupby('coin_id', group_keys=False).apply(
    lambda x: ta_lib.volatility.AverageTrueRange(
        high=x['high'], low=x['low'], close=x['close'], window=ATR_PERIOD
    ).average_true_range()
)
df.dropna(subset=['atr'], inplace=True)
print("Calculo de ATR completado.")

# --- 3. Configuracion de la Simulacion Avanzada ---
print("\nPaso 2: Configurando la simulacion avanzada...")
INITIAL_CAPITAL = 1000.0
TRADE_SIZE = 100.0
MAX_OPEN_POSITIONS = 5
COMMISSION_FEE = 0.001
SLIPPAGE_PERCENT = 0.0005
ATR_MULTIPLIER_TP = 3.0
TRAILING_STOP_PERCENT = 0.10
MAX_HOLDING_DAYS = 21
PROBABILITY_THRESHOLD = 0.70

trades = []
active_positions = {}
portfolio_value = INITIAL_CAPITAL
model = None

# --- 4. Bucle de Simulacion Walk-Forward ---
unique_dates = sorted(df.index.get_level_values('timestamp').unique())
train_period_end_date = unique_dates[int(len(unique_dates) * 0.2)]
test_dates = [d for d in unique_dates if d > train_period_end_date]

print("\nPaso 3: Ejecutando el bucle de simulacion...")
for current_date in tqdm(test_dates, desc="Procesando anos"):
    
    train_df = df[df.index.get_level_values('timestamp') < current_date]
    if not train_df.empty:
        y_train = train_df['target']
        X_train = train_df.drop(columns=['target', 'atr']).select_dtypes(include=np.number)
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)

    positions_to_close = []
    for pos, pos_data in active_positions.items():
        coin, entry_date = pos
        if (coin, current_date) in df.index:
            current_price = df.loc[(coin, current_date), 'close']
            pos_data['high_price'] = max(pos_data['high_price'], current_price)
            pos_data['trailing_stop_price'] = pos_data['high_price'] * (1 - TRAILING_STOP_PERCENT)

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

    if model and len(active_positions) < MAX_OPEN_POSITIONS:
        daily_data = df[df.index.get_level_values('timestamp') == current_date]
        if not daily_data.empty:
            X_today = daily_data.drop(columns=['target', 'atr']).select_dtypes(include=np.number)
            probabilities = model.predict_proba(X_today)
            daily_data['buy_probability'] = probabilities[:, 3]
            
            buy_signals = daily_data[daily_data['buy_probability'] >= PROBABILITY_THRESHOLD].sort_values('buy_probability', ascending=False)
            
            for coin, row in buy_signals.iterrows():
                if len(active_positions) >= MAX_OPEN_POSITIONS:
                    break
                
                coin_id = coin[0]
                if not any(p[0] == coin_id for p in active_positions.keys()):
                    entry_price_slippage = row['close'] * (1 + SLIPPAGE_PERCENT)
                    entry_fee = TRADE_SIZE * COMMISSION_FEE
                    
                    atr = row['atr']
                    take_profit_price = entry_price_slippage + (atr * ATR_MULTIPLIER_TP)

                    active_positions[(coin_id, current_date)] = {
                        'entry_price': entry_price_slippage,
                        'entry_fee': entry_fee,
                        'high_price': entry_price_slippage,
                        'take_profit_price': take_profit_price,
                        'trailing_stop_price': entry_price_slippage * (1 - TRAILING_STOP_PERCENT)
                    }

# --- 5. Calculo de Metricas ---
print("\n--- RESULTADOS DEL BACKTEST AVANZADO (V2) ---")
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
    print(f"Numero de Operaciones: {num_trades}")
    print(f"Tasa de Acierto (Win Rate): {win_rate:.2%}")
    print(f"Drawdown Maximo: {max_drawdown:.2%}")
    print("\nAnalisis de Salidas:")
    print(trades_df['reason'].value_counts(normalize=True))

    plt.figure(figsize=(12, 7))
    plt.plot(trades_df['date'], trades_df['portfolio_value'], label='Estrategia Avanzada')
    plt.title('Curva de Capital (Gestion de Riesgo y Salida Dinamica)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (€)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('curva_de_capital_avanzada.png')
    print("\nGrafico guardado como 'curva_de_capital_avanzada.png'")

print("\n¡PROYECTO COMPLETADO!")
