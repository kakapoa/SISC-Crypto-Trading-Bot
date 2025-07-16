# -*- coding: utf-8 -*-
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_top_100_coin_ids():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
    try:
        response = requests.get(url, params=params )
        response.raise_for_status()
        return {item['symbol'].upper(): item['id'] for item in response.json()}
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la lista de monedas de CoinGecko: {e}")
        return None

def get_binance_full_historical_data(symbol, years=10):
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = datetime.utcnow( )
    start_date = end_time - timedelta(days=years * 365)
    
    print(f"   -> Pidiendo datos desde {start_date.strftime('%Y-%m-%d')} hasta hoy.")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        print(f"      -> Pidiendo 1000 dias de datos hasta {end_time.strftime('%Y-%m-%d')}. Esperando respuesta de la API...")
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"      -> Error en la peticion a Binance para {symbol}: {e}")
            return None

        if not data:
            print("      -> Se ha llegado al inicio del historial disponible. Saliendo del bucle.")
            break

        all_data.extend(data)
        
        first_timestamp_ms = data[0][0]
        end_time = datetime.fromtimestamp(first_timestamp_ms / 1000) - timedelta(days=1)
        
        if end_time < start_date:
            print("      -> Se ha alcanzado la fecha de inicio deseada.")
            break
            
        time.sleep(0.5)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    print(f"   -> Exito final! Se encontraron un total de {len(df)} dias de datos para {symbol}.")
    return df

def main():
    print("Iniciando la descarga de datos historicos completos (10 anos)...")
    coins_map = get_top_100_coin_ids()
    if not coins_map:
        return

    all_coins_df = []
    found_coins_count = 0
    
    for i, (symbol_upper, coin_id) in enumerate(coins_map.items()):
        print(f"({i+1}/{len(coins_map)}) Buscando historial completo para: {coin_id} ({symbol_upper}USDT)...")
        
        # Intentamos con el par USDT, que es el más común
        trading_pair = f"{symbol_upper}USDT"
        historical_df = get_binance_full_historical_data(trading_pair, years=10)
        
        if historical_df is not None and not historical_df.empty:
            historical_df['coin_id'] = coin_id
            all_coins_df.append(historical_df)
            found_coins_count += 1
        else:
            print(f"   -> No se encontraron datos para {trading_pair}. Saltando.")
        
        time.sleep(1)

    if not all_coins_df:
        print("No se pudieron descargar datos para ninguna moneda.")
        return

    final_df = pd.concat(all_coins_df, ignore_index=True)
    output_filename = 'crypto_market_data_binance_10y.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("\n¡Descarga completada!")
    print(f"Se han guardado los datos de {found_coins_count} monedas en el archivo '{output_filename}'.")
    print(f"Total de filas: {len(final_df)}")

if __name__ == "__main__":
    main()

