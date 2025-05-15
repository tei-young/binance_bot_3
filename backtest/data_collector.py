import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

class DataCollector:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.data_dir = './data/historical'
        
    def collect_historical_data(self, symbol: str, timeframe: str, 
                              start_date: str, end_date: str):
        """과거 데이터 수집 및 저장"""
        
        print(f"Collecting {symbol} {timeframe} data from {start_date} to {end_date}")
        
        all_data = []
        current_start = pd.to_datetime(start_date)
        end_timestamp = pd.to_datetime(end_date)
        
        while current_start < end_timestamp:
            # 바이낸스 제한에 맞춰 1000개씩 수집
            since = int(current_start.timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )
            
            if not ohlcv:
                break
                
            all_data.extend(ohlcv)
            
            # 다음 시작점 계산
            last_time = ohlcv[-1][0]
            current_start = pd.to_datetime(last_time, unit='ms')
            
            time.sleep(1)  # Rate limit 대응
        
        # DataFrame 변환 및 저장
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        
        # 저장
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        df.to_parquet(filepath)
        
        print(f"Saved {len(df)} records to {filepath}")
        return df