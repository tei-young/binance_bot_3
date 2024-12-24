import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ta
import logging
import os
from logging.handlers import RotatingFileHandler

# 거래 설정
TIMEFRAME = '5m'      # '1m' 또는 '5m'
LEVERAGE = 10         # 레버리지 설정
MARGIN_AMOUNT = 10    # 실제 사용할 증거금 (USDT)
SLOPE_PERIOD = 10     # Slope 계산을 위한 기간
THRESHOLD = 4         # MA angles JD threshold

# 거래 심볼 목록
TRADING_SYMBOLS = [ #'BTC/USDT',
                 'TIA/USDT', 'DOGS/USDT', 'BAN/USDT', 'BOME/USDT', 'ORCA/USDT', 'AMB/USDT',
                 'BOND/USDT', 'NEAR/USDT', 'HIPPO/USDT', 'BAKE/USDT', 'FXS/USDT', '1000PEPE/USDT',
                'ACX/USDT', 'LINK/USDT', 'POL/USDT', 'MOODENG/USDT', 'ATOM/USDT', 
                'ORDI/USDT', 'DOGE/USDT', 'XLM/USDT', 'GALA/USDT', 'TNSR/USDT', 
                'DOT/USDT', 'ZRO/USDT', 'BNB/USDT', 'THETA/USDT', 'ARPA/USDT', 
                'XRP/USDT', 'SOL/USDT', 'ADA/USDT', 'WLD/USDT', 'RENDER/USDT', 
                'NEAR/USDT', 'SUI/USDT', 'AVAX/USDT', 'MOVE/USDT']

class TradingBot:
    def __init__(self, api_key, api_secret):
        self.setup_logging()
        
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # 각 심볼에 대해 레버리지 설정
        for symbol in TRADING_SYMBOLS:
            try:
                self.exchange.fapiPrivate_post_leverage({
                    'symbol': symbol.replace('/', ''),
                    'leverage': LEVERAGE
                })
                self.trading_logger.info(f"Leverage set for {symbol}: {LEVERAGE}x")
            except Exception as e:
                self.trading_logger.error(f"Error setting leverage for {symbol}: {e}")

    def setup_logging(self):
        """로깅 설정"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 로거 설정 함수
        def setup_logger(name, log_file):
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            handler = RotatingFileHandler(
                f'logs/{log_file}',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger
        
        self.trading_logger = setup_logger('trading', 'trading.log')
        self.signal_logger = setup_logger('signal', 'signals.log')
        self.execution_logger = setup_logger('execution', 'executions.log')

    def calculate_jurik_ma(self, data, period=30):
        """Jurik Moving Average 계산 (simplified version)"""
        # 실제 JMA는 더 복잡한 계산이 필요하지만, 여기서는 EMA로 단순화
        return ta.trend.ema_indicator(data, window=period)

    def calculate_slope(self, data, period=SLOPE_PERIOD):
        """주어진 데이터의 기울기 계산"""
        return (data - data.shift(period)) / period

    def get_historical_data(self, symbol):
        """과거 데이터 조회"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=TIMEFRAME,
                limit=300  # 충분한 기록을 위해 300개 캔들
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.trading_logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        """모든 지표 계산"""
        try:
            # SMA 200 계산
            df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # EMA 계산
            df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD 계산
            macd = ta.trend.MACD(
                df['close'],
                window_fast=12,
                window_slow=26,
                window_sign=9
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # MA angles JD 계산
            df['jma'] = self.calculate_jurik_ma(df['close'])
            df['ma83'] = ta.trend.sma_indicator(df['close'], window=83)
            df['ma278'] = ta.trend.sma_indicator(df['close'], window=278)
            
            # Slopes 계산
            df['jma_slope'] = self.calculate_slope(df['jma'])
            df['ma83_slope'] = self.calculate_slope(df['ma83'])
            df['ma278_slope'] = self.calculate_slope(df['ma278'])
            
            # MA angles JD 색상 결정
            df['mangles_jd_color'] = np.where(
                (df['jma_slope'] > THRESHOLD/100) & 
                (df['ma83_slope'] > THRESHOLD/100) & 
                (df['ma278_slope'] > THRESHOLD/100),
                'green',
                np.where(
                    (df['jma_slope'] < -THRESHOLD/100) & 
                    (df['ma83_slope'] < -THRESHOLD/100) & 
                    (df['ma278_slope'] < -THRESHOLD/100),
                    'red',
                    'neutral'
                )
            )
            
            return df
        except Exception as e:
            self.trading_logger.error(f"Error calculating indicators: {e}")
            return None

    def check_cross_conditions(self, df, position_type):
        """EMA와 MACD 크로스 조건 확인"""
        current_idx = len(df) - 1
        
        crosses_found = {
            'ema_cross_idx': None,
            'macd_cross_idx': None
        }
        
        # 최근 5개 캔들에서 크로스 찾기
        for i in range(current_idx-5, current_idx+1):
            # EMA 크로스 체크
            if position_type == 'long':
                ema_cross = (
                    df['ema12'].iloc[i-1] < df['ema26'].iloc[i-1] and
                    df['ema12'].iloc[i] > df['ema26'].iloc[i]
                )
                macd_cross = (
                    df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1] and
                    df['macd'].iloc[i] > df['macd_signal'].iloc[i]
                )
            else:  # short
                ema_cross = (
                    df['ema12'].iloc[i-1] > df['ema26'].iloc[i-1] and
                    df['ema12'].iloc[i] < df['ema26'].iloc[i]
                )
                macd_cross = (
                    df['macd'].iloc[i-1] > df['macd_signal'].iloc[i-1] and
                    df['macd'].iloc[i] < df['macd_signal'].iloc[i]
                )
            
            if ema_cross and crosses_found['ema_cross_idx'] is None:
                crosses_found['ema_cross_idx'] = i
            if macd_cross and crosses_found['macd_cross_idx'] is None:
                crosses_found['macd_cross_idx'] = i
        
        return crosses_found

    def check_entry_conditions(self, df, symbol):
        """진입 조건 확인"""
        current_idx = len(df) - 1
        current_price = df['close'].iloc[-1]
        
        # 1. SMA200 기준 추세 확인
        above_sma200 = current_price > df['sma200'].iloc[-1]
        self.signal_logger.info(
            f"\n=== Signal Analysis for {symbol} ===\n"
            f"1. SMA 200 Position: {'Above - Long only' if above_sma200 else 'Below - Short only'}"
        )
        
        # 2. MA angles JD 색상 확인
        mangles_color = df['mangles_jd_color'].iloc[-1]
        self.signal_logger.info(
            f"2. MA angles JD Color: {mangles_color.upper()} - "
            f"{'True' if (above_sma200 and mangles_color == 'green') or (not above_sma200 and mangles_color == 'red') else 'False'}"
        )
        
        # 첫 두 조건이 맞지 않으면 early return
        if above_sma200 and mangles_color != 'green':
            return None, None
        if not above_sma200 and mangles_color != 'red':
            return None, None
        
        # 3, 4. EMA와 MACD 크로스 확인
        if above_sma200:  # Long position check
            crosses = self.check_cross_conditions(df, 'long')
            if crosses['ema_cross_idx'] is not None:
                self.signal_logger.info("3. EMA Cross: Golden - True")
            else:
                self.signal_logger.info("3. EMA Cross: No Golden Cross - False")
                
            if crosses['macd_cross_idx'] is not None:
                self.signal_logger.info("4. MACD Cross: Golden - True")
            else:
                self.signal_logger.info("4. MACD Cross: No Golden Cross - False")
                
            if crosses['ema_cross_idx'] is not None and crosses['macd_cross_idx'] is not None:
                self.signal_logger.info(f"5. Condition Match! {symbol} LONG position entry")
                return 'long', crosses
                
        else:  # Short position check
            crosses = self.check_cross_conditions(df, 'short')
            if crosses['ema_cross_idx'] is not None:
                self.signal_logger.info("3. EMA Cross: Dead - True")
            else:
                self.signal_logger.info("3. EMA Cross: No Dead Cross - False")
                
            if crosses['macd_cross_idx'] is not None:
                self.signal_logger.info("4. MACD Cross: Dead - True")
            else:
                self.signal_logger.info("4. MACD Cross: No Dead Cross - False")
                
            if crosses['ema_cross_idx'] is not None and crosses['macd_cross_idx'] is not None:
                self.signal_logger.info(f"5. Condition Match! {symbol} SHORT position entry")
                return 'short', crosses
        
        return None, None

    def determine_stop_loss(self, df, crosses, position_type):
        """손절가 계산"""
        # 먼저 발생한 크로스 찾기
        first_cross_idx = min(
            crosses['ema_cross_idx'],
            crosses['macd_cross_idx']
        )
        
        if position_type == 'long':
            return df['low'].iloc[first_cross_idx]
        else:
            return df['high'].iloc[first_cross_idx]

    def calculate_take_profit(self, entry_price, stop_loss, position_type):
        """목표가 계산"""
        stop_loss_distance = abs(entry_price - stop_loss)
        if position_type == 'long':
            return entry_price + (stop_loss_distance * 2)
        else:
            return entry_price - (stop_loss_distance * 2)

    def execute_trade(self, symbol, position_type, entry_price, stop_loss, take_profit):
        """주문 실행"""
        try:
            # 레버리지를 고려한 실제 포지션 크기 계산
            total_position_size = MARGIN_AMOUNT * LEVERAGE
            position_size = total_position_size / entry_price
            
            # 주문 실행
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='buy' if position_type == 'long' else 'sell',
                amount=position_size,
                price=entry_price
            )
            
            # 손절 주문
            stop_loss_order = self.exchange.create_order(
                symbol=symbol,
                type='stop_loss',
                side='sell' if position_type == 'long' else 'buy',
                amount=position_size,
                price=stop_loss
            )
            
            # 익절 주문
            take_profit_order = self.exchange.create_order(
                symbol=symbol,
                type='take_profit',
                side='sell' if position_type == 'long' else 'buy',
                amount=position_size,
                price=take_profit
            )
            
            # 체결 로깅
            self.execution_logger.info(
                f"Trade Executed - {symbol}\n"
                f"Type: {position_type}\n"
                f"Entry: {entry_price}\n"
                f"Stop Loss: {stop_loss}\n"
                f"Take Profit: {take_profit}\n"
                f"Position Size: {position_size}\n"
                f"Margin Used: {MARGIN_AMOUNT} USDT\n"
                f"Total Position Value: {total_position_size} USDT\n"
                f"Order ID: {order['id']}"
            )
            
            return True
            
        except Exception as e:
            self.trading_logger.error(f"Error executing trade: {e}")
            return False

    def run(self):
        """메인 로직"""
        self.trading_logger.info(
            f"Bot started running\n"
            f"Leverage: {LEVERAGE}x\n"
            f"Margin Amount: {MARGIN_AMOUNT} USDT\n"
            f"Trading Symbols: {TRADING_SYMBOLS}"
        )
        
        while True:
            try:
                for symbol in TRADING_SYMBOLS:
                    # 데이터 수집 및 지표 계산
                    df = self.get_historical_data(symbol)
                    if df is None:
                        continue
                        
                    df = self.calculate_indicators(df)
                    if df is None:
                        continue
                    
                    # 진입 조건 확인
                    position_type, crosses = self.check_entry_conditions(df, symbol)
                    
                    if position_type and crosses:
                        entry_price = df['close'].iloc[-1]
                        stop_loss = self.determine_stop_loss(df, crosses, position_type)
                        take_profit = self.calculate_take_profit(entry_price, stop_loss, position_type)
                        
                        # 주문 실행
                        success = self.execute_trade(
                            symbol,
                            position_type,
                            entry_price,
                            stop_loss,
                            take_profit
                        )
                        
                        if success:
                            self.trading_logger.info(
                                f"Successfully entered {position_type} position for {symbol}"
                            )
                
                # 타임프레임에 따른 대기
                sleep_time = 60 if TIMEFRAME == '1m' else 300
                time.sleep(sleep_time)
                
            except Exception as e:
                self.trading_logger.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    API_KEY = "vf7WGUJSNjVUjsqH8H6BKj0eKpmIecotvP5S1NQlLy041py9LuuFsiK2rksaSomq"
    API_SECRET = "bBOK1BVnOYaVIvKUKoSQsTAt450Ps64WkjQjINDUXkpzvxmQDJUrZT9vSbvkwAKZ"
    
    bot = TradingBot(API_KEY, API_SECRET)
    bot.run()