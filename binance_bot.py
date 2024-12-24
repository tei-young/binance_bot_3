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
        
        # 크로스 히스토리 초기화
        self.cross_history = {
            symbol: {
                'ema': [],
                'macd': []
            }
            for symbol in TRADING_SYMBOLS
        }
        
        # 레버리지 설정
        for symbol in TRADING_SYMBOLS:
            try:
                self.exchange.set_leverage(LEVERAGE, symbol)
                self.trading_logger.info(f"Leverage set for {symbol}: {LEVERAGE}x")
            except Exception as e:
                self.trading_logger.error(f"Error setting leverage for {symbol}: {e}")

    def setup_logging(self):
        """로깅 설정"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 오늘 날짜 가져오기
        today = datetime.now().strftime('%y%m%d')
        
        # 로거 설정 함수
        def setup_logger(name, log_file):
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            
            # 파일명에 날짜 추가
            dated_log_file = f"{today}_{log_file}"
            handler = RotatingFileHandler(
                f'logs/{dated_log_file}',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger
        
        # 각각의 로거 설정
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
                limit=300
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            # timestamp를 인덱스로 설정
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.trading_logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def check_balance(self, symbol):
        """잔액 체크"""
        try:
            balance = self.exchange.fetch_balance()
            free_usdt = balance['USDT']['free']
            
            if free_usdt < MARGIN_AMOUNT:
                self.trading_logger.error(
                    f"===== Insufficient Balance =====\n"
                    f"Symbol: {symbol}\n"
                    f"Required: {MARGIN_AMOUNT} USDT\n"
                    f"Available: {free_usdt} USDT"
                )
                return False
            return True
        except Exception as e:
            self.trading_logger.error(f"Balance check error: {str(e)}")
            return False
        
    def check_symbol_tradable(self, symbol):
        """심볼 거래 가능 여부 체크"""
        try:
            market = self.exchange.market(symbol)
            if not market['active']:
                self.trading_logger.error(
                    f"===== Symbol Not Tradable =====\n"
                    f"Symbol: {symbol}\n"
                    f"Status: Trading suspended or delisted"
                )
                return False
            return True
        except Exception as e:
            self.trading_logger.error(f"Symbol status check error: {str(e)}")
            return False

    def calculate_indicators(self, df):
        """지표 계산"""
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
            
            # JMA slope 계산 및 MA angles JD 색상 결정
            df['jma'] = self.calculate_jurik_ma(df['close'])
            df['jma_slope'] = self.calculate_slope(df['jma'])
            df['mangles_jd_color'] = np.where(
                df['jma_slope'] > THRESHOLD/100,
                'green',
                'red'
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

    def store_cross_data(self, df, symbol):
        """크로스 데이터 저장"""
        if symbol not in self.cross_history:
            self.cross_history[symbol] = {
                'ema': [],  # [(time, type), ...]
                'macd': []  # [(time, type), ...]
            }
        
        current_idx = len(df) - 1
        current_time = df.index[current_idx]
        
        # EMA 크로스 체크
        if (df['ema12'].iloc[current_idx-1] <= df['ema26'].iloc[current_idx-1] and 
            df['ema12'].iloc[current_idx] > df['ema26'].iloc[current_idx]):
            self.cross_history[symbol]['ema'].append((current_time, 'golden'))
            self.signal_logger.info(f"EMA Cross Check - Golden Cross detected at {current_time}")
        elif (df['ema12'].iloc[current_idx-1] >= df['ema26'].iloc[current_idx-1] and 
            df['ema12'].iloc[current_idx] < df['ema26'].iloc[current_idx]):
            self.cross_history[symbol]['ema'].append((current_time, 'dead'))
            self.signal_logger.info(f"EMA Cross Check - Dead Cross detected at {current_time}")
            
        # MACD 크로스 체크
        if (df['macd'].iloc[current_idx-1] <= df['macd_signal'].iloc[current_idx-1] and 
            df['macd'].iloc[current_idx] > df['macd_signal'].iloc[current_idx]):
            self.cross_history[symbol]['macd'].append((current_time, 'golden'))
            self.signal_logger.info(f"MACD Cross Check - Golden Cross detected at {current_time}")
        elif (df['macd'].iloc[current_idx-1] >= df['macd_signal'].iloc[current_idx-1] and 
            df['macd'].iloc[current_idx] < df['macd_signal'].iloc[current_idx]):
            self.cross_history[symbol]['macd'].append((current_time, 'dead'))
            self.signal_logger.info(f"MACD Cross Check - Dead Cross detected at {current_time}")
        
        self._cleanup_old_crosses(symbol, current_time)


    def _cleanup_old_crosses(self, symbol, current_time):
        """오래된 크로스 데이터 제거"""
        candle_interval = pd.Timedelta(minutes=1 if TIMEFRAME == '1m' else 5)
        cutoff_time = current_time - (candle_interval * 10)  # 전후 5캔들을 고려하여 10캔들 범위 유지
        
        self.cross_history[symbol]['ema'] = [
            (time, type_) for time, type_ in self.cross_history[symbol]['ema']
            if time > cutoff_time
        ]
        self.cross_history[symbol]['macd'] = [
            (time, type_) for time, type_ in self.cross_history[symbol]['macd']
            if time > cutoff_time
        ]
        
    def find_matching_cross(self, symbol, cross_time, cross_type, base_indicator):
        """특정 크로스 시점 기준으로 전후 5캔들 내의 매칭되는 크로스 찾기"""
        candle_interval = pd.Timedelta(minutes=1 if TIMEFRAME == '1m' else 5)
        start_time = cross_time - (candle_interval * 5)
        end_time = cross_time + (candle_interval * 5)
        
        # base_indicator가 'ema'면 'macd'를 찾고, 반대도 마찬가지
        target_indicator = 'macd' if base_indicator == 'ema' else 'ema'
        
        for time, type_ in self.cross_history[symbol][target_indicator]:
            if start_time <= time <= end_time and type_ == cross_type:
                return time
                
        return None

    def check_cross_validity(self, symbol, position_type):
        """크로스 유효성 확인"""
        cross_type = 'golden' if position_type == 'long' else 'dead'
        valid_cross = False
        
        # EMA 크로스를 기준으로 확인
        for ema_time, ema_type in self.cross_history[symbol]['ema']:
            if ema_type == cross_type:
                matching_time = self.find_matching_cross(
                    symbol, ema_time, cross_type, 'ema'
                )
                if matching_time:
                    self.signal_logger.info(
                        f"Valid crosses found for {position_type}:\n"
                        f"EMA {cross_type} cross at {ema_time}\n"
                        f"MACD {cross_type} cross at {matching_time}"
                    )
                    valid_cross = True
                    break
        
        # EMA에서 못찾았다면 MACD 크로스를 기준으로 다시 확인
        if not valid_cross:
            for macd_time, macd_type in self.cross_history[symbol]['macd']:
                if macd_type == cross_type:
                    matching_time = self.find_matching_cross(
                        symbol, macd_time, cross_type, 'macd'
                    )
                    if matching_time:
                        self.signal_logger.info(
                            f"Valid crosses found for {position_type}:\n"
                            f"MACD {cross_type} cross at {macd_time}\n"
                            f"EMA {cross_type} cross at {matching_time}"
                        )
                        valid_cross = True
                        break
        
        return valid_cross

    def check_entry_conditions(self, df, symbol):
        """진입 조건 확인"""
        # 1. 크로스 데이터 먼저 저장
        self.store_cross_data(df, symbol)
        
        current_idx = len(df) - 1
        current_price = df['close'].iloc[-1]
        
        # 2. SMA200 기준 추세 확인
        above_sma200 = current_price > df['sma200'].iloc[-1]
        self.signal_logger.info(
            f"\n=== Signal Analysis for {symbol} ===\n"
            f"1. SMA 200 Position: {'Above - Long only' if above_sma200 else 'Below - Short only'}"
        )
        
        # 3. MA angles JD 색상 확인
        mangles_color = df['mangles_jd_color'].iloc[-1]
        mangles_valid = (above_sma200 and mangles_color == 'green') or (not above_sma200 and mangles_color == 'red')
        self.signal_logger.info(
            f"2. MA angles JD Color: {mangles_color.upper()} - "
            f"{'True' if mangles_valid else 'False'}"
        )
        
        if not mangles_valid:
            return None, None
        
        position_type = 'long' if above_sma200 else 'short'
        if self.check_cross_validity(symbol, position_type):
            self.signal_logger.info(f"5. All conditions matched! {symbol} {position_type.upper()} position entry signal")
            return position_type, self.cross_history[symbol]
        
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
            # 1. 잔액 확인
            if not self.check_balance(symbol):
                return False

            # 2. 거래 가능 상태 확인
            if not self.check_symbol_tradable(symbol):
                return False

            # 포지션 크기 계산
            total_position_size = MARGIN_AMOUNT * LEVERAGE
            position_size = total_position_size / entry_price

            # 3. 최소 주문 수량 확인
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            if position_size < min_amount:
                self.trading_logger.error(
                    f"===== Order Amount Too Small =====\n"
                    f"Symbol: {symbol}\n"
                    f"Minimum required: {min_amount}\n"
                    f"Calculated amount: {position_size}"
                )
                return False

            # 4. API 요청 실행 및 에러 처리
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='buy' if position_type == 'long' else 'sell',
                    amount=position_size,
                    price=entry_price
                )
            except Exception as e:
                self.trading_logger.error(
                    f"===== API Error =====\n"
                    f"Symbol: {symbol}\n"
                    f"Error: {str(e)}"
                )
                return False

            # 이후 손절/익절 주문 실행
            try:
                stop_loss_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop_loss',
                    side='sell' if position_type == 'long' else 'buy',
                    amount=position_size,
                    price=stop_loss
                )

                take_profit_order = self.exchange.create_order(
                    symbol=symbol,
                    type='take_profit',
                    side='sell' if position_type == 'long' else 'buy',
                    amount=position_size,
                    price=take_profit
                )
            except Exception as e:
                self.trading_logger.error(
                    f"===== Stop Loss/Take Profit Order Error =====\n"
                    f"Symbol: {symbol}\n"
                    f"Error: {str(e)}"
                )
                # 기존 주문 취소 시도
                try:
                    self.exchange.cancel_order(order['id'], symbol)
                except:
                    pass
                return False

            # 성공적인 주문 실행 로깅
            self.execution_logger.info(
                f"===== Trade Successfully Executed =====\n"
                f"Symbol: {symbol}\n"
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
            self.trading_logger.error(f"Unexpected error in execute_trade: {str(e)}")
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