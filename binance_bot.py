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
    
    def calculate_jurik_ma(self, data, length=10, phase=50, power=1):
        """Jurik Moving Average 구현"""
        try:
            # Phase ratio 계산
            phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5
            
            # Beta, Alpha 계산
            beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
            alpha = pow(beta, power)
            
            # JMA 계산을 위한 초기 시리즈
            e0 = pd.Series(index=data.index, dtype=float)
            e1 = pd.Series(index=data.index, dtype=float)
            e2 = pd.Series(index=data.index, dtype=float)
            jma = pd.Series(index=data.index, dtype=float)
            
            # 첫 값 초기화
            e0.iloc[0] = data.iloc[0]
            e1.iloc[0] = 0
            e2.iloc[0] = 0
            jma.iloc[0] = data.iloc[0]
            
            # JMA 계산
            for i in range(1, len(data)):
                e0.iloc[i] = (1 - alpha) * data.iloc[i] + alpha * e0.iloc[i-1]
                e1.iloc[i] = (data.iloc[i] - e0.iloc[i]) * (1 - beta) + beta * e1.iloc[i-1]
                e2.iloc[i] = (e0.iloc[i] + phase_ratio * e1.iloc[i] - jma.iloc[i-1]) * pow(1 - alpha, 2) + pow(alpha, 2) * e2.iloc[i-1]
                jma.iloc[i] = e2.iloc[i] + jma.iloc[i-1]
            
            return jma
        except Exception as e:
            self.trading_logger.error(f"Error calculating JMA: {e}")
            return None

    def calculate_angle(self, data, df, length=14):  # df 파라미터 추가
        """각도 계산"""
        try:
            # ATR 계산
            atr = ta.volatility.average_true_range(
                df['high'],  # self.df -> df
                df['low'],   # self.df -> df
                df['close'], # self.df -> df
                length
            )
            
            # 각도 계산
            diff = data - data.shift(1)
            angle = (180 / np.pi) * np.arctan(diff / atr)
            
            return angle
        except Exception as e:
            self.trading_logger.error(f"Error calculating angle: {e}")
            return None

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
            # UTC를 KST(UTC+9)로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
            df.set_index('timestamp', inplace=True)
            
            self.trading_logger.info(f"Retrieved data for {symbol}: {df.shape[0]} rows")
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

    def calculate_indicators(self, df, symbol):  
        try:
            # 데이터 확인 로깅
            self.trading_logger.info(f"Calculating indicators for {symbol}")
            self.trading_logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            # 기본 지표 계산
            df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
            df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # JMA 계산
            df['jma'] = self.calculate_jurik_ma(df['close'])
            if df['jma'] is None:
                return None
            
            # Angle 계산
            jma_slope = self.calculate_angle(df['jma'], df)  # df 전달
            if jma_slope is None:
                return None
                
            # 디버그 로깅
            self.signal_logger.info(f"\nJMA Slope Debug for {symbol}:")
            self.signal_logger.info(f"Current JMA value: {df['jma'].iloc[-1]}")
            self.signal_logger.info(f"Previous JMA value: {df['jma'].iloc[-2]}")
            self.signal_logger.info(f"Calculated slope: {jma_slope.iloc[-1]}")
            self.signal_logger.info(f"Threshold comparison: {jma_slope.iloc[-1]} > {THRESHOLD}")
            
            # MA angles JD 색상 결정
            df['mangles_jd_color'] = np.where(jma_slope > THRESHOLD, 'green', 'red')
            
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
        current_idx = len(df) - 1
        current_time = df.index[current_idx]
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M')  # 타임스탬프 포맷 변경

        self.signal_logger.info(f"\n=== Cross Check for {symbol} ===")
        
        # EMA 크로스 체크 - 새로운 크로스 발생 시 기존 데이터 초기화
        if (df['ema12'].iloc[current_idx-1] <= df['ema26'].iloc[current_idx-1] and 
            df['ema12'].iloc[current_idx] > df['ema26'].iloc[current_idx]):
            self.cross_history[symbol]['ema'] = [(formatted_time, 'golden')]  # 리스트 초기화하고 새 데이터만 저장
            self.signal_logger.info(f"NEW EMA Golden Cross at {formatted_time}")
        elif (df['ema12'].iloc[current_idx-1] >= df['ema26'].iloc[current_idx-1] and 
            df['ema12'].iloc[current_idx] < df['ema26'].iloc[current_idx]):
            self.cross_history[symbol]['ema'] = [(formatted_time, 'dead')]  # 리스트 초기화하고 새 데이터만 저장
            self.signal_logger.info(f"NEW EMA Dead Cross at {formatted_time}")
        else:
            self.signal_logger.info("No new EMA cross")
            
        # MACD 크로스 체크 - 새로운 크로스 발생 시 기존 데이터 초기화
        if (df['macd'].iloc[current_idx-1] <= df['macd_signal'].iloc[current_idx-1] and 
            df['macd'].iloc[current_idx] > df['macd_signal'].iloc[current_idx]):
            self.cross_history[symbol]['macd'] = [(formatted_time, 'golden')]  # 리스트 초기화하고 새 데이터만 저장
            self.signal_logger.info(f"NEW MACD Golden Cross at {formatted_time}")
        elif (df['macd'].iloc[current_idx-1] >= df['macd_signal'].iloc[current_idx-1] and 
            df['macd'].iloc[current_idx] < df['macd_signal'].iloc[current_idx]):
            self.cross_history[symbol]['macd'] = [(formatted_time, 'dead')]  # 리스트 초기화하고 새 데이터만 저장
            self.signal_logger.info(f"NEW MACD Dead Cross at {formatted_time}")
        else:
            self.signal_logger.info("No new MACD cross")

        self._cleanup_old_crosses(symbol, current_time)
        self.signal_logger.info(f"Current Cross History for {symbol}:")
        self.signal_logger.info(f"EMA crosses: {self.cross_history[symbol]['ema']}")
        self.signal_logger.info(f"MACD crosses: {self.cross_history[symbol]['macd']}")


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
        valid_pairs = []

        # 각 EMA 크로스에 대해 ±5캔들 범위 내의 MACD 크로스 찾기
        for ema_time, ema_type in self.cross_history[symbol]['ema']:
            if ema_type == cross_type:
                matching_macd = self.find_matching_cross(symbol, ema_time, cross_type, 'ema')
                if matching_macd:
                    valid_pairs.append((ema_time, matching_macd))

        # 각 MACD 크로스에 대해 ±5캔들 범위 내의 EMA 크로스 찾기
        for macd_time, macd_type in self.cross_history[symbol]['macd']:
            if macd_type == cross_type:
                matching_ema = self.find_matching_cross(symbol, macd_time, cross_type, 'macd')
                if matching_ema:
                    pair = tuple(sorted([macd_time, matching_ema]))
                    if pair not in valid_pairs:
                        valid_pairs.append(pair)

        if valid_pairs:
            self.signal_logger.info(f"Found valid cross pairs for {symbol} {position_type}:")
            for time1, time2 in valid_pairs:
                self.signal_logger.info(f"Cross pair: {time1} and {time2}")
            return True

        return False

    def check_entry_conditions(self, df, symbol):
        """진입 조건 확인"""
        # 1-1. 지표 계산 먼저
        df = self.calculate_indicators(df, symbol)
        if df is None:
            return None, None
        
        # 1-2. 크로스 데이터 저장
        self.store_cross_data(df, symbol)
        
        # 2. 현재 가격 확인 및 SMA 200 추세 확인
        current_price = df['close'].iloc[-1]  # 여기서 먼저 정의
        above_sma200 = current_price > df['sma200'].iloc[-1]
        
        self.signal_logger.info(
            f"\n=== Position Analysis for {symbol} ===\n"
            f"1. SMA 200 Position: {'Above - Long only' if above_sma200 else 'Below - Short only'}"
        )
        
        # 3. MA angles JD 색상 확인
        mangles_color = df['mangles_jd_color'].iloc[-1]
        mangles_valid = (above_sma200 and mangles_color == 'green') or (not above_sma200 and mangles_color == 'red')
        self.signal_logger.info(
            f"2. MA angles JD Color: {mangles_color.upper()} - {mangles_valid}"
        )
        
        if not mangles_valid:
            return None, None
        
        position_type = 'long' if above_sma200 else 'short'
        if self.check_cross_validity(symbol, position_type):
            self.signal_logger.info(f"=== ENTRY SIGNAL: {symbol} {position_type.upper()} ===")
            return position_type, self.cross_history[symbol]
        
        return None, None

    def determine_stop_loss(self, df, crosses, position_type):
        """손절가 계산"""
        try:
            # 크로스 데이터에서 시간 추출
            ema_time = crosses['ema'][0][0] if crosses['ema'] else None
            macd_time = crosses['macd'][0][0] if crosses['macd'] else None
            
            if not ema_time or not macd_time:
                self.execution_logger.error(f"Missing cross data for stop loss calculation")
                return None
                
            # 시간 기준으로 인덱스 찾기
            ema_idx = df.index.get_loc(ema_time)
            macd_idx = df.index.get_loc(macd_time)
            
            # 먼저 발생한 크로스의 캔들 선택
            first_cross_idx = min(ema_idx, macd_idx)
            
            if position_type == 'long':
                return df['low'].iloc[first_cross_idx]
            else:
                return df['high'].iloc[first_cross_idx]
                
        except Exception as e:
            self.execution_logger.error(f"Error calculating stop loss: {e}")
            return None

    def calculate_take_profit(self, entry_price, stop_loss, position_type):
        """목표가 계산"""
        stop_loss_distance = abs(entry_price - stop_loss)
        if position_type == 'long':
            return entry_price + (stop_loss_distance * 2)
        else:
            return entry_price - (stop_loss_distance * 2)

    def check_existing_position(self, symbol):
        """현재 보유 중인 포지션 확인"""
        try:
            position = self.exchange.fetch_position(symbol)
            
            # 포지션 수량이 있는지 확인
            if position and abs(float(position['contracts'] or 0)) > 0:
                self.trading_logger.info(
                    f"Position already exists for {symbol}\n"
                    f"Size: {position['contracts']}\n"
                    f"Side: {position['side']}\n"
                    f"Entry Price: {position['entryPrice']}"
                )
                return True
                
            return False
            
        except Exception as e:
            self.trading_logger.error(f"Error checking position for {symbol}: {e}")
            return True  # 에러 발생 시 안전하게 True 반환

    def execute_trade(self, symbol, position_type, entry_price, stop_loss, take_profit):
        """주문 실행"""
        try:
            # 손절가 없으면 진입 불가
            if stop_loss is None:
                self.execution_logger.error(f"Failed to enter position for {symbol}: Stop loss calculation failed")
                return False
                
            # 1. 잔액 확인
            if not self.check_balance(symbol):
                self.execution_logger.error(f"Failed to enter position for {symbol}: Insufficient balance")
                return False
                
            # 기존 포지션 체크
            if self.check_existing_position(symbol):
                self.execution_logger.error(f"Failed to enter position for {symbol}: Position already exists")
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
        self.trading_logger.info(f"Bot started running\n"
                               f"Leverage: {LEVERAGE}x\n"
                               f"Margin Amount: {MARGIN_AMOUNT} USDT\n"
                               f"Trading Symbols: {TRADING_SYMBOLS}")
        
        while True:
            try:
                # 각 심볼에 대해 트레이딩 로직 실행
                for symbol in TRADING_SYMBOLS:
                    # 데이터 수집 및 지표 계산
                    df = self.get_historical_data(symbol)
                    if df is None:
                        continue
                        
                    # 진입 조건 확인
                    position_type, crosses = self.check_entry_conditions(df, symbol)
                    
                    if position_type and crosses:
                        entry_price = df['close'].iloc[-1]
                        stop_loss = self.determine_stop_loss(df, crosses, position_type)
                        
                        if stop_loss:
                            take_profit = self.calculate_take_profit(
                                entry_price,
                                stop_loss,
                                position_type
                            )
                            
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
                
                # 1분마다 체크
                time.sleep(60)
                
            except Exception as e:
                self.trading_logger.error(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    API_KEY = "vf7WGUJSNjVUjsqH8H6BKj0eKpmIecotvP5S1NQlLy041py9LuuFsiK2rksaSomq"
    API_SECRET = "bBOK1BVnOYaVIvKUKoSQsTAt450Ps64WkjQjINDUXkpzvxmQDJUrZT9vSbvkwAKZ"
    
    bot = TradingBot(API_KEY, API_SECRET)
    bot.run()