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
MARGIN_AMOUNT = 20    # 실제 사용할 증거금 (USDT)
MAX_DAILY_LOSS = 10   # 일일 최대 손실 제한 (USDT)
SLOPE_PERIOD = 10     # Slope 계산을 위한 기간
THRESHOLD = 4         # MA angles JD threshold

# 거래 심볼 목록
TRADING_SYMBOLS = [ #'BTC/USDT',
                 'TIA/USDT', 'DOGS/USDT', 'BAN/USDT', 'BOME/USDT', 'ORCA/USDT', 'AMB/USDT',
                 'BOND/USDT', 'NEAR/USDT', 'HIPPO/USDT', 'BAKE/USDT', 'FXS/USDT', '1000PEPE/USDT',
                'ACX/USDT', 'LINK/USDT', 'POL/USDT', 'MOODENG/USDT', 'ATOM/USDT', 'PHA/USDT',
                'ORDI/USDT', 'DOGE/USDT', 'XLM/USDT', 'GALA/USDT', 'TNSR/USDT', 
                'DOT/USDT', 'ZRO/USDT', 'BNB/USDT', 'THETA/USDT', 'ARPA/USDT', 'EOS/USDT',
                'XRP/USDT', 'ADA/USDT', 'WLD/USDT', 'RENDER/USDT', 'PENGU/USDT', 'AIXBT/USDT', 'ATA/USDT',
                'NEAR/USDT', 'SUI/USDT', 'AVAX/USDT', 'MOVE/USDT', 'GOAT/USDT', 'HIVE/USDT', 'COW/USDT',
                'ZEN/USDT', 'ONDOUSDT', 'USUAL/USDT', 'BRETT/USDT', '1000PEPE/USDT']

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
        
        # 시그널 체크 기록 초기화
        self.last_signal_check = {symbol: None for symbol in TRADING_SYMBOLS}
        
        # 포지션 정보 저장
        self.positions = {
            symbol: {
                'entry_order': None,
                'sl_order': None,
                'tp_order': None,
                'trailing_sl_order': None,    # 추가: 트레일링 스탑 주문 ID
                'position_type': None,
                'trailing_stop_applied': False,
                'entry_price': None,
                'last_trailing_price': None   # 추가: 마지막 트레일링 스탑 가격
            }
            for symbol in TRADING_SYMBOLS
        }
        
        # 손절된 거래 기록 저장용 딕셔너리 추가
        self.sl_history = {
            symbol: {
                'long': None,  # 마지막 롱 손절 시간
                'short': None  # 마지막 숏 손절 시간
            }
            for symbol in TRADING_SYMBOLS
        }
        
        # 손익 관련 속성 추가
        self.daily_losses = 0  # 순수 손실 USDT
        self.daily_profits = 0  # 순수 이익 USDT
        self.last_pnl_reset = datetime.now().date()
    
    # 레버리지 설정
    for symbol in TRADING_SYMBOLS:
        try:
            self.exchange.set_leverage(LEVERAGE, symbol)
            self.trading_logger.info(f"Leverage set for {symbol}: {LEVERAGE}x")
        except Exception as e:
            self.trading_logger.error(f"Error setting leverage for {symbol}: {e}")

    def setup_logging(self, new_date=None):
        """로깅 설정"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 로거 초기화 (기존 핸들러 제거)
        for logger_name in ['trading', 'signal', 'execution', 'profit']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # 날짜 설정
        today = new_date if new_date else datetime.now().strftime('%y%m%d')
        
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
        self.profit_logger = setup_logger('profit', 'profits.log')
        
        # 현재 로그 날짜 저장
        self.current_log_date = datetime.now().date()

    def check_and_update_loggers(self):
        """날짜 변경 확인 및 로거 업데이트"""
        current_date = datetime.now().date()
        if current_date != self.current_log_date:
            # 새로운 날짜로 로거 재설정
            self.setup_logging(current_date.strftime('%y%m%d'))
            self.trading_logger.info(f"New day started - Logs reset for {current_date}")
        
    def check_daily_pnl(self):
        """일일 손익 체크 및 리셋"""
        try:
            current_date = datetime.now().date()
            
            # 날짜가 바뀌었다면 손익 카운트 리셋
            if current_date != self.last_pnl_reset:
                # 리셋 전 마지막 집계 기록
                self.profit_logger.info(
                    f"=== Daily Summary {self.last_pnl_reset} ===\n"
                    f"Total Profits: +{self.daily_profits} USDT\n"
                    f"Total Losses: -{self.daily_losses} USDT"
                )
                
                self.daily_losses = 0
                self.daily_profits = 0
                self.last_pnl_reset = current_date
            
            # 일일 최대 손실 초과 체크
            if self.daily_losses >= MAX_DAILY_LOSS:
                self.profit_logger.error(
                    f"!!! MAX DAILY LOSS EXCEEDED !!!\n"
                    f"Current Losses: -{self.daily_losses} USDT\n"
                    f"Max Allowed: -{MAX_DAILY_LOSS} USDT\n"
                    f"Bot shutdown initiated"
                )
                return False
            return True
                
        except Exception as e:
            self.profit_logger.error(f"Error in check_daily_pnl: {e}")
            return False

    def update_daily_pnl(self, symbol, pnl_amount):
        """손익 발생 시 업데이트"""
        try:
            if pnl_amount > 0:  # 이익
                self.daily_profits += pnl_amount
                self.profit_logger.info(
                    f"{symbol}: +{pnl_amount:.2f} USDT"
                )
            else:  # 손실
                loss_amount = abs(pnl_amount)
                self.daily_losses += loss_amount
                self.profit_logger.info(
                    f"{symbol}: -{loss_amount:.2f} USDT"
                )
        except Exception as e:
            self.profit_logger.error(f"Error updating PnL: {e}")
    
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
                self.execution_logger.error(  # trading_logger -> execution_logger
                    f"===== Insufficient Balance =====\n"
                    f"Symbol: {symbol}\n"
                    f"Required: {MARGIN_AMOUNT} USDT\n"
                    f"Available: {free_usdt} USDT"
                )
                return False
            return True
        except Exception as e:
            self.execution_logger.error(f"Balance check error: {str(e)}")  # 변경
            return False
        
    def check_symbol_tradable(self, symbol):
        try:
            market = self.exchange.market(symbol)
            if not market['active']:
                self.execution_logger.error(  # trading_logger -> execution_logger
                    f"===== Symbol Not Tradable =====\n"
                    f"Symbol: {symbol}\n"
                    f"Status: Trading suspended or delisted"
                )
                return False
            return True
        except Exception as e:
            self.execution_logger.error(f"Symbol status check error: {str(e)}")  # 변경
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

    def calculate_ema_cross_angle(self, df, current_idx):
        """EMA 크로스의 각도 계산"""
        try:
            ema12_diff = df['ema12'].iloc[current_idx] - df['ema12'].iloc[current_idx-1]
            ema26_diff = df['ema26'].iloc[current_idx] - df['ema26'].iloc[current_idx-1]
            slope_diff = abs(ema12_diff - ema26_diff)
            price = df['close'].iloc[current_idx]
            relative_slope = (slope_diff / price) * 100
            
            # 더 자세한 로깅 추가
            self.signal_logger.info(
                f"\nDetailed EMA Cross Analysis:\n"
                f"Current EMA12: {df['ema12'].iloc[current_idx]}, Previous: {df['ema12'].iloc[current_idx-1]}\n"
                f"Current EMA26: {df['ema26'].iloc[current_idx]}, Previous: {df['ema26'].iloc[current_idx-1]}\n"
                f"EMA12 change: {ema12_diff}\n"
                f"EMA26 change: {ema26_diff}\n"
                f"Absolute slope difference: {slope_diff}\n"
                f"Current price: {price}\n"
                f"Relative slope: {relative_slope}%"
            )
            
            return relative_slope
            
        except Exception as e:
            self.signal_logger.error(f"Error calculating EMA cross angle: {e}")
            return 0

    def calculate_macd_cross_angle(self, df, current_idx):
        """MACD 크로스의 각도 계산"""
        try:
            macd_diff = df['macd'].iloc[current_idx] - df['macd'].iloc[current_idx-1]
            signal_diff = df['macd_signal'].iloc[current_idx] - df['macd_signal'].iloc[current_idx-1]
            slope_diff = abs(macd_diff - signal_diff)
            price = df['close'].iloc[current_idx]
            relative_slope = (slope_diff / price) * 100
            
            self.signal_logger.info(
                f"\nDetailed MACD Cross Analysis:\n"
                f"Current MACD: {df['macd'].iloc[current_idx]}, Previous: {df['macd'].iloc[current_idx-1]}\n"
                f"Current Signal: {df['macd_signal'].iloc[current_idx]}, Previous: {df['macd_signal'].iloc[current_idx-1]}\n"
                f"MACD change: {macd_diff}\n"
                f"Signal change: {signal_diff}\n"
                f"Absolute slope difference: {slope_diff}\n"
                f"Current price: {price}\n"
                f"Relative slope: {relative_slope}%"
            )
            
            return relative_slope
            
        except Exception as e:
            self.signal_logger.error(f"Error calculating MACD cross angle: {e}")
            return 0

    def store_cross_data(self, df, symbol):
        try:
            current_idx = len(df) - 1
            current_time = df.index[current_idx]
            formatted_time = current_time.floor('5min')
            
            above_sma200 = df['close'].iloc[current_idx] > df['sma200'].iloc[current_idx]
            ma_color = df['mangles_jd_color'].iloc[current_idx]
            
            candle_start = formatted_time
            candle_end = formatted_time + pd.Timedelta(minutes=5)
            candle_mask = (df.index >= candle_start) & (df.index < candle_end)
            candle_data = df[candle_mask]
            
            period_high = candle_data['high'].max()
            period_low = candle_data['low'].min()

            self.signal_logger.info(f"\n=== Cross Check for {symbol} ===")
            
            MIN_SLOPE = 0.04
            THRESHOLD = 0.00005
            
            # EMA 크로스 체크
            if ((df['ema12'].iloc[current_idx-1] < df['ema26'].iloc[current_idx-1] and 
                    df['ema12'].iloc[current_idx] > df['ema26'].iloc[current_idx]) or
                (abs(df['ema12'].iloc[current_idx] - df['ema12'].iloc[current_idx-1]) > THRESHOLD and
                    df['ema12'].iloc[current_idx] > df['ema26'].iloc[current_idx] and 
                    df['ema12'].iloc[current_idx-1] < df['ema26'].iloc[current_idx-1])):
                
                cross_slope = self.calculate_ema_cross_angle(df, current_idx)
                
                if above_sma200 and ma_color == 'green' and cross_slope >= MIN_SLOPE:
                    self.cross_history[symbol]['ema'] = [(
                        current_time,
                        'golden',
                        period_high,
                        period_low
                    )]
                    self.signal_logger.info(
                        f"NEW EMA Golden Cross at {current_time}\n"
                        f"Cross Slope: {cross_slope}%\n"
                        f"Candle High: {period_high}\n"
                        f"Candle Low: {period_low}\n"
                        f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                        f"MA Color: {ma_color}"
                    )
                    
                    # 이전 25분간의 MACD 크로스 확인
                    macd_cross_time, macd_high, macd_low = self.check_historical_crosses(df, current_time, 'golden', 'ema')
                    if macd_cross_time:
                        self.cross_history[symbol]['macd'] = [(
                            macd_cross_time,
                            'golden',
                            macd_high,
                            macd_low
                        )]
                        return  # 유효한 크로스 쌍 발견 시 종료
                        
                else:
                    self.signal_logger.info(
                        f"EMA Golden Cross ignored - conditions not met\n"
                        f"Above SMA200: {above_sma200}\n"
                        f"MA Color: {ma_color}\n"
                        f"Cross Slope: {cross_slope}%"
                    )
                        
            elif ((df['ema12'].iloc[current_idx-1] > df['ema26'].iloc[current_idx-1] and 
                    df['ema12'].iloc[current_idx] < df['ema26'].iloc[current_idx]) or
                    (abs(df['ema12'].iloc[current_idx] - df['ema12'].iloc[current_idx-1]) > THRESHOLD and
                    df['ema12'].iloc[current_idx] < df['ema26'].iloc[current_idx] and 
                    df['ema12'].iloc[current_idx-1] > df['ema26'].iloc[current_idx-1])):
                
                cross_slope = self.calculate_ema_cross_angle(df, current_idx)
                
                if not above_sma200 and ma_color == 'red' and cross_slope >= MIN_SLOPE:
                    self.cross_history[symbol]['ema'] = [(
                        current_time,
                        'dead',
                        period_high,
                        period_low
                    )]
                    self.signal_logger.info(
                        f"NEW EMA Dead Cross at {current_time}\n"
                        f"Cross Slope: {cross_slope}%\n"
                        f"Candle High: {period_high}\n"
                        f"Candle Low: {period_low}\n"
                        f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                        f"MA Color: {ma_color}"
                    )
                    
                    # 이전 25분간의 MACD 크로스 확인
                    macd_cross_time, macd_high, macd_low = self.check_historical_crosses(df, current_time, 'dead', 'ema')
                    if macd_cross_time:
                        self.cross_history[symbol]['macd'] = [(
                            macd_cross_time,
                            'dead',
                            macd_high,
                            macd_low
                        )]
                        return  # 유효한 크로스 쌍 발견 시 종료
                        
                else:
                    self.signal_logger.info(
                        f"EMA Dead Cross ignored - conditions not met\n"
                        f"Below SMA200: {not above_sma200}\n"
                        f"MA Color: {ma_color}\n"
                        f"Cross Slope: {cross_slope}%"
                    )
            else:
                self.signal_logger.info("No new EMA cross")
                        
            # MACD 크로스 체크
            if ((df['macd'].iloc[current_idx-1] < df['macd_signal'].iloc[current_idx-1] and 
                    df['macd'].iloc[current_idx] > df['macd_signal'].iloc[current_idx]) or
                (abs(df['macd'].iloc[current_idx] - df['macd'].iloc[current_idx-1]) > THRESHOLD and
                    df['macd'].iloc[current_idx] > df['macd_signal'].iloc[current_idx] and 
                    df['macd'].iloc[current_idx-1] < df['macd_signal'].iloc[current_idx-1])):
                
                cross_slope = self.calculate_macd_cross_angle(df, current_idx)
                
                if above_sma200 and ma_color == 'green' and cross_slope >= MIN_SLOPE:
                    self.cross_history[symbol]['macd'] = [(
                        current_time,
                        'golden',
                        period_high,
                        period_low
                    )]
                    self.signal_logger.info(
                        f"NEW MACD Golden Cross at {current_time}\n"
                        f"Cross Slope: {cross_slope}%\n"
                        f"Candle High: {period_high}\n"
                        f"Candle Low: {period_low}\n"
                        f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                        f"MA Color: {ma_color}"
                    )
                    
                    # 이전 25분간의 EMA 크로스 확인
                    ema_cross_time, ema_high, ema_low = self.check_historical_crosses(df, current_time, 'golden', 'macd')
                    if ema_cross_time:
                        self.cross_history[symbol]['ema'] = [(
                            ema_cross_time,
                            'golden',
                            ema_high,
                            ema_low
                        )]
                        return  # 유효한 크로스 쌍 발견 시 종료
                        
                else:
                    self.signal_logger.info(
                        f"MACD Golden Cross ignored - conditions not met\n"
                        f"Above SMA200: {above_sma200}\n"
                        f"MA Color: {ma_color}\n"
                        f"Cross Slope: {cross_slope}%"
                    )
                        
            elif ((df['macd'].iloc[current_idx-1] > df['macd_signal'].iloc[current_idx-1] and 
                    df['macd'].iloc[current_idx] < df['macd_signal'].iloc[current_idx]) or
                    (abs(df['macd'].iloc[current_idx] - df['macd'].iloc[current_idx-1]) > THRESHOLD and
                    df['macd'].iloc[current_idx] < df['macd_signal'].iloc[current_idx] and 
                    df['macd'].iloc[current_idx-1] > df['macd_signal'].iloc[current_idx-1])):
                
                cross_slope = self.calculate_macd_cross_angle(df, current_idx)
                
                if not above_sma200 and ma_color == 'red' and cross_slope >= MIN_SLOPE:
                    self.cross_history[symbol]['macd'] = [(
                        current_time,
                        'dead',
                        period_high,
                        period_low
                    )]
                    self.signal_logger.info(
                        f"NEW MACD Dead Cross at {current_time}\n"
                        f"Cross Slope: {cross_slope}%\n"
                        f"Candle High: {period_high}\n"
                        f"Candle Low: {period_low}\n"
                        f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                        f"MA Color: {ma_color}"
                    )
                    
                    # 이전 25분간의 EMA 크로스 확인
                    ema_cross_time, ema_high, ema_low = self.check_historical_crosses(df, current_time, 'dead', 'macd')
                    if ema_cross_time:
                        self.cross_history[symbol]['ema'] = [(
                            ema_cross_time,
                            'dead',
                            ema_high,
                            ema_low
                        )]
                        return  # 유효한 크로스 쌍 발견 시 종료
                        
                else:
                    self.signal_logger.info(
                        f"MACD Dead Cross ignored - conditions not met\n"
                        f"Below SMA200: {not above_sma200}\n"
                        f"MA Color: {ma_color}\n"
                        f"Cross Slope: {cross_slope}%"
                    )
            else:
                self.signal_logger.info("No new MACD cross")

            self._cleanup_old_crosses(symbol, formatted_time)
            
            # 저장된 크로스 정보 로깅
            if self.cross_history[symbol]['ema']:
                cross_info = self.cross_history[symbol]['ema'][0]
                self.signal_logger.info(
                    f"Current EMA cross: Time={cross_info[0]}, "
                    f"Type={cross_info[1]}, "
                    f"High={cross_info[2]}, "
                    f"Low={cross_info[3]}"
                )
            if self.cross_history[symbol]['macd']:
                cross_info = self.cross_history[symbol]['macd'][0]
                self.signal_logger.info(
                    f"Current MACD cross: Time={cross_info[0]}, "
                    f"Type={cross_info[1]}, "
                    f"High={cross_info[2]}, "
                    f"Low={cross_info[3]}"
                )

        except Exception as e:
            self.signal_logger.error(f"Error in store_cross_data: {e}")
       
    def check_historical_crosses(self, df, current_time, cross_type, primary_indicator):
        try:
            window_start = current_time - pd.Timedelta(minutes=25)
            window_data = df[(df.index >= window_start) & (df.index <= current_time)]
            target_indicator = 'macd' if primary_indicator == 'ema' else 'ema'
            MIN_SLOPE = 0.042

            self.signal_logger.info(
                f"\nChecking Historical {target_indicator.upper()} Crosses\n"
                f"Time Range: {window_start} to {current_time}"
            )

            for i in range(0, len(window_data)-5, 5):
                check_idx = i + 5
                candle_start = window_data.index[check_idx] - pd.Timedelta(minutes=5)
                candle_data = window_data[(window_data.index >= candle_start) & (window_data.index <= window_data.index[check_idx])]
                period_high = candle_data['high'].max()
                period_low = candle_data['low'].min()

                if target_indicator == 'macd':
                    if ((window_data['macd'].iloc[check_idx-1] < window_data['macd_signal'].iloc[check_idx-1] and 
                            window_data['macd'].iloc[check_idx] > window_data['macd_signal'].iloc[check_idx] and
                            cross_type == 'golden') or
                        (window_data['macd'].iloc[check_idx-1] > window_data['macd_signal'].iloc[check_idx-1] and 
                            window_data['macd'].iloc[check_idx] < window_data['macd_signal'].iloc[check_idx] and
                            cross_type == 'dead')):
                        
                        # 해당 시점의 SMA200, ma_color 확인
                        above_sma200 = window_data['close'].iloc[check_idx] > window_data['sma200'].iloc[check_idx]
                        ma_color = window_data['mangles_jd_color'].iloc[check_idx]
                        
                        cross_slope = self.calculate_macd_cross_angle(window_data, check_idx)
                        if cross_slope >= MIN_SLOPE and ((cross_type == 'golden' and above_sma200 and ma_color == 'green') or
                                                        (cross_type == 'dead' and not above_sma200 and ma_color == 'red')):
                            self.signal_logger.info(
                                f"Found Historical MACD {cross_type.title()} Cross\n"
                                f"Time: {window_data.index[check_idx]}\n"
                                f"Cross Slope: {cross_slope}%\n"
                                f"Candle High: {period_high}\n"
                                f"Candle Low: {period_low}\n"
                                f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                                f"MA Color: {ma_color}"
                            )
                            return window_data.index[check_idx], period_high, period_low

                else:
                    if ((window_data['ema12'].iloc[check_idx-1] < window_data['ema26'].iloc[check_idx-1] and 
                            window_data['ema12'].iloc[check_idx] > window_data['ema26'].iloc[check_idx] and
                            cross_type == 'golden') or
                        (window_data['ema12'].iloc[check_idx-1] > window_data['ema26'].iloc[check_idx-1] and 
                            window_data['ema12'].iloc[check_idx] < window_data['ema26'].iloc[check_idx] and
                            cross_type == 'dead')):
                        
                        # 해당 시점의 SMA200, ma_color 확인
                        above_sma200 = window_data['close'].iloc[check_idx] > window_data['sma200'].iloc[check_idx]
                        ma_color = window_data['mangles_jd_color'].iloc[check_idx]
                        
                        cross_slope = self.calculate_ema_cross_angle(window_data, check_idx)
                        if cross_slope >= MIN_SLOPE and ((cross_type == 'golden' and above_sma200 and ma_color == 'green') or
                                                        (cross_type == 'dead' and not above_sma200 and ma_color == 'red')):
                            self.signal_logger.info(
                                f"Found Historical EMA {cross_type.title()} Cross\n"
                                f"Time: {window_data.index[check_idx]}\n"
                                f"Cross Slope: {cross_slope}%\n"
                                f"Candle High: {period_high}\n"
                                f"Candle Low: {period_low}\n"
                                f"SMA200: {'Above' if above_sma200 else 'Below'}\n"
                                f"MA Color: {ma_color}"
                            )
                            return window_data.index[check_idx], period_high, period_low

            return None, None, None

        except Exception as e:
            self.signal_logger.error(f"Error checking historical crosses: {e}")
            return None, None, None

    def _cleanup_old_crosses(self, symbol, formatted_time):
        """오래된 크로스 데이터 제거"""
        try:
            # 이미 formatted_time으로 들어오므로 변환 필요 없음
            try:
                current_time = pd.to_datetime(formatted_time)
            except Exception as e:
                self.trading_logger.error(f"Error converting formatted time: {e}")
                return
            
            # 기준 시간 설정 (5분봉 = 25분, 1분봉 = 5분)
            minutes = 25 if TIMEFRAME == '5m' else 5
            cutoff_time = current_time - pd.Timedelta(minutes=minutes)

            # 크로스 데이터 정리 함수
            def check_and_cleanup_crosses(cross_list, cross_type):
                if not cross_list:
                    return cross_list
                    
                try:
                    # 저장된 크로스 시간도 이미 formatted time
                    cross_time = pd.to_datetime(cross_list[0][0])
                    if cross_time <= cutoff_time:
                        self.signal_logger.info(
                            f"Removed expired {cross_type} cross from {cross_time.strftime('%Y-%m-%d %H:%M')}"
                        )
                        return []
                    return cross_list
                except Exception as e:
                    self.trading_logger.error(f"Error processing {cross_type} cross: {e}")
                    return cross_list

            # EMA와 MACD 크로스 정리
            self.cross_history[symbol]['ema'] = check_and_cleanup_crosses(
                self.cross_history[symbol]['ema'], 'EMA'
            )
            self.cross_history[symbol]['macd'] = check_and_cleanup_crosses(
                self.cross_history[symbol]['macd'], 'MACD'
            )

        except Exception as e:
            raise Exception(f"Error in cleanup_old_crosses: {e}")
        
    def find_matching_cross(self, symbol, cross_time, cross_type, base_indicator):
        """특정 크로스 시점 기준으로 전후 5캔들 내의 매칭되는 크로스 찾기"""
        try:
            # 입력받은 cross_time은 이미 formatted_time
            cross_time = pd.to_datetime(cross_time)
            candle_interval = pd.Timedelta(minutes=1 if TIMEFRAME == '1m' else 5)
            # 전후 5캔들 범위 계산
            start_time = cross_time - (candle_interval * 5)
            end_time = cross_time + (candle_interval * 5)
            
            self.signal_logger.info(f"\nMatching Cross Check:")
            self.signal_logger.info(f"Base Time: {cross_time.strftime('%Y-%m-%d %H:%M')}")
            self.signal_logger.info(f"Valid Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            
            # base_indicator가 'ema'면 'macd'를 찾고, 반대도 마찬가지
            target_indicator = 'macd' if base_indicator == 'ema' else 'ema'
            
            for time, type_ in self.cross_history[symbol][target_indicator]:
                # 저장된 크로스 시간도 formatted_time
                check_time = pd.to_datetime(time)
                self.signal_logger.info(f"Checking {target_indicator.upper()} cross at {check_time.strftime('%Y-%m-%d %H:%M')} ({type_})")
                
                if start_time <= check_time <= end_time and type_ == cross_type:
                    self.signal_logger.info(f"Found matching {target_indicator.upper()} cross!")
                    return time
                else:
                    self.signal_logger.info(
                        f"Not matched. Time in range: {start_time <= check_time <= end_time}, "
                        f"Type match: {type_ == cross_type}"
                    )
                    
            self.signal_logger.info(f"No matching {target_indicator.upper()} cross found")
            return None
                
        except Exception as e:
            self.signal_logger.error(f"Error in find_matching_cross: {e}")
            return None

    def check_cross_validity(self, symbol, position_type):
        """크로스 유효성 확인"""
        cross_type = 'golden' if position_type == 'long' else 'dead'
        
        self.signal_logger.info(f"\nCross Validity Check for {symbol}:")
        self.signal_logger.info(f"Position Type: {position_type}")
        self.signal_logger.info(f"Expected Cross Type: {cross_type}")
        self.signal_logger.info(f"Stored EMA crosses: {self.cross_history[symbol]['ema']}")
        self.signal_logger.info(f"Stored MACD crosses: {self.cross_history[symbol]['macd']}")

        try:
            # 저장된 크로스 확인
            if not self.cross_history[symbol]['ema'] or not self.cross_history[symbol]['macd']:
                return False

            # 크로스 타입 체크
            ema_cross = self.cross_history[symbol]['ema'][0]
            macd_cross = self.cross_history[symbol]['macd'][0]
            
            if ema_cross[1] != cross_type or macd_cross[1] != cross_type:
                return False
                
            # 시간 차이 체크 (25분 이내)
            # 저장된 시간이 이미 formatted_time이므로 그대로 사용
            ema_time = pd.to_datetime(ema_cross[0])
            macd_time = pd.to_datetime(macd_cross[0])
            time_diff = abs((ema_time - macd_time).total_seconds() / 60)
            
            if time_diff <= 25:  # 5분봉 기준 5캔들
                entry_message = (
                    f"\n{'='*20} ENTRY SIGNAL {'='*20}\n"
                    f"Symbol: {symbol}\n"
                    f"Position: {position_type.upper()}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Cross Pairs: [{ema_cross[0].strftime('%Y-%m-%d %H:%M'), macd_cross[0].strftime('%Y-%m-%d %H:%M')}]\n"
                    f"{'='*50}"
                )
                self.signal_logger.info(entry_message)
                self.execution_logger.info(entry_message)
                return True
                
            return False
            
        except Exception as e:
            self.signal_logger.error(f"Error in cross validity check: {e}")
            return False

    def check_entry_conditions(self, df, symbol):
        """진입 조건 확인"""
        # 1. 지표 계산 먼저
        df = self.calculate_indicators(df, symbol)
        if df is None:
            return None, None
        
        # 2. 현재 가격 확인 및 포지션 타입 결정
        current_price = df['close'].iloc[-1]
        above_sma200 = current_price > df['sma200'].iloc[-1]
        position_type = 'long' if above_sma200 else 'short'
        
        # 3. 최근 손절 이력 확인
        last_sl_time = self.sl_history[symbol][position_type]
        if last_sl_time:
            time_since_sl = (datetime.now() - last_sl_time).total_seconds() / 60
            if time_since_sl < 30:  # 30분 이내
                self.trading_logger.info(
                    f"Skipping {position_type} trade for {symbol}: "
                    f"Recent stop loss ({time_since_sl:.1f} mins ago)"
                )
                return None, None
        
        # 4. 크로스 데이터 저장
        self.store_cross_data(df, symbol)
        
        self.signal_logger.info(
            f"\n=== Position Analysis for {symbol} ===\n"
            f"1. SMA 200 Position: {'Above - Long only' if above_sma200 else 'Below - Short only'}"
        )
        
        # MA angles JD 색상 확인
        mangles_color = df['mangles_jd_color'].iloc[-1]
        mangles_valid = (above_sma200 and mangles_color == 'green') or (not above_sma200 and mangles_color == 'red')
        self.signal_logger.info(
            f"2. MA angles JD Color: {mangles_color.upper()} - {mangles_valid}"
        )
        
        if not mangles_valid:
            return None, None
        
        if self.check_cross_validity(symbol, position_type):
            entry_message = (
                f"\n{'='*20} ENTRY SIGNAL {'='*20}\n"
                f"Symbol: {symbol}\n"
                f"Position: {position_type.upper()}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: {current_price}\n"
                f"{'='*50}"
            )
            self.signal_logger.info(entry_message)
            self.execution_logger.info(entry_message)
            return position_type, self.cross_history[symbol]
        
        return None, None

    def determine_stop_loss(self, df, crosses, position_type, entry_price):
        """
        먼저 발생한 크로스의 시점을 기준으로 이전 5분간의 high/low로 stop loss 설정
        """
        try:
            if not crosses['ema'] or not crosses['macd']:
                self.execution_logger.error("Missing cross data")
                return None
                    
            # 두 크로스의 시간 비교
            ema_time = pd.to_datetime(crosses['ema'][0][0])
            macd_time = pd.to_datetime(crosses['macd'][0][0])
            
            # 먼저 발생한 크로스 찾기
            first_cross = crosses['ema'][0] if ema_time <= macd_time else crosses['macd'][0]
            first_cross_time = ema_time if ema_time <= macd_time else macd_time
                
            # 먼저 발생한 크로스 시점 기준 이전 5분 데이터 추출
            sl_start = first_cross_time - pd.Timedelta(minutes=5)
            sl_mask = (df.index >= sl_start) & (df.index <= first_cross_time)
            sl_data = df[sl_mask]
            
            # 해당 구간의 high/low 계산
            period_high = sl_data['high'].max()
            period_low = sl_data['low'].min()
                
            # position type에 따른 stop loss 설정
            stop_loss = period_low if position_type == 'long' else period_high
                
            # SL 거리 검증
            sl_distance = abs(stop_loss - entry_price)
            min_sl_distance = entry_price * 0.003  # 최소 0.3% 차이
                
            if sl_distance < min_sl_distance:
                self.execution_logger.warning(
                    f"Stop loss too close to entry price:\n"
                    f"Entry: {entry_price}\n"
                    f"Stop Loss: {stop_loss}\n"
                    f"Distance: {sl_distance} (minimum required: {min_sl_distance})"
                )
                return None
                
            self.execution_logger.info(
                f"Stop Loss calculation for {position_type}:\n"
                f"First cross time: {first_cross_time}\n"
                f"Data range: {sl_start} to {first_cross_time}\n"
                f"Stop loss price: {stop_loss}\n"
                f"Distance from entry: {sl_distance} ({(sl_distance/entry_price)*100:.3f}%)"
            )
                    
            return stop_loss
                        
        except Exception as e:
            self.execution_logger.error(f"Stop loss calculation error: {e}")
            return None

    def calculate_take_profit(self, entry_price, stop_loss, position_type):
        """목표가 계산"""
        try:
            stop_loss_distance = abs(entry_price - stop_loss)
            take_profit = entry_price + (stop_loss_distance * 2.25) if position_type == 'long' else entry_price - (stop_loss_distance * 2.25)
            
            self.execution_logger.info(
                f"Take Profit calculation:\n"
                f"Entry: {entry_price}\n"
                f"Stop Loss: {stop_loss}\n"
                f"SL Distance: {stop_loss_distance}\n"
                f"Take Profit: {take_profit}"
            )
            return take_profit
        
        except Exception as e:
            self.execution_logger.error(f"Take profit calculation error: {e}")
            return None

    def check_existing_position(self, symbol):
        """현재 보유 중인 포지션과 주문 상태 확인"""
        try:
            # 심볼별 현재 포지션 확인
            positions = self.exchange.fetch_positions([symbol])
            position_info = self.positions.get(symbol)
            
            # 포지션이 있는 경우
            if positions and float(positions[0]['contracts']) != 0:
                position_size = float(positions[0]['contracts'])
                position_side = 'long' if position_size > 0 else 'short'
                
                # 디버그 레벨로 변경하여 일반 로그에는 표시되지 않도록
                self.trading_logger.debug(
                    f"Active position exists for {symbol}:\n"
                    f"Side: {position_side}\n"
                    f"Size: {abs(position_size)}\n"
                    f"Entry Price: {positions[0]['entryPrice']}"
                )
                
                # 크로스 히스토리 초기화 추가
                if position_info and not position_info['entry_order']:
                    self.cross_history[symbol] = {
                        'ema': [],
                        'macd': []
                    }
                
                return True

            # 포지션이 없고, 이전에 포지션 정보가 있었다면 청산된 것
            if position_info and position_info['entry_order']:
                try:
                    # 진입 주문 상태 확인
                    entry_order = self.exchange.fetch_order(position_info['entry_order'], symbol)
                    
                    # 진입 주문이 체결(filled)되었는데 현재 포지션이 없다면 청산된 상태
                    if entry_order['status'] == 'filled':
                        self.trading_logger.debug(f"Position was closed for {symbol}, cleaning up remaining orders")
                        
                        # 남은 SL/TP 주문 취소
                        if position_info['sl_order']:
                            try:
                                self.exchange.cancel_order(position_info['sl_order'], symbol)
                                self.trading_logger.debug(f"Cancelled SL order {position_info['sl_order']} for {symbol}")
                            except Exception as e:
                                self.trading_logger.error(f"Error cancelling SL order: {e}")
                                
                        if position_info['tp_order']:
                            try:
                                self.exchange.cancel_order(position_info['tp_order'], symbol)
                                self.trading_logger.debug(f"Cancelled TP order {position_info['tp_order']} for {symbol}")
                            except Exception as e:
                                self.trading_logger.error(f"Error cancelling TP order: {e}")
                        
                        # 포지션 정보 초기화
                        self.positions[symbol] = {
                            'entry_order': None,
                            'sl_order': None,
                            'tp_order': None,
                            'position_type': None,
                            'trailing_stop_applied': False,
                            'entry_price': None
                        }
                        
                        # 크로스 히스토리 초기화
                        self.cross_history[symbol] = {
                            'ema': [],
                            'macd': []
                        }
                        
                except Exception as e:
                    self.trading_logger.error(f"Error checking entry order status: {e}")

            return False
                
        except Exception as e:
            self.trading_logger.error(f"Error checking position for {symbol}: {e}")
            return True  # 에러 시 안전하게 True 반환

    def execute_trade(self, symbol, position_type, entry_price, stop_loss, take_profit):
        """주문 실행"""
        try:
            # 손절가 없거나 너무 가까우면 진입 불가
            if stop_loss is None:
                self.execution_logger.error(f"Failed to enter position for {symbol}: Stop loss calculation failed")
                return False
                
            # 잔액 확인
            if not self.check_balance(symbol):
                self.execution_logger.error(f"Failed to enter position for {symbol}: Insufficient balance")
                return False
                
            # 기존 포지션 체크
            if self.check_existing_position(symbol):
                self.execution_logger.error(f"Failed to enter position for {symbol}: Position already exists")
                return False

            # 거래 가능 상태 확인
            if not self.check_symbol_tradable(symbol):
                self.execution_logger.error(f"Failed to enter position for {symbol}: Symbol not tradable")
                return False

            # 레버리지를 고려한 실제 포지션 크기 계산
            total_position_size = MARGIN_AMOUNT * LEVERAGE
            position_size = total_position_size / entry_price

            # 최소 주문 수량 확인
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            if position_size < min_amount:
                self.execution_logger.error(
                    f"===== Order Amount Too Small =====\n"
                    f"Symbol: {symbol}\n"
                    f"Minimum required: {min_amount}\n"
                    f"Calculated amount: {position_size}"
                )
                return False

            try:
                # 진입 주문 실행
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='buy' if position_type == 'long' else 'sell',
                    amount=position_size,
                    price=entry_price
                )

                # Stop Loss 주문
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='sell' if position_type == 'long' else 'buy',
                    amount=position_size,
                    price=stop_loss,
                    params={
                        'stopPrice': stop_loss,
                        'type': 'future',
                        'reduceOnly': 'true'
                    }
                )

                # Take Profit 주문
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='take_profit',
                    side='sell' if position_type == 'long' else 'buy',
                    amount=position_size,
                    price=take_profit,
                    params={
                        'stopPrice': take_profit,
                        'type': 'future',
                        'reduceOnly': 'true'
                    }
                )

                # 주문 정보 저장
                self.positions[symbol] = {
                    'entry_order': order['id'],
                    'sl_order': sl_order['id'],
                    'tp_order': tp_order['id'],
                    'position_type': position_type,
                    'trailing_stop_applied': False,
                    'entry_price': entry_price
                }

                # 크로스 히스토리 초기화
                self.cross_history[symbol] = {
                    'ema': [],
                    'macd': []
                }

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
                    f"Order IDs: Entry={order['id']}, SL={sl_order['id']}, TP={tp_order['id']}"
                )

                return True

            except Exception as e:
                if 'Order would immediately trigger' in str(e):
                    self.execution_logger.error(f"Order would trigger immediately for {symbol}: {e}")
                    return False
                    
                self.execution_logger.error(f"Order execution error for {symbol}: {e}")
                # 실패 시 생성된 주문들 취소 시도
                try:
                    if 'order' in locals():
                        self.exchange.cancel_order(order['id'], symbol)
                    if 'sl_order' in locals():
                        self.exchange.cancel_order(sl_order['id'], symbol)
                    if 'tp_order' in locals():
                        self.exchange.cancel_order(tp_order['id'], symbol)
                except Exception as cancel_error:
                    self.execution_logger.error(f"Error canceling orders after failure: {cancel_error}")
                return False

        except Exception as e:
            self.execution_logger.error(f"Unexpected error in execute_trade: {str(e)}")
            return False

    def update_trailing_stop(self, symbol, entry_price, current_price, signal, contract_amount):
        try:
            position_info = self.positions[symbol]
            
            # 이미 트레일링 스탑이 적용된 경우 스킵
            if position_info['trailing_stop_applied']:
                return True
            
            if signal == 'buy':
                profit_percent = ((current_price - entry_price) / entry_price) * 100
                if profit_percent >= 1.8:  # 1.8% 이상 수익 시
                    new_stop_loss = entry_price * 1.015
                    
                    # 새로운 트레일링 스탑 주문 생성 (기존 SL은 유지)
                    try:
                        trailing_sl_order = self.exchange.create_order(
                            symbol,
                            'stop',
                            'sell',
                            contract_amount,
                            new_stop_loss,
                            {'stopPrice': new_stop_loss, 'type': 'future', 'reduceOnly': True}
                        )
                        
                        # trailing_sl_order 정보 추가
                        self.positions[symbol]['trailing_sl_order'] = trailing_sl_order['id']
                        self.positions[symbol]['trailing_stop_applied'] = True
                        
                        self.execution_logger.info(
                            f"Trailing Stop Added (Original SL maintained)\n"
                            f"Symbol: {symbol}\n"
                            f"Time: {datetime.now()}\n"
                            f"Entry: {entry_price}\n"
                            f"Current Price: {current_price}\n"
                            f"Original SL: Maintained\n"
                            f"Trailing SL: {new_stop_loss}\n"
                            f"Profit %: {profit_percent:.2f}%\n"
                            f"Trailing Order ID: {trailing_sl_order['id']}"
                        )
                        return True
                        
                    except Exception as e:
                        self.execution_logger.error(f"Error creating trailing SL order: {e}")
                        return False
                        
            elif signal == 'sell':
                profit_percent = ((entry_price - current_price) / entry_price) * 100
                if profit_percent >= 1.8:  # 1.8% 이상 수익 시
                    new_stop_loss = entry_price * 0.985
                    
                    # 새로운 트레일링 스탑 주문 생성 (기존 SL은 유지)
                    try:
                        trailing_sl_order = self.exchange.create_order(
                            symbol,
                            'stop',
                            'buy',
                            contract_amount,
                            new_stop_loss,
                            {'stopPrice': new_stop_loss, 'type': 'future', 'reduceOnly': True}
                        )
                        
                        # trailing_sl_order 정보 추가
                        self.positions[symbol]['trailing_sl_order'] = trailing_sl_order['id']
                        self.positions[symbol]['trailing_stop_applied'] = True
                        
                        self.execution_logger.info(
                            f"Trailing Stop Added (Original SL maintained)\n"
                            f"Symbol: {symbol}\n"
                            f"Time: {datetime.now()}\n"
                            f"Entry: {entry_price}\n"
                            f"Current Price: {current_price}\n"
                            f"Original SL: Maintained\n"
                            f"Trailing SL: {new_stop_loss}\n"
                            f"Profit %: {profit_percent:.2f}%\n"
                            f"Trailing Order ID: {trailing_sl_order['id']}"
                        )
                        return True
                        
                    except Exception as e:
                        self.execution_logger.error(f"Error creating trailing SL order: {e}")
                        return False
            
            return True
            
        except Exception as e:
            self.execution_logger.error(f"Error updating trailing stop: {e}")
            return False

    def update_sl_history(self, symbol, position_type):
        """손절 발생 시 기록 업데이트"""
        try:
            self.sl_history[symbol][position_type] = datetime.now()
            self.execution_logger.info(
                f"Updated SL history for {symbol}:\n"
                f"Position: {position_type}\n"
                f"Time: {self.sl_history[symbol][position_type]}"
            )
        except Exception as e:
            self.execution_logger.error(f"Error updating SL history: {e}")

    def check_order_status(self, symbol):
        """
        주문 상태를 확인하고 필요한 업데이트를 수행합니다.
        - 포지션 청산 확인 (SL/TP/수동)
        - 남은 주문 취소
        - 손익 계산 및 기록
        - 트레일링 스탑 업데이트/체크
        """
        try:
            position_info = self.positions[symbol]
            if not position_info['entry_order']:
                return

            # 포지션이 있었는데 없어진 경우 확인
            positions = self.exchange.fetch_positions([symbol])
            if position_info['position_type'] and (not positions or float(positions[0]['contracts']) == 0):
                # 청산 타입 확인 (SL/TP/Trailing SL)
                closing_type = None
                # 기존 스탑로스 체크
                if position_info['sl_order']:
                    try:
                        sl_order = self.exchange.fetch_order(position_info['sl_order'], symbol)
                        if sl_order['status'] == 'filled':
                            closing_type = 'Stop Loss'
                    except Exception as e:
                        self.execution_logger.error(f"Error checking stop loss order: {e}")
                
                # 트레일링 스탑 체크
                if position_info['trailing_sl_order'] and not closing_type:
                    try:
                        trailing_sl_order = self.exchange.fetch_order(position_info['trailing_sl_order'], symbol)
                        if trailing_sl_order['status'] == 'filled':
                            closing_type = 'Trailing Stop'
                    except Exception as e:
                        self.execution_logger.error(f"Error checking trailing stop order: {e}")

                # TP 체크
                if position_info['tp_order'] and not closing_type:
                    try:
                        tp_order = self.exchange.fetch_order(position_info['tp_order'], symbol)
                        if tp_order['status'] == 'filled':
                            closing_type = 'Take Profit'
                    except Exception as e:
                        self.execution_logger.error(f"Error checking take profit order: {e}")

                # 손익 계산
                try:
                    if closing_type:  # 청산된 경우
                        entry_price = float(position_info['entry_price'])
                        
                        # 청산 가격 결정
                        if closing_type == 'Stop Loss':
                            close_price = float(sl_order['price'])
                        elif closing_type == 'Trailing Stop':
                            close_price = float(trailing_sl_order['price'])
                        elif closing_type == 'Take Profit':
                            close_price = float(tp_order['price'])
                        else:
                            close_price = float(self.exchange.fetch_ticker(symbol)['last'])
                                
                        # 원래 포지션 크기 계산
                        original_margin = MARGIN_AMOUNT  # 실제 사용된 증거금
                        
                        if position_info['position_type'] == 'long':
                            price_change_pct = (close_price - entry_price) / entry_price
                        else:  # short
                            price_change_pct = (entry_price - close_price) / entry_price
                        
                        # 실제 손익 계산 (레버리지 제외)
                        pnl_usdt = original_margin * price_change_pct
                        
                        # 손익 업데이트 및 로깅
                        self.update_daily_pnl(symbol, pnl_usdt)
                        
                        # 손실 한도 체크
                        if not self.check_daily_pnl():
                            sys.exit("Bot shutdown due to max daily loss exceeded")
                        
                        # 스탑로스나 트레일링 스탑으로 청산된 경우 히스토리 업데이트
                        if closing_type in ['Stop Loss', 'Trailing Stop']:
                            self.update_sl_history(symbol, position_info['position_type'])
                                
                except Exception as e:
                    self.profit_logger.error(f"Error calculating PnL for {symbol}: {e}")

                # 모든 열린 주문 취소
                try:
                    open_orders = self.exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        try:
                            self.exchange.cancel_order(order['id'], symbol)
                            self.trading_logger.info(
                                f"Cancelled remaining order for {symbol}: {order['id']}"
                            )
                        except Exception as cancel_error:
                            self.trading_logger.error(f"Error cancelling order {order['id']}: {cancel_error}")
                except Exception as e:
                    self.execution_logger.error(f"Error fetching open orders: {e}")

                # 포지션 청산 로깅
                self.execution_logger.info(
                    f"Position Closed - {closing_type if closing_type else 'Manual'}\n"
                    f"Symbol: {symbol}\n"
                    f"Time: {datetime.now()}\n"
                    f"Position Type: {position_info['position_type']}\n"
                    f"Entry Price: {position_info['entry_price']}\n"
                    f"Close Price: {close_price if 'close_price' in locals() else 'Unknown'}\n"
                    f"PnL: {pnl_usdt:.2f} USDT\n"
                    f"{'SL History updated' if closing_type in ['Stop Loss', 'Trailing Stop'] else ''}"
                )
                
                # 포지션 정보 초기화
                self.positions[symbol] = {
                    'entry_order': None,
                    'sl_order': None,
                    'tp_order': None,
                    'trailing_sl_order': None,
                    'position_type': None,
                    'trailing_stop_applied': False,
                    'entry_price': None,
                    'last_trailing_price': None
                }
                return

            # 활성 포지션이 있는 경우 트레일링 스탑 업데이트 체크
            if positions and float(positions[0]['contracts']) != 0 and position_info['entry_price']:
                try:
                    current_price = float(self.exchange.fetch_ticker(symbol)['last'])
                    
                    # 트레일링 스탑이 적용되었다면 last_trailing_price 기준으로 수익률 계산
                    reference_price = position_info['last_trailing_price'] if position_info['last_trailing_price'] else position_info['entry_price']
                    
                    self.update_trailing_stop(
                        symbol,
                        reference_price,  # 트레일링 스탑 적용 여부에 따라 기준가격 변경
                        current_price,
                        'buy' if position_info['position_type'] == 'long' else 'sell',
                        float(positions[0]['contracts'])
                    )
                except Exception as e:
                    self.execution_logger.error(f"Error updating trailing stop: {e}")

        except Exception as e:
            self.execution_logger.error(f"Error in check_order_status: {e}")

    def run(self):
        self.trading_logger.info(f"Bot started running\n"
                            f"Leverage: {LEVERAGE}x\n"
                            f"Margin Amount: {MARGIN_AMOUNT} USDT\n"
                            f"Max Daily Loss: {MAX_DAILY_LOSS} USDT\n"
                            f"Trading Symbols: {TRADING_SYMBOLS}")
        
        while True:
            try:
                # 로거 날짜 체크 및 업데이트
                self.check_and_update_loggers()
                
                # 일일 손실 한도 체크
                if not self.check_daily_pnl():
                    sys.exit("Bot shutdown due to max daily loss exceeded")
                
                for symbol in TRADING_SYMBOLS:
                    try:
                        # 주문 상태 확인
                        self.check_order_status(symbol)
                        
                        # 기존 포지션 확인을 먼저 수행
                        if self.check_existing_position(symbol):
                            # 크로스 히스토리 초기화 추가
                            self.cross_history[symbol] = {
                                'ema': [],
                                'macd': []
                            }
                            continue  # 포지션이 있으면 다음 심볼로 넘어감
                        
                        df = self.get_historical_data(symbol)
                        if df is None:
                            continue

                        current_time = df.index[-1].floor('5min')
                        
                        # 같은 시간대에 이미 체크했다면 스킵
                        if (self.last_signal_check[symbol] is not None and 
                            self.last_signal_check[symbol] == current_time):
                            continue

                        position_type, crosses = self.check_entry_conditions(df, symbol)
                        
                        if position_type and crosses:
                            # 현재 5분봉 시작 시점
                            candle_start = current_time
                            
                            # 해당 5분봉의 종가를 진입가로 사용
                            entry_idx = df.index.get_loc(candle_start)
                            entry_price = df['close'].iloc[entry_idx]
                            
                            self.execution_logger.info(
                                f"\nCalculating orders for {symbol}:\n"
                                f"Candle time: {candle_start}\n"
                                f"Entry price (close): {entry_price}"
                            )
                            
                            stop_loss = self.determine_stop_loss(df, crosses, position_type, entry_price)
                            if not stop_loss:
                                continue

                            take_profit = self.calculate_take_profit(entry_price, stop_loss, position_type)
                            if not take_profit:
                                continue

                            success = self.execute_trade(
                                symbol,
                                position_type,
                                entry_price,
                                stop_loss,
                                take_profit
                            )

                        # 체크 시간 업데이트
                        self.last_signal_check[symbol] = current_time
                            
                    except Exception as e:
                        self.trading_logger.error(f"Error processing {symbol}: {e}")
                        continue

                time.sleep(60)
                
            except Exception as e:
                self.trading_logger.error(f"Critical error in main loop: {e}")
                time.sleep(60)
           
if __name__ == "__main__":
    API_KEY = "vf7WGUJSNjVUjsqH8H6BKj0eKpmIecotvP5S1NQlLy041py9LuuFsiK2rksaSomq"
    API_SECRET = "bBOK1BVnOYaVIvKUKoSQsTAt450Ps64WkjQjINDUXkpzvxmQDJUrZT9vSbvkwAKZ"
    
    bot = TradingBot(API_KEY, API_SECRET)
    bot.run()