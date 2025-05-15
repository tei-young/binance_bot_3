import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import ta

# ===== 설정값 =====
CONFIG = {
    # 백테스팅할 심볼들
    'SYMBOLS': [
        'TIA/USDT', 'DOGS/USDT', 'BAN/USDT', 'BOME/USDT', 'ORCA/USDT', 'AMB/USDT',
        'BOND/USDT', 'NEAR/USDT', 'HIPPO/USDT', 'BAKE/USDT', 'FXS/USDT', '1000PEPE/USDT',
        'ACX/USDT', 'LINK/USDT', 'POL/USDT', 'MOODENG/USDT', 'ATOM/USDT', 'PHA/USDT',
        'ORDI/USDT', 'DOGE/USDT', 'XLM/USDT', 'GALA/USDT', 'TNSR/USDT', 'GRASS/USDT',
        'DOT/USDT', 'ZRO/USDT', 'BNB/USDT', 'THETA/USDT', 'ARPA/USDT', 'EOS/USDT',
        'XRP/USDT', 'ADA/USDT', 'WLD/USDT', 'RENDER/USDT', 'PENGU/USDT', 'AIXBT/USDT', 'ATA/USDT',
        'NEAR/USDT', 'SUI/USDT', 'AVAX/USDT', 'MOVE/USDT', 'GOAT/USDT', 'HIVE/USDT', 'COW/USDT',
        'ZEN/USDT', 'ONDOUSDT', 'USUAL/USDT', 'BRETT/USDT', '1000PEPE/USDT', 'VANA/USDT', 'MELANIA/USDT'
        ],
    
    # 백테스팅 기간
    'START_DATE': '2025-04-10',
    'END_DATE': '2025-05-14',
    'DAYS': 30,
    
    # 거래 설정 (현재 봇과 동일)
    'INITIAL_BALANCE': 1000,
    'LEVERAGE': 10,
    'MARGIN_AMOUNT': 20,
    'MAX_DAILY_LOSS': 10,
    'TIMEFRAME': '5m',
    
    # 파라미터 설정
    'EMA_FAST': 12,
    'EMA_SLOW': 26,
    'THRESHOLD': 4,  # MA angles JD threshold
    'TP_RATIO': 2.25,
    'CROSS_TIME_LIMIT': 25,  # 크로스 간 최대 허용 시간(분)
    'MIN_SLOPE': 0.04,  # MACD 크로스 최소 기울기
    'TRAILING_STOP_TRIGGER': 1.0,  # 트레일링 스탑 발동 수익률(%)
    'TRAILING_STOP_DISTANCE': 0.3,  # 트레일링 스탑 거리(%)
    
    # 경로 설정
    'DATA_DIR': './data/historical',
    'REPORT_DIR': './reports',
    
    # 옵션
    'FORCE_DOWNLOAD': False,
    'GENERATE_PLOTS': True,
    'SAVE_TRADES': True,
}

# =====================================

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.exchange = ccxt.binance()
        self.data_dir = config['DATA_DIR']
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def collect_data(self, symbol, timeframe, start_date, end_date):
        """데이터 수집"""
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath) and not self.config['FORCE_DOWNLOAD']:
            print(f"[{symbol}] 기존 데이터 로드: {filepath}")
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            # KST 시간대 변환
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
            return df
        
        print(f"[{symbol}] 데이터 다운로드 중... ({start_date} ~ {end_date})")
        
        all_data = []
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[{symbol}] 에러: {e}")
                time.sleep(5)
                continue
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        # KST 시간대 설정
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
        
        df.to_csv(filepath)
        print(f"[{symbol}] 저장 완료: {len(df)}개 캔들")
        
        return df

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.initial_balance = config['INITIAL_BALANCE']
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        
        # 크로스 히스토리 (실제 봇과 동일)
        self.cross_history = {}
        
        # 포지션 정보
        self.positions = {}
        
        # 손절 히스토리
        self.sl_history = {}
        
        # 일일 손익
        self.daily_losses = 0
        self.daily_profits = 0
        self.last_pnl_reset = datetime.now().date()
    
    def calculate_jurik_ma(self, data, length=10, phase=50, power=1):
        """Jurik Moving Average 구현 (실제 봇에서 복사)"""
        try:
            phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5
            beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
            alpha = pow(beta, power)
            
            e0 = pd.Series(index=data.index, dtype=float)
            e1 = pd.Series(index=data.index, dtype=float)
            e2 = pd.Series(index=data.index, dtype=float)
            jma = pd.Series(index=data.index, dtype=float)
            
            e0.iloc[0] = data.iloc[0]
            e1.iloc[0] = 0
            e2.iloc[0] = 0
            jma.iloc[0] = data.iloc[0]
            
            for i in range(1, len(data)):
                e0.iloc[i] = (1 - alpha) * data.iloc[i] + alpha * e0.iloc[i-1]
                e1.iloc[i] = (data.iloc[i] - e0.iloc[i]) * (1 - beta) + beta * e1.iloc[i-1]
                e2.iloc[i] = (e0.iloc[i] + phase_ratio * e1.iloc[i] - jma.iloc[i-1]) * pow(1 - alpha, 2) + pow(alpha, 2) * e2.iloc[i-1]
                jma.iloc[i] = e2.iloc[i] + jma.iloc[i-1]
            
            return jma
        except Exception as e:
            print(f"JMA 계산 오류: {e}")
            return None
    
    def calculate_angle(self, data, df, length=14):
        """각도 계산"""
        try:
            atr = ta.volatility.average_true_range(
                df['high'],
                df['low'],
                df['close'],
                length
            )
            
            diff = data - data.shift(1)
            angle = (180 / np.pi) * np.arctan(diff / atr)
            
            return angle
        except Exception as e:
            print(f"각도 계산 오류: {e}")
            return None
    
    def calculate_indicators(self, df):
        """지표 계산 (실제 봇과 동일)"""
        # 기본 지표
        df['sma200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema12'] = ta.trend.ema_indicator(df['close'], window=self.config['EMA_FAST'])
        df['ema26'] = ta.trend.ema_indicator(df['close'], window=self.config['EMA_SLOW'])
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # JMA
        df['jma'] = self.calculate_jurik_ma(df['close'])
        if df['jma'] is None:
            df['jma'] = df['close'].rolling(10).mean()  # 대체
        
        # MA angles
        jma_slope = self.calculate_angle(df['jma'], df)
        if jma_slope is None:
            jma_slope = df['jma'].diff()
        
        df['mangles_jd_color'] = jma_slope.apply(
            lambda x: 'green' if x > self.config['THRESHOLD'] else 'red'
        )
        
        return df
    
    def is_strong_cross(self, ema_distances, ema12_changes):
        """강한 크로스 판별 (실제 봇에서 복사)"""
        try:
            distances = [float(d) for d in ema_distances]
            changes = [float(c) for c in ema12_changes]
            
            max_distance = max(distances)
            max_change = max(abs(c) for c in changes)
            
            # 거리가 줄어드는 패턴
            distance_decrease = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
            
            # 일관된 방향성
            consistent_direction = all(c > 0 for c in changes) or all(c < 0 for c in changes)
            
            # 평균 변화율
            avg_change = sum(changes) / len(changes)
            
            return (
                max_distance > 0.2 and
                max_change > 0.2 and
                distance_decrease and
                consistent_direction and
                abs(avg_change) > 0.1
            )
            
        except Exception as e:
            return False
    
    def check_entry_conditions(self, df, idx, symbol):
        """진입 조건 체크 (실제 봇의 로직)"""
        if idx < 200:
            return None, None
        
        current_time = df.index[idx]
        current_price = df['close'].iloc[idx]
        above_sma200 = current_price > df['sma200'].iloc[idx]
        ma_color = df['mangles_jd_color'].iloc[idx]
        
        position_type = 'long' if above_sma200 else 'short'
        
        # 최근 손절 이력 확인
        if symbol in self.sl_history:
            last_sl_time = self.sl_history[symbol].get(position_type)
            if last_sl_time:
                time_since_sl = (current_time - last_sl_time).total_seconds() / 60
                if time_since_sl < 30:
                    return None, None
        
        # 크로스 체크
        formatted_time = current_time.floor('5min')
        
        # EMA 크로스 체크
        ema_cross_found = False
        pre_cross = {
            'ema_distances': [],
            'ema12_changes': []
        }
        
        # t-2, t-1, t에서 크로스 확인
        for check_idx in [idx-2, idx-1, idx]:
            if check_idx < 0:
                continue
                
            if position_type == 'long':
                if (df['ema12'].iloc[check_idx-1] < df['ema26'].iloc[check_idx-1] and
                    df['ema12'].iloc[check_idx] > df['ema26'].iloc[check_idx]):
                    
                    # 크로스 데이터 수집
                    for i in range(check_idx - 3, check_idx + 1):
                        if i > 0:
                            distance = abs(df['ema12'].iloc[i] - df['ema26'].iloc[i])
                            normalized_distance = (distance / df['close'].iloc[i]) * 100
                            pre_cross['ema_distances'].append(f"{normalized_distance:.3f}")
                            
                            ema12_change = ((df['ema12'].iloc[i] - df['ema12'].iloc[i-1]) / 
                                          df['ema12'].iloc[i-1]) * 100
                            pre_cross['ema12_changes'].append(f"{ema12_change:.3f}")
                    
                    if self.is_strong_cross(pre_cross['ema_distances'], pre_cross['ema12_changes']):
                        if above_sma200 and ma_color == 'green':
                            ema_cross_found = True
                            if symbol not in self.cross_history:
                                self.cross_history[symbol] = {'ema': [], 'macd': []}
                            
                            self.cross_history[symbol]['ema'] = [(
                                df.index[check_idx],
                                'golden',
                                df['high'].iloc[check_idx],
                                df['low'].iloc[check_idx]
                            )]
                            break
            else:  # short
                if (df['ema12'].iloc[check_idx-1] > df['ema26'].iloc[check_idx-1] and
                    df['ema12'].iloc[check_idx] < df['ema26'].iloc[check_idx]):
                    
                    # 크로스 데이터 수집
                    for i in range(check_idx - 3, check_idx + 1):
                        if i > 0:
                            distance = abs(df['ema12'].iloc[i] - df['ema26'].iloc[i])
                            normalized_distance = (distance / df['close'].iloc[i]) * 100
                            pre_cross['ema_distances'].append(f"{normalized_distance:.3f}")
                            
                            ema12_change = ((df['ema12'].iloc[i] - df['ema12'].iloc[i-1]) / 
                                          df['ema12'].iloc[i-1]) * 100
                            pre_cross['ema12_changes'].append(f"{ema12_change:.3f}")
                    
                    if self.is_strong_cross(pre_cross['ema_distances'], pre_cross['ema12_changes']):
                        if not above_sma200 and ma_color == 'red':
                            ema_cross_found = True
                            if symbol not in self.cross_history:
                                self.cross_history[symbol] = {'ema': [], 'macd': []}
                            
                            self.cross_history[symbol]['ema'] = [(
                                df.index[check_idx],
                                'dead',
                                df['high'].iloc[check_idx],
                                df['low'].iloc[check_idx]
                            )]
                            break
        
        # MACD 크로스 체크
        macd_cross_found = False
        
        for check_idx in [idx-2, idx-1, idx]:
            if check_idx < 0:
                continue
                
            if position_type == 'long':
                if (df['macd'].iloc[check_idx-1] < df['macd_signal'].iloc[check_idx-1] and
                    df['macd'].iloc[check_idx] > df['macd_signal'].iloc[check_idx]):
                    
                    # 기울기 계산
                    macd_diff = df['macd'].iloc[check_idx] - df['macd'].iloc[check_idx-1]
                    signal_diff = df['macd_signal'].iloc[check_idx] - df['macd_signal'].iloc[check_idx-1]
                    slope_diff = abs(macd_diff - signal_diff)
                    relative_slope = (slope_diff / df['close'].iloc[check_idx]) * 100
                    
                    if relative_slope >= self.config['MIN_SLOPE']:
                        if above_sma200 and ma_color == 'green':
                            macd_cross_found = True
                            if symbol not in self.cross_history:
                                self.cross_history[symbol] = {'ema': [], 'macd': []}
                            
                            self.cross_history[symbol]['macd'] = [(
                                df.index[check_idx],
                                'golden',
                                df['high'].iloc[check_idx],
                                df['low'].iloc[check_idx]
                            )]
                            break
            else:  # short
                if (df['macd'].iloc[check_idx-1] > df['macd_signal'].iloc[check_idx-1] and
                    df['macd'].iloc[check_idx] < df['macd_signal'].iloc[check_idx]):
                    
                    # 기울기 계산
                    macd_diff = df['macd'].iloc[check_idx] - df['macd'].iloc[check_idx-1]
                    signal_diff = df['macd_signal'].iloc[check_idx] - df['macd_signal'].iloc[check_idx-1]
                    slope_diff = abs(macd_diff - signal_diff)
                    relative_slope = (slope_diff / df['close'].iloc[check_idx]) * 100
                    
                    if relative_slope >= self.config['MIN_SLOPE']:
                        if not above_sma200 and ma_color == 'red':
                            macd_cross_found = True
                            if symbol not in self.cross_history:
                                self.cross_history[symbol] = {'ema': [], 'macd': []}
                            
                            self.cross_history[symbol]['macd'] = [(
                                df.index[check_idx],
                                'dead',
                                df['high'].iloc[check_idx],
                                df['low'].iloc[check_idx]
                            )]
                            break
        
        # 크로스 유효성 확인
        if ema_cross_found and macd_cross_found:
            ema_time = pd.to_datetime(self.cross_history[symbol]['ema'][0][0])
            macd_time = pd.to_datetime(self.cross_history[symbol]['macd'][0][0])
            time_diff = abs((ema_time - macd_time).total_seconds() / 60)
            
            if time_diff <= self.config['CROSS_TIME_LIMIT']:
                return position_type, self.cross_history[symbol]
        
        # 오래된 크로스 정리
        self._cleanup_old_crosses(symbol, formatted_time)
        
        return None, None
    
    def _cleanup_old_crosses(self, symbol, current_time):
        """오래된 크로스 정리"""
        if symbol not in self.cross_history:
            return
            
        cutoff_time = current_time - pd.Timedelta(minutes=self.config['CROSS_TIME_LIMIT'])
        
        # EMA 크로스 정리
        if self.cross_history[symbol]['ema']:
            cross_time = pd.to_datetime(self.cross_history[symbol]['ema'][0][0])
            if cross_time <= cutoff_time:
                self.cross_history[symbol]['ema'] = []
        
        # MACD 크로스 정리
        if self.cross_history[symbol]['macd']:
            cross_time = pd.to_datetime(self.cross_history[symbol]['macd'][0][0])
            if cross_time <= cutoff_time:
                self.cross_history[symbol]['macd'] = []
    
    def determine_stop_loss(self, df, crosses, position_type, entry_price):
        """손절가 결정 (실제 봇과 동일)"""
        if not crosses['ema'] or not crosses['macd']:
            return None
        
        # 먼저 발생한 크로스 찾기
        ema_time = pd.to_datetime(crosses['ema'][0][0])
        macd_time = pd.to_datetime(crosses['macd'][0][0])
        
        first_cross_time = ema_time if ema_time <= macd_time else macd_time
        
        # 첫 크로스 이전 5분 데이터
        sl_start = first_cross_time - pd.Timedelta(minutes=5)
        sl_mask = (df.index >= sl_start) & (df.index <= first_cross_time)
        sl_data = df[sl_mask]
        
        if sl_data.empty:
            return None
        
        period_high = sl_data['high'].max()
        period_low = sl_data['low'].min()
        
        stop_loss = period_low if position_type == 'long' else period_high
        
        # 최소 거리 검증
        sl_distance = abs(stop_loss - entry_price)
        min_sl_distance = entry_price * 0.003
        
        if sl_distance < min_sl_distance:
            return None
        
        return stop_loss
    
    def execute_trade(self, symbol, position_type, entry_time, entry_price, df, idx):
        """거래 실행 시뮬레이션"""
        # 크로스 정보 가져오기
        crosses = self.cross_history.get(symbol, {'ema': [], 'macd': []})
        
        # SL/TP 계산
        stop_loss = self.determine_stop_loss(df[:idx+1], crosses, position_type, entry_price)
        if stop_loss is None:
            # 기본 손절 사용
            if position_type == 'long':
                stop_loss = entry_price * 0.98
            else:
                stop_loss = entry_price * 1.02
        
        # TP 계산
        sl_distance = abs(entry_price - stop_loss)
        if position_type == 'long':
            take_profit = entry_price + (sl_distance * self.config['TP_RATIO'])
        else:
            take_profit = entry_price - (sl_distance * self.config['TP_RATIO'])
        
        # 포지션 정보 저장
        if symbol not in self.positions:
            self.positions[symbol] = {}
        
        self.positions[symbol] = {
            'position_type': position_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_applied': False
        }
        
        # 결과 시뮬레이션
        exit_time = None
        exit_price = None
        exit_reason = None
        highest_profit = 0
        
        for i in range(idx + 1, len(df)):
            current_price = df['close'].iloc[i]
            
            # 트레일링 스탑 체크
            if position_type == 'long':
                profit_percent = ((current_price - entry_price) / entry_price) * 100
                
                if profit_percent >= self.config['TRAILING_STOP_TRIGGER']:
                    if not self.positions[symbol]['trailing_stop_applied']:
                        # 트레일링 스탑 적용
                        new_stop_loss = entry_price * (1 + self.config['TRAILING_STOP_DISTANCE'] / 100)
                        self.positions[symbol]['stop_loss'] = max(stop_loss, new_stop_loss)
                        self.positions[symbol]['trailing_stop_applied'] = True
                
                if current_price <= self.positions[symbol]['stop_loss']:
                    exit_price = self.positions[symbol]['stop_loss']
                    exit_reason = 'stop_loss'
                    if self.positions[symbol]['trailing_stop_applied']:
                        exit_reason = 'trailing_stop'
                    break
                elif current_price >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
                    
            else:  # short
                profit_percent = ((entry_price - current_price) / entry_price) * 100
                
                if profit_percent >= self.config['TRAILING_STOP_TRIGGER']:
                    if not self.positions[symbol]['trailing_stop_applied']:
                        # 트레일링 스탑 적용
                        new_stop_loss = entry_price * (1 - self.config['TRAILING_STOP_DISTANCE'] / 100)
                        self.positions[symbol]['stop_loss'] = min(stop_loss, new_stop_loss)
                        self.positions[symbol]['trailing_stop_applied'] = True
                
                if current_price >= self.positions[symbol]['stop_loss']:
                    exit_price = self.positions[symbol]['stop_loss']
                    exit_reason = 'stop_loss'
                    if self.positions[symbol]['trailing_stop_applied']:
                        exit_reason = 'trailing_stop'
                    break
                elif current_price <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
            
            exit_time = df.index[i]
        
        # 백테스트 기간 끝까지 안 닫히면
        if exit_price is None:
            exit_price = df['close'].iloc[-1]
            exit_time = df.index[-1]
            exit_reason = 'end_of_data'
        
        # 손익 계산
        if position_type == 'long':
            price_change = (exit_price - entry_price) / entry_price
        else:
            price_change = (entry_price - exit_price) / entry_price
        
        pnl = self.config['MARGIN_AMOUNT'] * price_change
        self.balance += pnl
        
        # 손절 히스토리 업데이트
        if exit_reason == 'stop_loss':
            if symbol not in self.sl_history:
                self.sl_history[symbol] = {}
            self.sl_history[symbol][position_type] = exit_time
        
        # 거래 기록
        trade = {
            'symbol': symbol,
            'position_type': position_type,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_applied': self.positions[symbol]['trailing_stop_applied'],
            'pnl': pnl,
            'balance': self.balance,
            'profit_percent': price_change * 100
        }
        
        self.trades.append(trade)
        
        # 포지션 초기화
        self.positions[symbol] = {}
        self.cross_history[symbol] = {'ema': [], 'macd': []}
        
        return trade
    
    def run_backtest(self, symbol, data):
        """심볼별 백테스트 실행"""
        print(f"\n[{symbol}] 백테스팅 시작...")
        
        # 초기화
        self.cross_history[symbol] = {'ema': [], 'macd': []}
        self.positions[symbol] = {}
        self.sl_history[symbol] = {}
        
        # 지표 계산
        df = self.calculate_indicators(data.copy())
        
        # 시뮬레이션
        for idx in range(200, len(df)):
            current_time = df.index[idx]
            
            # 일일 손익 체크
            current_date = current_time.date()
            if current_date != self.last_pnl_reset:
                self.daily_losses = 0
                self.daily_profits = 0
                self.last_pnl_reset = current_date
            
            # 최대 손실 체크
            if self.daily_losses >= self.config['MAX_DAILY_LOSS']:
                print(f"[{symbol}] 일일 최대 손실 도달: {self.daily_losses}")
                break
            
            # 포지션이 없을 때만 진입 체크
            if not self.positions.get(symbol):
                position_type, crosses = self.check_entry_conditions(df, idx, symbol)
                
                if position_type and crosses:
                    trade = self.execute_trade(
                        symbol,
                        position_type,
                        current_time,
                        df['close'].iloc[idx],
                        df,
                        idx
                    )
                    
                    # 일일 손익 업데이트
                    if trade['pnl'] > 0:
                        self.daily_profits += trade['pnl']
                    else:
                        self.daily_losses += abs(trade['pnl'])
                    
                    print(f"거래 실행: {position_type} @ {trade['entry_price']:.4f}, "
                          f"결과: {trade['exit_reason']} @ {trade['exit_price']:.4f}, "
                          f"손익: ${trade['pnl']:.2f}")
            
            # 자산 곡선 기록
            self.equity_curve.append({
                'time': current_time,
                'balance': self.balance
            })
        
        symbol_trades = [t for t in self.trades if t['symbol'] == symbol]
        print(f"[{symbol}] 백테스팅 완료: {len(symbol_trades)}개 거래")

class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.report_dir = config['REPORT_DIR']
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
    
    def generate_report(self, trades, equity_curve):
        """종합 리포트 생성"""
        print("\n===== 백테스팅 결과 =====")
        
        # 기본 통계
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # 최종 수익
        final_balance = equity_curve[-1]['balance'] if equity_curve else self.config['INITIAL_BALANCE']
        total_return = ((final_balance - self.config['INITIAL_BALANCE']) / self.config['INITIAL_BALANCE']) * 100
        
        # 최대 손실폭 계산
        equity_values = [e['balance'] for e in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # 샤프 비율 계산
        if len(trades) > 1:
            returns = [(t['balance'] - trades[i-1]['balance']) / trades[i-1]['balance'] 
                      for i, t in enumerate(trades) if i > 0]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 결과 출력
        print(f"총 거래 수: {total_trades}")
        print(f"승리 거래: {winning_trades} ({win_rate:.2f}%)")
        print(f"패배 거래: {losing_trades}")
        print(f"평균 수익: ${avg_win:.2f}")
        print(f"평균 손실: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"최대 손실폭: {max_drawdown:.2f}%")
        print(f"샤프 비율: {sharpe_ratio:.2f}")
        print(f"초기 자본: ${self.config['INITIAL_BALANCE']}")
        print(f"최종 자본: ${final_balance:.2f}")
        print(f"총 수익률: {total_return:.2f}%")
        
        # 추가 통계
        if trades:
            # 심볼별 성과
            print("\n=== 심볼별 성과 ===")
            symbol_performance = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0,
                        'win_rate': 0
                    }
                
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['total_pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    symbol_performance[symbol]['wins'] += 1
            
            for symbol, perf in symbol_performance.items():
                perf['win_rate'] = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
                print(f"{symbol}: {perf['trades']}거래, 승률 {perf['win_rate']:.1f}%, 손익 ${perf['total_pnl']:.2f}")
            
            # Exit reason 분석
            print("\n=== 청산 사유 분석 ===")
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'total_pnl': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['total_pnl'] += trade['pnl']
            
            for reason, stats in exit_reasons.items():
                avg_pnl = stats['total_pnl'] / stats['count']
                print(f"{reason}: {stats['count']}회, 평균손익 ${avg_pnl:.2f}")
            
            # 트레일링 스탑 분석
            trailing_trades = [t for t in trades if t['trailing_stop_applied']]
            if trailing_trades:
                print(f"\n=== 트레일링 스탑 분석 ===")
                print(f"트레일링 스탑 발동: {len(trailing_trades)}회")
                trailing_pnl = sum(t['pnl'] for t in trailing_trades)
                print(f"트레일링 스탑 총 손익: ${trailing_pnl:.2f}")
        
        # 차트 생성
        if self.config['GENERATE_PLOTS']:
            self.plot_results(trades, equity_curve)
        
        # 거래 내역 저장
        if self.config['SAVE_TRADES']:
            trades_df = pd.DataFrame(trades)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trades_file = os.path.join(self.report_dir, f'trades_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"\n거래 내역 저장: {trades_file}")
            
            # 요약 정보도 저장
            summary = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'final_balance': final_balance
            }
            
            summary_file = os.path.join(self.report_dir, f'summary_{timestamp}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"요약 정보 저장: {summary_file}")
    
    def plot_results(self, trades, equity_curve):
        """결과 시각화"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 자산 곡선
        equity_df = pd.DataFrame(equity_curve)
        axes[0, 0].plot(equity_df['time'], equity_df['balance'], linewidth=2)
        axes[0, 0].axhline(y=self.config['INITIAL_BALANCE'], color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('자산 곡선', fontsize=14, weight='bold')
        axes[0, 0].set_xlabel('시간')
        axes[0, 0].set_ylabel('잔고 ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 최대 손실폭 표시
        running_max = np.maximum.accumulate(equity_df['balance'])
        drawdown = (equity_df['balance'] - running_max) / running_max * 100
        
        ax_dd = axes[0, 0].twinx()
        ax_dd.fill_between(equity_df['time'], drawdown, alpha=0.3, color='red')
        ax_dd.set_ylabel('Drawdown (%)', color='red')
        ax_dd.tick_params(axis='y', labelcolor='red')
        
        # 2. 심볼별 손익
        symbol_pnl = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = 0
            symbol_pnl[symbol] += trade['pnl']
        
        if symbol_pnl:
            symbols = list(symbol_pnl.keys())
            pnls = list(symbol_pnl.values())
            colors = ['green' if p > 0 else 'red' for p in pnls]
            
            bars = axes[0, 1].bar(range(len(symbols)), pnls, color=colors, alpha=0.7)
            axes[0, 1].set_xticks(range(len(symbols)))
            axes[0, 1].set_xticklabels([s.replace('/USDT', '') for s in symbols], rotation=45)
            axes[0, 1].set_title('심볼별 손익', fontsize=14, weight='bold')
            axes[0, 1].set_ylabel('손익 ($)')
            axes[0, 1].grid(True, axis='y', alpha=0.3)
            
            # 손익 값 표시
            for bar, value in zip(bars, pnls):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'${value:.0f}',
                              ha='center', va='bottom' if value > 0 else 'top')
        
        # 3. 월별 수익률
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
            
            # 월별 손익과 거래 수
            monthly_pnl = trades_df.groupby('month')['pnl'].sum()
            monthly_trades = trades_df.groupby('month').size()
            
            if not monthly_pnl.empty:
                ax1 = axes[1, 0]
                ax2 = ax1.twinx()
                
                # 막대 그래프 (손익)
                colors = ['green' if p > 0 else 'red' for p in monthly_pnl.values]
                bars = ax1.bar(range(len(monthly_pnl)), monthly_pnl.values, 
                              color=colors, alpha=0.7, label='손익')
                ax1.set_xticks(range(len(monthly_pnl)))
                ax1.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45)
                ax1.set_ylabel('손익 ($)', color='black')
                ax1.set_title('월별 성과', fontsize=14, weight='bold')
                ax1.grid(True, axis='y', alpha=0.3)
                
                # 선 그래프 (거래 수)
                line = ax2.plot(range(len(monthly_trades)), monthly_trades.values, 
                               'b-o', linewidth=2, label='거래 수')
                ax2.set_ylabel('거래 수', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                
                # 범례
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
        
        # 4. 손익 분포
        if trades:
            pnls = [t['pnl'] for t in trades]
            profit_pcts = [t['profit_percent'] for t in trades]
            
            # 히스토그램
            axes[1, 1].hist(profit_pcts, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('수익률 분포', fontsize=14, weight='bold')
            axes[1, 1].set_xlabel('수익률 (%)')
            axes[1, 1].set_ylabel('빈도')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].grid(True, axis='y', alpha=0.3)
            
            # 통계 정보 추가
            mean_pct = np.mean(profit_pcts)
            axes[1, 1].axvline(x=mean_pct, color='green', linestyle='--', 
                              linewidth=2, label=f'평균: {mean_pct:.2f}%')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.report_dir, f'backtest_report_{timestamp}.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"차트 저장: {report_path}")
        
        # 추가 차트: 승률 추이
        self.plot_win_rate_progression(trades)
        
        # 추가 차트: 포지션별 지속 시간
        self.plot_trade_duration(trades)
    
    def plot_win_rate_progression(self, trades):
        """승률 추이 차트"""
        if not trades:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 누적 승률 계산
        cumulative_wins = 0
        win_rates = []
        
        for i, trade in enumerate(trades):
            if trade['pnl'] > 0:
                cumulative_wins += 1
            win_rate = (cumulative_wins / (i + 1)) * 100
            win_rates.append(win_rate)
        
        # 플롯
        ax.plot(range(1, len(trades) + 1), win_rates, linewidth=2)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% 기준선')
        ax.set_title('누적 승률 추이', fontsize=14, weight='bold')
        ax.set_xlabel('거래 수')
        ax.set_ylabel('승률 (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.report_dir, f'win_rate_progression_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"승률 추이 차트 저장: {filepath}")
    
    def plot_trade_duration(self, trades):
        """거래 지속 시간 분석"""
        if not trades:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 지속 시간 계산
        durations = []
        durations_by_outcome = {'win': [], 'loss': []}
        
        for trade in trades:
            duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
            durations.append(duration)
            
            if trade['pnl'] > 0:
                durations_by_outcome['win'].append(duration)
            else:
                durations_by_outcome['loss'].append(duration)
        
        # 1. 전체 지속 시간 분포
        axes[0].hist(durations, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title('거래 지속 시간 분포', fontsize=14, weight='bold')
        axes[0].set_xlabel('시간 (분)')
        axes[0].set_ylabel('빈도')
        axes[0].grid(True, alpha=0.3)
        
        # 평균 표시
        avg_duration = np.mean(durations)
        axes[0].axvline(x=avg_duration, color='red', linestyle='--', 
                        linewidth=2, label=f'평균: {avg_duration:.1f}분')
        axes[0].legend()
        
        # 2. 승/패별 지속 시간
        if durations_by_outcome['win'] and durations_by_outcome['loss']:
            data = [durations_by_outcome['win'], durations_by_outcome['loss']]
            axes[1].boxplot(data, labels=['승리', '패배'])
            axes[1].set_title('승/패별 거래 지속 시간', fontsize=14, weight='bold')
            axes[1].set_ylabel('시간 (분)')
            axes[1].grid(True, alpha=0.3)
            
            # 평균 표시
            for i, (key, values) in enumerate(durations_by_outcome.items()):
                if values:
                    avg = np.mean(values)
                    axes[1].scatter(i+1, avg, color='red', s=100, zorder=5)
                    axes[1].text(i+1.1, avg, f'{avg:.1f}분', va='center')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.report_dir, f'trade_duration_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"거래 지속 시간 차트 저장: {filepath}")

def main():
    """메인 실행 함수"""
    print("=== 백테스팅 시작 ===")
    print(f"심볼: {CONFIG['SYMBOLS']}")
    print(f"기간: {CONFIG['START_DATE']} ~ {CONFIG['END_DATE']}")
    print(f"초기 자본: ${CONFIG['INITIAL_BALANCE']}")
    print(f"레버리지: {CONFIG['LEVERAGE']}x")
    print(f"포지션 크기: ${CONFIG['MARGIN_AMOUNT']}")
    
    # 1. 데이터 수집
    collector = DataCollector(CONFIG)
    all_data = {}
    
    for symbol in CONFIG['SYMBOLS']:
        try:
            print(f"\n[{symbol}] 데이터 준비 중...")
            data = collector.collect_data(
                symbol,
                CONFIG['TIMEFRAME'],
                CONFIG['START_DATE'],
                CONFIG['END_DATE']
            )
            all_data[symbol] = data
        except Exception as e:
            print(f"[{symbol}] 데이터 수집 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        print("데이터를 수집할 수 없습니다. 종료합니다.")
        return
    
    # 2. 백테스팅
    engine = BacktestEngine(CONFIG)
    
    for symbol, data in all_data.items():
        try:
            engine.run_backtest(symbol, data)
        except Exception as e:
            print(f"[{symbol}] 백테스팅 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 리포트 생성
    reporter = ReportGenerator(CONFIG)
    reporter.generate_report(engine.trades, engine.equity_curve)
    
    print("\n=== 백테스팅 완료 ===")

if __name__ == "__main__":
    main()