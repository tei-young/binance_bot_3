import sys
sys.path.append('..')  # 상위 디렉토리 접근

from binance_bot import TradingBot
import pandas as pd
import numpy as np

class StrategyTester:
    def __init__(self):
        # 현재 봇의 로직을 상속
        self.bot_logic = TradingBot(None, None)  # API 키 없이 초기화
        
    def test_current_strategy(self, data: pd.DataFrame, symbol: str):
        """현재 전략 테스트"""
        results = []
        
        for i in range(300, len(data)):  # 충분한 히스토리 확보
            current_data = data.iloc[:i+1].copy()
            
            # 현재 봇의 로직 활용
            position_type, crosses = self.bot_logic.check_entry_conditions(
                current_data, 
                symbol
            )
            
            if position_type:
                # 가상 거래 실행
                trade = self.execute_paper_trade(
                    current_data,
                    position_type,
                    crosses
                )
                results.append(trade)
        
        return self.analyze_results(results)
    
    def test_modified_strategy(self, data: pd.DataFrame, modifications: dict):
        """수정된 전략 테스트"""
        # 예: EMA 기간 변경, TP 비율 변경 등
        pass