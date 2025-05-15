from backtest.backtest_engine import BacktestEngine
from backtest.data_collector import DataCollector
from backtest.report_generator import ReportGenerator
from config.trading_config import TRADING_SYMBOLS
import pandas as pd

def main():
    # 1. 설정
    config = {
        'initial_balance': 1000,
        'leverage': 10,
        'margin_amount': 20,
        'max_daily_loss': 10
    }
    
    # 2. 데이터 수집 (필요시)
    collector = DataCollector()
    
    # 3개월 데이터 수집 예시
    start_date = '2024-10-01'
    end_date = '2025-01-31'
    
    for symbol in TRADING_SYMBOLS[:5]:  # 테스트로 5개만
        try:
            collector.collect_historical_data(
                symbol,
                '5m',
                start_date,
                end_date
            )
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
    
    # 3. 백테스팅 실행
    engine = BacktestEngine(config)
    results = engine.run_backtest(
        TRADING_SYMBOLS[:5],
        start_date,
        end_date
    )
    
    # 4. 리포트 생성
    reporter = ReportGenerator()
    report = reporter.generate_html_report(results)
    
    print("백테스팅 완료!")
    print(f"최종 잔고: ${results['final_balance']}")
    print(f"승률: {results['win_rate']}%")
    print(f"샤프 비율: {results['sharpe_ratio']}")

if __name__ == "__main__":
    main()