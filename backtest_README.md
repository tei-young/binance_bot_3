# 백테스팅 가이드

## 빠른 시작
1. `auto_backtest.py` 파일의 CONFIG 섹션 수정
2. `python auto_backtest.py` 실행

## 설정 변경
```python
CONFIG = {
    'SYMBOLS': ['BTC/USDT', 'ETH/USDT'],  # 테스트할 심볼
    'START_DATE': '2024-01-01',           # 시작일
    'END_DATE': '2024-12-31',             # 종료일
    'INITIAL_BALANCE': 1000,              # 초기 자본
    # ... 기타 설정
}