바이낸스 자동매매 봇 구조 설명
1. 주요 매매 로직

EMA(12,26)와 MACD 크로스를 주요 시그널로 활용
MA angles의 색상(green/red)을 추가 필터로 사용
25분 이내에 발생한 두 크로스(EMA, MACD)가 동일 방향일 때 진입
먼저 발생한 크로스의 캔들 high/low를 SL로 설정
현재가 기준으로 지정가 진입
TP는 SL 거리의 2.25배

2. 위험 관리

최대 일일 손실 제한 (MAX_DAILY_LOSS)
기본 SL/TP 설정
트레일링 스탑 (1.4% 이상 수익 시 0.5% 이익 보장)
최근 추가: -50% 손실 시 반대 시그널 체크 및 포지션 전환

3. 주요 클래스 및 메서드

초기화 및 설정

__init__: 봇 초기화, 레버리지 설정
setup_logging: 로깅 시스템 설정

지표 계산

calculate_indicators: 기본 지표 계산 (EMA, MACD)
calculate_jurik_ma: MA angles 계산
calculate_angle: 기울기 계산

크로스 체크

store_cross_data: 크로스 발생 및 저장
check_cross_validity: 크로스 유효성 검증
analyze_cross_pattern: 크로스 패턴 분석
is_strong_cross: 크로스 강도 검증

포지션 관리

execute_trade: 포지션 진입
determine_stop_loss: SL 계산
calculate_take_profit: TP 계산
update_trailing_stop: 트레일링 스탑 업데이트
check_position_loss: -50% 손실 체크 (신규)
check_opposite_signal: 반대 시그널 체크 (신규)
close_and_convert_position: 포지션 전환 (신규)

상태 관리

check_order_status: 주문 상태 관리
check_existing_position: 기존 포지션 확인
update_daily_pnl: 손익 업데이트
check_daily_pnl: 일일 손실 한도 체크

메인 루프

run: 메인 실행 루프

4. 데이터 구조

pythonCopyself.positions = {
    symbol: {
        'entry_order': None,
        'sl_order': None,
        'tp_order': None,
        'trailing_sl_order': None,
        'position_type': None,
        'trailing_stop_applied': False,
        'entry_price': None
    }
}

self.cross_history = {
    symbol: {
        'ema': [],
        'macd': []
    }
}

5. 주요 설정 값

pythonCopyTIMEFRAME = '5m'      # 봉 단위
LEVERAGE = 10         # 레버리지
MARGIN_AMOUNT = 20    # 증거금 (USDT)
MAX_DAILY_LOSS = 10   # 일일 최대 손실 (USDT)
THRESHOLD = 4         # MA angles 임계값

6. 신규 기능 개발 시 고려사항

기존 SL/TP 로직과의 충돌 방지
포지션 상태 관리의 일관성 유지
적절한 로깅 구현
예외 처리 철저
기존 기능과의 독립성 보장

이 봇은 지속적으로 발전하고 있으며, 최근 큰 손실을 방지하기 위한 포지션 전환 로직이 추가되었습니다. 신규 기능 개발 시 기존 매매 로직의 안정성을 해치지 않도록 주의해야 합니다.
