#!/usr/bin/env python
"""시스템 시간과 Binance 서버 시간 비교"""
import ccxt
import time
from datetime import datetime

print("=" * 60)
print("TIME SYNCHRONIZATION DIAGNOSTIC")
print("=" * 60)
print()

# 로컬 시간 측정
local_before = int(time.time() * 1000)
print(f"1. Local system time:")
print(f"   {datetime.fromtimestamp(local_before / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
print(f"   ({local_before}ms)")
print()

# Binance 서버 시간 가져오기
print("2. Fetching Binance server time...")
exchange = ccxt.binance({'enableRateLimit': True})
try:
    server_time = exchange.fetch_time()
    local_after = int(time.time() * 1000)

    print(f"   {datetime.fromtimestamp(server_time / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"   ({server_time}ms)")
    print()

    # 시간 차이 계산 (네트워크 지연 고려)
    local_avg = (local_before + local_after) // 2
    network_latency = local_after - local_before
    diff_ms = local_avg - server_time

    print(f"3. Analysis:")
    print(f"   Network latency: {network_latency}ms")
    print(f"   Time difference: {diff_ms}ms")
    print()

    print("4. Diagnosis:")
    if diff_ms > 1000:
        print(f"   ❌ CRITICAL: Your local time is {diff_ms}ms AHEAD of Binance")
        print(f"   This is WHY you're getting timestamp errors!")
        print()
        print("   Solutions:")
        print("   1. Enable NTP time sync: timedatectl set-ntp true")
        print("   2. Manually sync time: ntpdate -u pool.ntp.org")
        print("   3. Or set timeDifference manually in code")
    elif diff_ms < -1000:
        print(f"   ℹ️ INFO: Your local time is {abs(diff_ms)}ms BEHIND Binance")
        print(f"   This is usually OK (won't cause 'ahead' errors)")
    else:
        print(f"   ✅ GOOD: Time difference is within acceptable range")
        print(f"   The timestamp errors might be from other causes")

except Exception as e:
    print(f"   ERROR: {e}")

print()
print("=" * 60)
