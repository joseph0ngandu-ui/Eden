"""
Test news filter to ensure it works correctly
"""
from trading.news_filter import get_news_filter

print("=== Testing News Filter ===\n")

# Initialize filter
news_filter = get_news_filter(buffer_before=30, buffer_after=30)

print("✅ News filter initialized")
print(f"Buffer: 30 minutes before/after high-impact news\n")

# Update news events
print("📰 Fetching high-impact news events...")
success = news_filter.update()

if success:
    print(f"✅ Loaded {len(news_filter.news_events)} news events")
    
    # Show first few events
    if news_filter.news_events:
        print("\nUpcoming high-impact events:")
        for i, event in enumerate(news_filter.news_events[:5]):
            print(f"  {i+1}. {event.get('time', 'N/A')} - {event.get('currency', 'N/A')} - {event.get('event', 'N/A')}")
    else:
        print("ℹ️  No high-impact events found (or outside trading hours)")
else:
    print("⚠️  Could not fetch news (this is OK - will use cached data)")

# Test if we're in news time
print("\n🔍 Checking if current time is news time...")
test_pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm']
is_blocked, reason = news_filter.is_news_time(test_pairs)

if is_blocked:
    print(f"🚫 TRADING BLOCKED: {reason}")
else:
    print(f"✅ Safe to trade (no high-impact news in the next 30 minutes)")

# Get next event
next_event = news_filter.get_next_news_event(test_pairs)
if next_event:
    print(f"\n📅 Next high-impact event:")
    print(f"   Time: {next_event.get('time', 'N/A')}")
    print(f"   Currency: {next_event.get('currency', 'N/A')}")
    print(f"   Event: {next_event.get('event', 'N/A')}")
else:
    print("\nℹ️  No upcoming news events found")

print("\n✅ News filter test complete!")
print("\nNote: News filter will:")
print("  • Block trades 30 min before high-impact news")
print("  • Block trades 30 min after high-impact news")
print("  • Only blocks if news affects the trading currency")
print("  • Can be disabled in config: news_filter_enabled: false")
