#!/usr/bin/env python3
"""
News Event Filter
Blocks trades during high-impact news events to avoid wide spreads and slippage.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from bs4 import BeautifulSoup
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsEventFilter:
    """Filter trades based on high-impact news events."""
    
    def __init__(self, 
                 buffer_minutes_before: int = 30,
                 buffer_minutes_after: int = 30,
                 cache_hours: int = 24):
        """
        Initialize news filter.
        
        Args:
            buffer_minutes_before: Minutes before news to block trading
            buffer_minutes_after: Minutes after news to block trading
            cache_hours: Hours to cache news data before refreshing
        """
        self.buffer_before = timedelta(minutes=buffer_minutes_before)
        self.buffer_after = timedelta(minutes=buffer_minutes_after)
        self.cache_hours = cache_hours
        
        self.cache_file = Path(__file__).parent.parent / "logs" / "news_cache.json"
        self.cache_file.parent.mkdir(exist_ok=True)
        
        self.news_events: List[Dict] = []
        self.last_update = None
    
    def _fetch_forex_factory_news(self) -> List[Dict]:
        """
        Fetch high-impact news from ForexFactory calendar.
        Returns list of news events with time and currency.
        """
        try:
            # ForexFactory calendar URL
            url = "https://www.forexfactory.com/calendar"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch news: HTTP {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            events = []
            
            # Parse calendar events (high impact only)
            # Note: This is a simplified parser - ForexFactory structure may change
            for row in soup.find_all('tr', class_='calendar__row'):
                try:
                    impact = row.find('td', class_='calendar__impact')
                    if impact and 'high' in impact.get('class', []):
                        time_elem = row.find('td', class_='calendar__time')
                        currency_elem = row.find('td', class_='calendar__currency')
                        event_elem = row.find('td', class_='calendar__event')
                        
                        if time_elem and currency_elem and event_elem:
                            time_str = time_elem.get_text(strip=True)
                            if time_str and time_str != 'All Day':
                                events.append({
                                    'time': time_str,
                                    'currency': currency_elem.get_text(strip=True),
                                    'event': event_elem.get_text(strip=True),
                                    'impact': 'high'
                                })
                except Exception as e:
                    continue
            
            logger.info(f"Fetched {len(events)} high-impact news events")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching ForexFactory news: {e}")
            return []
    
    def _load_cache(self) -> bool:
        """Load news from cache if recent enough."""
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            cache_time = datetime.fromisoformat(cache['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=self.cache_hours):
                self.news_events = cache['events']
                self.last_update = cache_time
                logger.info(f"Loaded {len(self.news_events)} events from cache")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Failed to load news cache: {e}")
            return False
    
    def _save_cache(self):
        """Save news events to cache."""
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'events': self.news_events
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save news cache: {e}")
    
    def update(self) -> bool:
        """
        Update news events from ForexFactory.
        Returns True if update successful.
        """
        # Try cache first
        if self._load_cache():
            return True
        
        # Fetch fresh data
        events = self._fetch_forex_factory_news()
        if events:
            self.news_events = events
            self.last_update = datetime.now()
            self._save_cache()
            return True
        
        return False
    
    def is_news_time(self, currency_pairs: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
        """
        Check if current time is within news event buffer.
        
        Args:
            currency_pairs: List of currency pairs to check (e.g., ['EURUSD', 'GBPUSD'])
                           If None, checks all currencies
        
        Returns:
            (is_blocked, reason): True if trading should be blocked, with reason
        """
        # Update news if cache is old
        if not self.last_update or (datetime.now() - self.last_update) > timedelta(hours=self.cache_hours):
            self.update()
        
        # If no events loaded (API down), allow trading (fail-safe open)
        if not self.news_events:
            logger.debug("News filter: No events loaded, allowing trade")
            return False, None
        
        now = datetime.now()
        
        # Extract currencies from pairs
        currencies = set()
        if currency_pairs:
            for pair in currency_pairs:
                # EURUSDm -> EUR, USD
                clean_pair = pair.replace('m', '').replace('/', '')
                if len(clean_pair) >= 6:
                    currencies.add(clean_pair[:3])  # First currency
                    currencies.add(clean_pair[3:6])  # Second currency
        
        # Check each news event
        for event in self.news_events:
            try:
                # Parse event time (assuming today's events for now)
                # In production, you'd want to handle multi-day calendars
                event_time_str = event.get('time', '')
                if not event_time_str or ':' not in event_time_str:
                    continue
                
                # Parse time (format: "10:30am" or "2:15pm")
                event_hour, event_min_period = event_time_str.split(':')
                event_min = event_min_period[:2]
                period = event_min_period[2:].lower()
                
                event_hour = int(event_hour)
                event_min = int(event_min)
                
                if period == 'pm' and event_hour != 12:
                    event_hour += 12
                elif period == 'am' and event_hour == 12:
                    event_hour = 0
                
                event_time = now.replace(hour=event_hour, minute=event_min, second=0, microsecond=0)
                
                # Check if we're in the buffer zone
                time_to_event = event_time - now
                time_since_event = now - event_time
                
                is_before_event = -self.buffer_before <= time_to_event <= timedelta(0)
                is_after_event = timedelta(0) <= time_since_event <= self.buffer_after
                
                if is_before_event or is_after_event:
                    # Check if this event affects our trading currencies
                    event_currency = event.get('currency', '')
                    if not currencies or event_currency in currencies or not event_currency:
                        reason = f"High-impact {event_currency} news: '{event['event']}' at {event_time_str}"
                        logger.warning(f"BLOCKING TRADE: {reason}")
                        return True, reason
                
            except Exception as e:
                logger.debug(f"Error parsing news event: {e}")
                continue
        
        return False, None
    
    def get_next_news_event(self, currency_pairs: Optional[List[str]] = None) -> Optional[Dict]:
        """Get the next upcoming high-impact news event."""
        if not self.news_events:
            self.update()
        
        # Extract currencies
        currencies = set()
        if currency_pairs:
            for pair in currency_pairs:
                clean_pair = pair.replace('m', '').replace('/', '')
                if len(clean_pair) >= 6:
                    currencies.add(clean_pair[:3])
                    currencies.add(clean_pair[3:6])
        
        now = datetime.now()
        upcoming_events = []
        
        for event in self.news_events:
            try:
                event_time_str = event.get('time', '')
                if not event_time_str or ':' not in event_time_str:
                    continue
                
                # Simple time parsing (same as above)
                parts = event_time_str.split(':')
                event_hour = int(parts[0])
                event_min_period = parts[1]
                event_min = int(event_min_period[:2])
                period = event_min_period[2:].lower()
                
                if period == 'pm' and event_hour != 12:
                    event_hour += 12
                elif period == 'am' and event_hour == 12:
                    event_hour = 0
                
                event_time = now.replace(hour=event_hour, minute=event_min, second=0)
                
                if event_time > now:
                    event_currency = event.get('currency', '')
                    if not currencies or event_currency in currencies:
                        upcoming_events.append({
                            **event,
                            'event_time': event_time
                        })
            except:
                continue
        
        if upcoming_events:
            return min(upcoming_events, key=lambda x: x['event_time'])
        
        return None


# Singleton instance
_news_filter = None

def get_news_filter(buffer_before: int = 30, buffer_after: int = 30) -> NewsEventFilter:
    """Get or create global news filter instance."""
    global _news_filter
    if _news_filter is None:
        _news_filter = NewsEventFilter(buffer_before, buffer_after)
    return _news_filter
