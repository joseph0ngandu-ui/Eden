"""
Multi-Timeframe Data Fetcher
Fetches VIX100 data from MetaTrader 5 across multiple timeframes
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class MTFDataFetcher:
    """Fetch multi-timeframe OHLCV data from MT5 for VIX100"""

    def __init__(
        self,
        symbol: str = "Volatility 100 Index",
        cache_dir: Path = Path("data/cache"),
        raw_dir: Path = Path("data/raw"),
    ):
        self.symbol = symbol
        self.cache_dir = Path(cache_dir)
        self.raw_dir = Path(raw_dir)

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Try alternative symbol names for VIX100
        self.symbol_alternatives = [
            "Volatility 100 Index",
            "VIX100",
            "Volatility100",
            "Vol100",
            "VIX 100",
        ]

        self.mt5_available = False
        self.mt5 = None
        self._init_mt5()

    def _init_mt5(self):
        """Initialize MetaTrader 5 connection"""
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5

            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            logger.info("MT5 initialized successfully")
            self.mt5_available = True

            # Try to find the correct symbol name
            for symbol_name in self.symbol_alternatives:
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info is not None:
                    self.symbol = symbol_name
                    logger.info(f"Using symbol: {symbol_name}")
                    # Enable symbol
                    mt5.symbol_select(symbol_name, True)
                    return True

            logger.warning(
                f"VIX100 symbol not found in MT5. Tried: {self.symbol_alternatives}"
            )
            return False

        except ImportError:
            logger.warning("MetaTrader5 module not available")
            return False
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def _get_mt5_timeframe(self, timeframe: str):
        """Convert string timeframe to MT5 constant"""
        if not self.mt5:
            return None

        mapping = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M5": self.mt5.TIMEFRAME_M5,
            "15M": self.mt5.TIMEFRAME_M15,
            "1H": self.mt5.TIMEFRAME_H1,
            "4H": self.mt5.TIMEFRAME_H4,
            "1D": self.mt5.TIMEFRAME_D1,
        }
        return mapping.get(timeframe)

    def fetch_timeframe(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single timeframe
        """
        # Check cache first
        cache_file = (
            self.cache_dir
            / f"{self.symbol.replace(' ', '_')}_{timeframe}_{start_date.date()}_{end_date.date()}.csv"
        )

        if use_cache and cache_file.exists():
            try:
                logger.info(f"Loading {timeframe} from cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fetch from MT5
        if not self.mt5_available:
            logger.error(f"Cannot fetch {timeframe}: MT5 not available")
            return None

        try:
            tf_const = self._get_mt5_timeframe(timeframe)
            if tf_const is None:
                logger.error(f"Unknown timeframe: {timeframe}")
                return None

            logger.info(
                f"Fetching {self.symbol} {timeframe} from {start_date} to {end_date}"
            )

            rates = self.mt5.copy_rates_range(
                self.symbol, tf_const, start_date, end_date
            )

            if rates is None or len(rates) == 0:
                logger.error(
                    f"No data returned for {timeframe}: {self.mt5.last_error()}"
                )
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("datetime")
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df = df.rename(columns={"tick_volume": "volume"})

            logger.info(f"Fetched {len(df)} bars for {timeframe}")

            # Save to cache
            try:
                df.to_csv(cache_file)
                logger.info(f"Cached {timeframe} data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")

            # Save raw data
            try:
                raw_file = (
                    self.raw_dir
                    / f"{self.symbol.replace(' ', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                df.to_csv(raw_file)
            except Exception as e:
                logger.warning(f"Failed to save raw data: {e}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {timeframe}: {e}")
            return None

    def fetch_all_timeframes(
        self, timeframes: List[str], days_back: int = 7, use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all specified timeframes

        Args:
            timeframes: List of timeframe strings (e.g., ['M1', 'M5', '15M', '1H', '4H', '1D'])
            days_back: Number of days to fetch (from today)
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping timeframe -> DataFrame
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Fetching timeframes: {timeframes}")
        logger.info(f"Date range: {start_date} to {end_date}")

        data_dict = {}

        for tf in timeframes:
            df = self.fetch_timeframe(tf, start_date, end_date, use_cache=use_cache)
            if df is not None and not df.empty:
                data_dict[tf] = df
                logger.info(f"✓ {tf}: {len(df)} bars")
            else:
                logger.warning(f"✗ {tf}: No data")

        if not data_dict:
            logger.error("No data fetched for any timeframe")
            # Try to load from any available cache
            logger.info("Attempting to load from cache...")
            for tf in timeframes:
                cache_pattern = f"{self.symbol.replace(' ', '_')}_{tf}_*.csv"
                cache_files = list(self.cache_dir.glob(cache_pattern))
                if cache_files:
                    latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
                    try:
                        df = pd.read_csv(latest_cache, index_col=0, parse_dates=True)
                        data_dict[tf] = df
                        logger.info(f"✓ Loaded {tf} from cache: {latest_cache.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load cache {latest_cache}: {e}")

        return data_dict

    def fetch_all_timeframes_range(
        self,
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for specified timeframes over an explicit date range."""
        logger.info(
            f"Fetching timeframes {timeframes} between {start_date} and {end_date}"
        )
        data_dict: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            df = self.fetch_timeframe(tf, start_date, end_date, use_cache=use_cache)
            if df is not None and not df.empty:
                data_dict[tf] = df
        return data_dict

    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.mt5_available and self.mt5:
            try:
                self.mt5.shutdown()
                logger.info("MT5 connection closed")
            except Exception as e:
                logger.warning(f"Error closing MT5: {e}")


def fetch_vix100_data(
    timeframes: List[str] = None, days_back: int = 7, use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch VIX100 data

    Args:
        timeframes: List of timeframes (default: ['M1', 'M5', '15M', '1H', '4H', '1D'])
        days_back: Number of days to fetch
        use_cache: Use cached data if available

    Returns:
        Dictionary mapping timeframe -> DataFrame
    """
    if timeframes is None:
        timeframes = ["M1", "M5", "15M", "1H", "4H", "1D"]

    fetcher = MTFDataFetcher()

    try:
        data = fetcher.fetch_all_timeframes(
            timeframes=timeframes, days_back=days_back, use_cache=use_cache
        )
        return data
    finally:
        fetcher.shutdown()


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    print("Testing MTF Data Fetcher...")
    data = fetch_vix100_data(
        timeframes=["M5", "15M", "1H", "4H"], days_back=7, use_cache=False
    )

    print(f"\nFetched {len(data)} timeframes:")
    for tf, df in data.items():
        print(f"  {tf}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
