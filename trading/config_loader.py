#!/usr/bin/env python3
"""
Configuration Loader

Dynamically loads strategy parameters from config/strategy.yml instead of hardcoding.
This enables easy optimization via config swaps and runtime customization.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage strategy configuration from YAML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to strategy.yml. Defaults to config/strategy.yml
        """
        if config_path is None:
            # Default to config/strategy.yml relative to project root
            config_path = Path(__file__).parent.parent / "config" / "strategy.yml"
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            self.config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            self.config = {}
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value from config.
        
        Args:
            key: Parameter key (supports dot notation, e.g., 'parameters.fast_ma_period')
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        params = self.get_parameter('parameters', {})
        return {
            'fast_ma_period': params.get('fast_ma_period', 3),
            'slow_ma_period': params.get('slow_ma_period', 10),
            'timeframe': params.get('timeframe', 'M5'),
            'hold_bars': params.get('hold_bars', 5),
        }
    
    def get_trading_symbols(self) -> list:
        """Get list of trading symbols."""
        # Try to get from strategy config
        symbols = self.get_parameter('trading_symbols', [])
        if symbols:
            return symbols
        
        # Fallback to defaults (Exness broker format with 'm' suffix)
        return [
            "EURUSDm",      # Pro_Overlap_Scalper, Pro_Volatility_Expansion
            "GBPUSDm",      # Pro_Overlap_Scalper, Pro_Volatility_Expansion
            "USDJPYm",      # Pro_Asian_Fade, Pro_Volatility_Expansion
            "AUDJPYm",      # Pro_Asian_Fade, Pro_Volatility_Expansion
            "XAUUSDm",      # Pro_Gold_Breakout
        ]
    
    def get_risk_management(self) -> Dict[str, Any]:
        """Get risk management parameters."""
        risk = self.get_parameter('risk_management', {})
        return {
            'position_size': risk.get('position_size', 1.0),
            'max_concurrent_positions': risk.get('max_concurrent_positions', 10),
            'max_drawdown_percent': risk.get('max_drawdown_percent', None),
            'stop_loss_pips': risk.get('stop_loss_pips'),
            'take_profit_pips': risk.get('take_profit_pips'),
        }
    
    def get_live_trading_config(self) -> Dict[str, Any]:
        """Get live trading configuration."""
        live = self.get_parameter('live_trading', {})
        return {
            'enabled': live.get('enabled', True),
            'check_interval': live.get('check_interval', 300),
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        logging_cfg = self.get_parameter('logging', {})
        return {
            'level': logging_cfg.get('level', 'INFO'),
            'format': logging_cfg.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            'file': logging_cfg.get('file', 'logs/trading.log'),
        }
    
    def get_version(self) -> str:
        """Get strategy version."""
        return self.get_parameter('strategy.version', '1.0.0')
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return self.get_parameter('strategy.name', 'MA Crossover Strategy')
    
    def get_growth_mode_config(self) -> Dict[str, Any]:
        """Get growth mode configuration."""
        growth = self.get_parameter('growth_mode', {})
        return {
            'enabled': growth.get('enabled', False),
            'high_aggression_below': growth.get('high_aggression_below', 30.0),
            'equity_step_size': growth.get('equity_step_size', 50.0),
            'equity_step_drawdown_limit': growth.get('equity_step_drawdown_limit', 0.15),
            'lot_sizing': growth.get('lot_sizing', 'simple'),
            'pip_value': growth.get('pip_value', 10.0),
        }
