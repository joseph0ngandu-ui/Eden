"""
Eden Configuration Management System
Professional settings management with UI integration and persistence
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

# Try to import encryption for secure API key storage
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GeneralConfig:
    """General application configuration"""
    startup_mode: str = "GUI"  # GUI, CLI, ASK
    auto_check_updates: bool = True
    save_on_exit: bool = True
    recent_files_count: int = 10
    theme: str = "system"  # light, dark, system
    language: str = "en"
    confirm_exit: bool = True
    restore_window_state: bool = True

@dataclass
class DataConfig:
    """Data sources and management configuration"""
    default_data_source: str = "yahoo"
    cache_data: bool = True
    cache_duration_hours: int = 24
    auto_update_data: bool = True
    max_cache_size_mb: int = 1024
    data_directory: str = ""
    backup_data: bool = True
    
@dataclass
class TradingConfig:
    """Trading and risk management configuration"""
    max_position_size: float = 0.10  # 10% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    require_confirmation: bool = True
    paper_trading: bool = True
    default_order_type: str = "market"
    commission_per_trade: float = 0.0
    slippage_bps: int = 5  # basis points

@dataclass
class PerformanceConfig:
    """System performance configuration"""
    cpu_cores: Union[str, int] = "auto"
    memory_limit: Union[str, int] = "auto"
    log_level: str = "INFO"
    enable_gpu: bool = False
    max_concurrent_downloads: int = 5
    optimize_for_speed: bool = True

@dataclass
class UIConfig:
    """User interface configuration"""
    window_width: int = 1200
    window_height: int = 800
    window_x: int = -1  # -1 means center
    window_y: int = -1  # -1 means center
    maximized: bool = False
    show_splash: bool = True
    animation_enabled: bool = True
    show_tooltips: bool = True
    toolbar_size: str = "medium"  # small, medium, large
    chart_style: str = "candlestick"

@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    encrypt_api_keys: bool = True
    auto_lock_timeout: int = 30  # minutes
    require_password: bool = False
    log_user_actions: bool = True
    anonymous_analytics: bool = True

@dataclass
class EdenConfig:
    """Main Eden configuration class"""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    last_modified: str = ""
    user_id: str = ""

class ConfigManager:
    """Professional configuration management system"""
    
    def __init__(self):
        self.app_name = "Eden"
        self.config_dir = Path.home() / "AppData" / "Roaming" / self.app_name
        self.config_file = self.config_dir / "config.yaml"
        self.api_keys_file = self.config_dir / "api_keys.encrypted"
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption key for API keys
        self._encryption_key = None
        self._load_or_create_encryption_key()
        
        # Load or create default configuration
        self.config = self._load_config()
        
    def _load_or_create_encryption_key(self):
        """Load or create encryption key for sensitive data"""
        if not ENCRYPTION_AVAILABLE:
            logger.warning("Cryptography not available - API keys will be stored in plain text")
            return
        
        key_file = self.config_dir / ".key"
        
        try:
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self._encryption_key)
                
                # Hide the key file on Windows
                if os.name == 'nt':
                    import subprocess
                    subprocess.run(['attrib', '+H', str(key_file)], check=False)
                    
        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self._encryption_key = None
    
    def _load_config(self) -> EdenConfig:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Convert dict to dataclass
                config = EdenConfig()
                
                if 'general' in data:
                    config.general = GeneralConfig(**data['general'])
                if 'data' in data:
                    config.data = DataConfig(**data['data'])
                if 'trading' in data:
                    config.trading = TradingConfig(**data['trading'])
                if 'performance' in data:
                    config.performance = PerformanceConfig(**data['performance'])
                if 'ui' in data:
                    config.ui = UIConfig(**data['ui'])
                if 'security' in data:
                    config.security = SecurityConfig(**data['security'])
                
                # Update metadata
                config.config_version = data.get('config_version', '1.0.0')
                config.last_modified = data.get('last_modified', '')
                config.user_id = data.get('user_id', '')
                
                logger.info("Configuration loaded successfully")
                return config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                logger.info("Using default configuration")
        
        # Return default configuration
        config = EdenConfig()
        config.last_modified = datetime.now().isoformat()
        return config
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            # Update metadata
            self.config.last_modified = datetime.now().isoformat()
            
            # Convert to dict
            config_dict = asdict(self.config)
            
            # Save to YAML file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        try:
            category_obj = getattr(self.config, category, None)
            if category_obj:
                return getattr(category_obj, key, default)
            return default
        except Exception:
            return default
    
    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """Set a specific setting value"""
        try:
            category_obj = getattr(self.config, category, None)
            if category_obj and hasattr(category_obj, key):
                setattr(category_obj, key, value)
                return True
            return False
        except Exception:
            return False
    
    def reset_to_defaults(self, category: Optional[str] = None) -> bool:
        """Reset configuration to defaults"""
        try:
            if category:
                # Reset specific category
                if category == 'general':
                    self.config.general = GeneralConfig()
                elif category == 'data':
                    self.config.data = DataConfig()
                elif category == 'trading':
                    self.config.trading = TradingConfig()
                elif category == 'performance':
                    self.config.performance = PerformanceConfig()
                elif category == 'ui':
                    self.config.ui = UIConfig()
                elif category == 'security':
                    self.config.security = SecurityConfig()
                else:
                    return False
            else:
                # Reset all
                self.config = EdenConfig()
                self.config.last_modified = datetime.now().isoformat()
            
            logger.info(f"Configuration reset to defaults: {category or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def export_config(self, file_path: str) -> bool:
        """Export configuration to file"""
        try:
            export_data = asdict(self.config)
            export_data['export_timestamp'] = datetime.now().isoformat()
            export_data['export_source'] = 'Eden Configuration Manager'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(export_data, f, indent=2)
                else:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, file_path: str) -> bool:
        """Import configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            # Backup current config
            backup_file = self.config_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            self.export_config(str(backup_file))
            
            # Import new configuration
            imported_config = EdenConfig()
            
            if 'general' in data:
                imported_config.general = GeneralConfig(**data['general'])
            if 'data' in data:
                imported_config.data = DataConfig(**data['data'])
            if 'trading' in data:
                imported_config.trading = TradingConfig(**data['trading'])
            if 'performance' in data:
                imported_config.performance = PerformanceConfig(**data['performance'])
            if 'ui' in data:
                imported_config.ui = UIConfig(**data['ui'])
            if 'security' in data:
                imported_config.security = SecurityConfig(**data['security'])
            
            # Update metadata
            imported_config.last_modified = datetime.now().isoformat()
            imported_config.config_version = data.get('config_version', '1.0.0')
            
            self.config = imported_config
            
            logger.info(f"Configuration imported from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
    
    def store_api_key(self, provider: str, api_key: str) -> bool:
        """Store API key securely"""
        try:
            # Load existing keys
            api_keys = self._load_api_keys()
            
            # Add/update key
            api_keys[provider] = api_key
            
            # Save encrypted
            return self._save_api_keys(api_keys)
            
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Retrieve API key securely"""
        try:
            api_keys = self._load_api_keys()
            return api_keys.get(provider)
        except Exception as e:
            logger.error(f"Failed to retrieve API key: {e}")
            return None
    
    def remove_api_key(self, provider: str) -> bool:
        """Remove stored API key"""
        try:
            api_keys = self._load_api_keys()
            if provider in api_keys:
                del api_keys[provider]
                return self._save_api_keys(api_keys)
            return True
        except Exception as e:
            logger.error(f"Failed to remove API key: {e}")
            return False
    
    def list_api_keys(self) -> List[str]:
        """List stored API key providers"""
        try:
            api_keys = self._load_api_keys()
            return list(api_keys.keys())
        except Exception:
            return []
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from encrypted file"""
        if not self.api_keys_file.exists():
            return {}
        
        try:
            with open(self.api_keys_file, 'rb') as f:
                encrypted_data = f.read()
            
            if ENCRYPTION_AVAILABLE and self._encryption_key:
                fernet = Fernet(self._encryption_key)
                decrypted_data = fernet.decrypt(encrypted_data)
                return json.loads(decrypted_data.decode())
            else:
                # Fallback to plain text (not recommended)
                return json.loads(encrypted_data.decode())
                
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}
    
    def _save_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """Save API keys to encrypted file"""
        try:
            data = json.dumps(api_keys).encode()
            
            if ENCRYPTION_AVAILABLE and self._encryption_key:
                fernet = Fernet(self._encryption_key)
                encrypted_data = fernet.encrypt(data)
            else:
                encrypted_data = data
                logger.warning("API keys stored without encryption")
            
            with open(self.api_keys_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        try:
            # Validate general settings
            if self.config.general.startup_mode not in ['GUI', 'CLI', 'ASK']:
                issues.append("Invalid startup mode")
            
            if not 1 <= self.config.general.recent_files_count <= 50:
                issues.append("Recent files count must be between 1 and 50")
            
            # Validate data settings
            if self.config.data.cache_duration_hours < 0:
                issues.append("Cache duration cannot be negative")
            
            if self.config.data.max_cache_size_mb < 10:
                issues.append("Cache size must be at least 10 MB")
            
            # Validate trading settings
            if not 0 < self.config.trading.max_position_size <= 1:
                issues.append("Max position size must be between 0 and 1")
            
            if not 0 <= self.config.trading.max_daily_loss <= 1:
                issues.append("Max daily loss must be between 0 and 1")
            
            # Validate UI settings
            if not 600 <= self.config.ui.window_width <= 5000:
                issues.append("Window width must be between 600 and 5000")
            
            if not 400 <= self.config.ui.window_height <= 3000:
                issues.append("Window height must be between 400 and 3000")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_setting(category: str, key: str, default: Any = None) -> Any:
    """Convenience function to get a setting"""
    return get_config_manager().get_setting(category, key, default)

def set_setting(category: str, key: str, value: Any) -> bool:
    """Convenience function to set a setting"""
    result = get_config_manager().set_setting(category, key, value)
    if result:
        get_config_manager().save_config()
    return result

if __name__ == "__main__":
    # Test configuration management
    config_manager = ConfigManager()
    
    print("🔧 Eden Configuration Manager Test")
    print(f"Config Directory: {config_manager.config_dir}")
    print(f"Startup Mode: {config_manager.config.general.startup_mode}")
    print(f"Data Source: {config_manager.config.data.default_data_source}")
    print(f"Paper Trading: {config_manager.config.trading.paper_trading}")
    
    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")
    
    # Save configuration
    if config_manager.save_config():
        print("✅ Configuration saved successfully")
    
    print(f"📊 API Keys: {len(config_manager.list_api_keys())} stored")