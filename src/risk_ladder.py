#!/usr/bin/env python3
"""
Risk Ladder & Equity-Based Position Sizing

Implements:
1. Risk tier system - adjusts lot sizing as balance increases
2. ATR/Equity-based lot sizing - dynamic position sizing per trade
3. Equity step lock - protects growth stages from drawdown
4. Dynamic compounding - recalculates position size per trade
5. Aggression filter - fast scaling for very small accounts

Ensures sustainable compounding while protecting profits.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tier levels."""
    ULTRA_AGGRESSIVE = "ULTRA_AGGRESSIVE"  # $10-$30
    VERY_AGGRESSIVE = "VERY_AGGRESSIVE"    # $30-$100
    AGGRESSIVE = "AGGRESSIVE"              # $100-$500
    MODERATE = "MODERATE"                  # $500-$1000
    CONSERVATIVE = "CONSERVATIVE"          # $1000+


@dataclass
class RiskTierConfig:
    """Configuration for a risk tier."""
    balance_min: float
    balance_max: float
    risk_per_trade: float  # Percentage, e.g., 10.0 for 10%
    tier: RiskTier
    multiplier: float = 1.0  # Position size multiplier


@dataclass
class EquityStep:
    """Tracks an equity milestone."""
    step_balance: float
    reached_at: datetime
    highest_equity_in_step: float
    lowest_equity_in_step: float
    trades_in_step: int = 0
    step_pnl: float = 0.0


class RiskLadder:
    """
    Dynamic risk management system with tiered position sizing.
    
    Automatically adjusts risk as account grows, protecting profits
    at each equity stage.
    """
    
    def __init__(
        self,
        initial_balance: float,
        growth_mode_enabled: bool = True,
        high_aggression_below: float = 30.0,
        equity_step_size: float = 50.0,
        equity_step_drawdown_limit: float = 0.15,
    ):
        """
        Initialize Risk Ladder.
        
        Args:
            initial_balance: Starting account balance
            growth_mode_enabled: Enable tiered risk scaling
            high_aggression_below: Balance threshold for high aggression (e.g., $30)
            equity_step_size: Equity milestone size (e.g., $50 steps)
            equity_step_drawdown_limit: Max drawdown from step high before reducing risk (15% = 0.15)
        """
        self.initial_balance = initial_balance
        self.growth_mode_enabled = growth_mode_enabled
        self.high_aggression_below = high_aggression_below
        self.equity_step_size = equity_step_size
        self.equity_step_drawdown_limit = equity_step_drawdown_limit
        
        # Risk tiers configuration (must be first)
        self.tiers = self._build_default_tiers()
        
        # Current state
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_tier = self._classify_tier(initial_balance)
        
        # Equity steps tracking
        self.equity_steps: List[EquityStep] = []
        self._create_initial_step()
        
        # Position size history
        self.position_history: List[Dict] = []
    
    def _build_default_tiers(self) -> List[RiskTierConfig]:
        """Build default risk tier configuration."""
        return [
            RiskTierConfig(
                balance_min=0,
                balance_max=self.high_aggression_below,
                risk_per_trade=20.0,  # 20% for ultra-aggressive
                tier=RiskTier.ULTRA_AGGRESSIVE,
                multiplier=2.0
            ),
            RiskTierConfig(
                balance_min=self.high_aggression_below,
                balance_max=100,
                risk_per_trade=10.0,  # 10% for very aggressive
                tier=RiskTier.VERY_AGGRESSIVE,
                multiplier=1.5
            ),
            RiskTierConfig(
                balance_min=100,
                balance_max=500,
                risk_per_trade=5.0,  # 5% for aggressive
                tier=RiskTier.AGGRESSIVE,
                multiplier=1.0
            ),
            RiskTierConfig(
                balance_min=500,
                balance_max=1000,
                risk_per_trade=3.0,  # 3% for moderate
                tier=RiskTier.MODERATE,
                multiplier=0.7
            ),
            RiskTierConfig(
                balance_min=1000,
                balance_max=float('inf'),
                risk_per_trade=1.0,  # 1% for conservative
                tier=RiskTier.CONSERVATIVE,
                multiplier=0.5
            ),
        ]
    
    def _create_initial_step(self) -> None:
        """Create the initial equity step."""
        step = EquityStep(
            step_balance=self.initial_balance,
            reached_at=datetime.now(),
            highest_equity_in_step=self.initial_balance,
            lowest_equity_in_step=self.initial_balance,
        )
        self.equity_steps.append(step)
    
    def _classify_tier(self, balance: float) -> RiskTierConfig:
        """Classify balance into a risk tier."""
        for tier_config in self.tiers:
            if tier_config.balance_min <= balance < tier_config.balance_max:
                return tier_config
        return self.tiers[-1]  # Default to most conservative
    
    def update_balance(self, new_balance: float) -> None:
        """
        Update current balance and track equity steps.
        
        Args:
            new_balance: New account balance
        """
        prev_balance = self.current_balance
        self.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # Update current tier
        prev_tier = self.current_tier
        self.current_tier = self._classify_tier(new_balance)
        
        if prev_tier.tier != self.current_tier.tier:
            logger.info(f"Risk tier changed: {prev_tier.tier.value} → {self.current_tier.tier.value}")
        
        # Track equity steps
        self._update_equity_steps(new_balance)
    
    def _update_equity_steps(self, current_balance: float) -> None:
        """Update equity step tracking."""
        current_step = self.equity_steps[-1]
        
        # Update step statistics
        if current_balance > current_step.highest_equity_in_step:
            current_step.highest_equity_in_step = current_balance
        if current_balance < current_step.lowest_equity_in_step:
            current_step.lowest_equity_in_step = current_balance
        
        # Check if new step reached (crossed multiple of equity_step_size)
        steps_reached = int(current_balance / self.equity_step_size)
        current_step_num = int(current_step.step_balance / self.equity_step_size)
        
        if steps_reached > current_step_num:
            # Create new equity step
            new_step = EquityStep(
                step_balance=current_balance,
                reached_at=datetime.now(),
                highest_equity_in_step=current_balance,
                lowest_equity_in_step=current_balance,
            )
            self.equity_steps.append(new_step)
            logger.info(f"✓ New equity step reached: ${current_balance:.2f}")
    
    def calculate_lot_size(
        self,
        equity: float,
        atr: float = None,
        pip_value: float = 1.0,
        base_risk_pct: float = None,
    ) -> float:
        """
        Calculate dynamic lot size based on equity and volatility.
        
        Uses two methods:
        1. Simple: position_size = base_risk * (equity / 100)
        2. ATR-based: position_size = (equity * risk_pct) / (atr * pip_value)
        
        Args:
            equity: Current account equity
            atr: Average True Range (optional, for volatility adjustment)
            pip_value: Value per pip (usually 10 for major pairs)
            base_risk_pct: Risk percentage override (uses tier if None)
            
        Returns:
            Lot size to trade
        """
        # Determine risk percentage
        if base_risk_pct is None:
            tier = self._classify_tier(equity)
            risk_pct = tier.risk_per_trade / 100.0
        else:
            risk_pct = base_risk_pct / 100.0
        
        if atr is None or atr == 0:
            # Simple equity-based sizing
            # Scales position size with equity
            lot_size = (risk_pct * equity) / 100.0
            return max(0.01, round(lot_size, 2))
        
        # ATR-based sizing (volatility-adjusted)
        risk_amount = equity * risk_pct
        lot_size = risk_amount / (atr * pip_value)
        
        return max(0.01, round(lot_size, 2))
    
    def check_equity_step_drawdown(self) -> Tuple[bool, float]:
        """
        Check if current equity has fallen too far from step high.
        
        Returns:
            Tuple of (is_safe, drawdown_percentage)
        """
        if not self.equity_steps:
            return True, 0.0
        
        current_step = self.equity_steps[-1]
        step_high = current_step.highest_equity_in_step
        current = self.current_balance
        
        if step_high == 0:
            return True, 0.0
        
        drawdown_pct = (step_high - current) / step_high
        is_safe = drawdown_pct <= self.equity_step_drawdown_limit
        
        return is_safe, drawdown_pct
    
    def should_reduce_risk(self) -> Tuple[bool, str]:
        """
        Determine if risk should be reduced based on equity step drawdown.
        
        Returns:
            Tuple of (should_reduce, reason)
        """
        is_safe, drawdown_pct = self.check_equity_step_drawdown()
        
        if not is_safe:
            reason = f"Equity step drawdown: {drawdown_pct*100:.1f}% (limit: {self.equity_step_drawdown_limit*100:.1f}%)"
            return True, reason
        
        return False, ""
    
    def get_adjusted_risk_pct(self) -> float:
        """
        Get risk percentage, potentially reduced if step drawdown limit breached.
        
        Returns:
            Adjusted risk percentage
        """
        tier = self._classify_tier(self.current_balance)
        base_risk = tier.risk_per_trade
        
        # Check if we should reduce risk due to step drawdown
        should_reduce, _ = self.should_reduce_risk()
        if should_reduce:
            # Cut risk in half when drawdown limit breached
            adjusted_risk = base_risk * 0.5
            logger.warning(f"Risk reduced: {base_risk}% → {adjusted_risk}% (step drawdown protection)")
            return adjusted_risk
        
        return base_risk
    
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pnl: float,
        atr: float = None,
    ) -> None:
        """
        Record a completed trade and update step statistics.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            lot_size: Lot size used
            pnl: Profit/loss
            atr: ATR at time of trade
        """
        if self.equity_steps:
            current_step = self.equity_steps[-1]
            current_step.trades_in_step += 1
            current_step.step_pnl += pnl
        
        self.position_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'lot_size': lot_size,
            'pnl': pnl,
            'atr': atr,
            'tier': self.current_tier.tier.value,
        })
        
        logger.debug(f"Trade recorded: {symbol} {lot_size}L PnL: {pnl:.2f}")
    
    def get_tier_summary(self) -> Dict:
        """Get current tier summary."""
        return {
            'tier': self.current_tier.tier.value,
            'balance_range': f"${self.current_tier.balance_min:.0f} - ${self.current_tier.balance_max:.0f}",
            'risk_per_trade': self.current_tier.risk_per_trade,
            'multiplier': self.current_tier.multiplier,
            'current_balance': self.current_balance,
        }
    
    def get_equity_step_summary(self) -> List[Dict]:
        """Get summary of all equity steps."""
        summary = []
        for i, step in enumerate(self.equity_steps, 1):
            step_range = step.highest_equity_in_step - step.lowest_equity_in_step
            summary.append({
                'step': i,
                'starting_balance': f"${step.step_balance:.2f}",
                'highest': f"${step.highest_equity_in_step:.2f}",
                'lowest': f"${step.lowest_equity_in_step:.2f}",
                'range': f"${step_range:.2f}",
                'trades': step.trades_in_step,
                'pnl': f"${step.step_pnl:.2f}",
                'reached_at': step.reached_at.isoformat(),
            })
        return summary
    
    def print_status(self) -> None:
        """Print current risk ladder status."""
        tier_info = self.get_tier_summary()
        is_safe, drawdown_pct = self.check_equity_step_drawdown()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RISK LADDER STATUS")
        logger.info(f"{'='*60}")
        logger.info(f"Current Balance: ${self.current_balance:.2f}")
        logger.info(f"Peak Balance: ${self.peak_balance:.2f}")
        logger.info(f"")
        logger.info(f"Current Tier: {tier_info['tier']}")
        logger.info(f"Balance Range: {tier_info['balance_range']}")
        logger.info(f"Risk Per Trade: {tier_info['risk_per_trade']}%")
        logger.info(f"Position Multiplier: {tier_info['multiplier']}x")
        logger.info(f"")
        logger.info(f"Equity Step Protection:")
        logger.info(f"  Step Drawdown: {drawdown_pct*100:.1f}% (limit: {self.equity_step_drawdown_limit*100:.1f}%)")
        logger.info(f"  Protected: {'✓ YES' if is_safe else '✗ NO - Risk reduced'}")
        logger.info(f"")
        
        if len(self.equity_steps) > 0:
            current_step = self.equity_steps[-1]
            logger.info(f"Current Step #{len(self.equity_steps)}:")
            logger.info(f"  Reached: {current_step.reached_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  Trades: {current_step.trades_in_step}")
            logger.info(f"  Step PnL: ${current_step.step_pnl:.2f}")
        
        logger.info(f"{'='*60}\n")


class PositionSizer:
    """
    Advanced position sizing calculator combining multiple factors.
    
    Considers:
    - Equity and account growth
    - Market volatility (ATR)
    - Risk per trade
    - Account tier
    - Equity step protection
    """
    
    def __init__(self, risk_ladder: RiskLadder, pip_value: float = 10.0):
        """
        Initialize position sizer.
        
        Args:
            risk_ladder: RiskLadder instance
            pip_value: Value per pip (default: 10 for major pairs)
        """
        self.risk_ladder = risk_ladder
        self.pip_value = pip_value
    
    def calculate(
        self,
        equity: float,
        atr: float = None,
        override_risk_pct: float = None,
    ) -> Dict:
        """
        Calculate comprehensive position sizing.
        
        Returns:
            Dictionary with position sizing details
        """
        # Get risk percentage (may be reduced by step protection)
        risk_pct = self.risk_ladder.get_adjusted_risk_pct()
        if override_risk_pct is not None:
            risk_pct = override_risk_pct
        
        # Calculate lot size
        lot_size = self.risk_ladder.calculate_lot_size(
            equity=equity,
            atr=atr,
            pip_value=self.pip_value,
            base_risk_pct=risk_pct
        )
        
        # Calculate risk amount
        risk_amount = (equity * risk_pct) / 100.0
        
        # Get tier info
        tier_info = self.risk_ladder.get_tier_summary()
        
        # Check equity step
        is_safe, drawdown_pct = self.risk_ladder.check_equity_step_drawdown()
        
        return {
            'lot_size': lot_size,
            'risk_pct': risk_pct,
            'risk_amount': risk_amount,
            'atr': atr,
            'tier': tier_info['tier'],
            'tier_multiplier': tier_info['multiplier'],
            'equity_step_safe': is_safe,
            'equity_step_drawdown': drawdown_pct,
            'tier_info': tier_info,
        }
    
    def print_calculation(self, sizing_result: Dict) -> None:
        """Print position sizing calculation details."""
        logger.info(f"\n{'='*60}")
        logger.info(f"POSITION SIZING")
        logger.info(f"{'='*60}")
        logger.info(f"Lot Size: {sizing_result['lot_size']}L")
        logger.info(f"Risk %: {sizing_result['risk_pct']:.1f}%")
        logger.info(f"Risk Amount: ${sizing_result['risk_amount']:.2f}")
        logger.info(f"ATR: {sizing_result['atr']}")
        logger.info(f"Tier: {sizing_result['tier']} (x{sizing_result['tier_multiplier']:.1f})")
        logger.info(f"Equity Step Safe: {'✓' if sizing_result['equity_step_safe'] else '✗'}")
        logger.info(f"{'='*60}\n")