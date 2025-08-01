"""
Enhanced Spectral Slope Adaptive Filter (SSAF) Strategy

This package provides advanced enhancements to the SSAF strategy including:
- Advanced spectral analysis
- Machine learning filters
- Regime detection
- Multi-asset support
- Performance optimization
"""

from .advanced_indicators import (
    AdvancedSSAFIndicator,
    RegimeAdaptiveSSAF,
    MultiAssetSSAF,
    MLSSAFIndicator
)

__version__ = "2.0.0"
__author__ = "SSAF Enhanced Team"

__all__ = [
    'AdvancedSSAFIndicator',
    'RegimeAdaptiveSSAF',
    'MultiAssetSSAF',
    'MLSSAFIndicator'
]
