import numpy as np
from abc import ABC, abstractmethod

def wire_gauge_to_metric(gauge):
    """
    Convert wire gauge to metric diameter in millimeters.
    
    Args:
        gauge (int): Wire gauge number.
    
    Returns:
        float: Diameter in meters.
    """
    if gauge < 0:
        raise ValueError("Gauge must be a non-negative integer.")
    
    # Standard wire gauge conversion table
    # Source: https://en.wikipedia.org/wiki/American_wire_gauge
    diameter_m = 0.127 * (92 ** ((36 - gauge) / 39)) * 1e-3
    
    return diameter_m


class BaseCoil(ABC):
    @abstractmethod
    def impedance(self, freqs):
        """Return Z(f) array in ohms."""
        pass

    @abstractmethod
    def mutual_inductance(self, other):
        """Return mutual inductance with another coil (H)."""
        pass
