import numpy as np

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