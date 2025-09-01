import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

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

def plot_survey_heatmaps(x_grid, y_grid, real_response, imag_response, pipe_positions, figsize=(15, 6)):
    """
    Plot real and imaginary response as heat maps.
    
    Args:
        x_grid, y_grid: coordinate grids
        real_response, imag_response: response matrices
        pipe_positions: positions of the pipe for reference
        figsize: figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Real part heat map
    im1 = ax1.contourf(x_grid, y_grid, real_response, levels=20, cmap='RdBu_r')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Real Part Response')
    ax1.set_aspect('equal')
    
    # Mark pipe position
    ax1.plot(pipe_positions[:, 0], pipe_positions[:, 1], 'ko', markersize=10, markerfacecolor='yellow', 
             markeredgecolor='black', markeredgewidth=2, label='Pipe Center')
    ax1.legend()
    
    plt.colorbar(im1, ax=ax1, label='Real Response')
    
    # Imaginary part heat map
    im2 = ax2.contourf(x_grid, y_grid, imag_response, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Imaginary Part Response')
    ax2.set_aspect('equal')
    
    # Mark pipe position
    ax2.plot(pipe_positions[:, 0], pipe_positions[:, 1], 'ko', markersize=10, markerfacecolor='yellow',
             markeredgecolor='black', markeredgewidth=2, label='Pipe Center')
    ax2.legend()
    
    plt.colorbar(im2, ax=ax2, label='Imaginary Response')
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)

def plot_hshp(frequencies, hshp, figsize=(8, 5)):
    """
    Plot the secondary/primary field ratio (HSHP) vs frequency.
    Uses two plots for real and imaginary parts.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.semilogx(frequencies, np.real(hshp), 'b.-')
    ax1.set_ylabel('Real HSHP')
    ax1.grid(True)
    ax1.set_title('Secondary/Primary Field Ratio (HSHP)')

    ax2.semilogx(frequencies, np.imag(hshp), 'r.-')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Imaginary HSHP')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    


class BaseCoil(ABC):
    @abstractmethod
    def impedance(self, freqs):
        """Return Z(f) array in ohms."""
        pass

    @abstractmethod
    def mutual_inductance(self, other):
        """Return mutual inductance with another coil (H)."""
        pass
