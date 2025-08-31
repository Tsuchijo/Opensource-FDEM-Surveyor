import numpy as np
import pytest
from simulation.ThreeCoilSimulation import WoundCoil

def dipole_field(m, r):
    """Theoretical dipole field at position r from dipole moment m."""
    mu0 = 4 * np.pi * 1e-7
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    factor = mu0 / (4 * np.pi * r_norm**3)
    return factor * (3 * np.dot(m, r_hat) * r_hat - m)

def test_biot_savart_vs_dipole():
    # Setup coil parameters
    coil = WoundCoil(orientation='z', position=(0, 0, 0), radius=0.05, turns=100)
    current = 1.0
    
    # Calculate magnetic moment: m = current * area * normal_vector * turns
    area = np.pi * coil.radius**2
    m = current * area * coil.normal * coil.turns
    
    # Test at increasing distances
    for dist in [0.5, 1.0, 2.0, 5.0]:
        r = np.array([0, 0, dist])
        B_sim = coil.b_field(r, current=current)
        B_dipole = dipole_field(m, r)
        # At large distances, the relative error should be small
        rel_error = np.linalg.norm(B_sim - B_dipole) / np.linalg.norm(B_dipole)
        assert rel_error < 0.05  # 5% error tolerance


def test_mutual_inductance_coplanar_coils():
    """
    Test that the magnetic flux from one coil through another coil in the same plane
    matches the dipole approximation when the coils are sufficiently separated.
    """
    # Create two coils in the same plane (z-oriented) but at different positions
    current = 1.0
    
    # Transmitter coil at origin
    tx_coil = WoundCoil(orientation='z', position=(0, 0, 0), radius=0.02, turns=50)
    
    # Test at various separations - all coils in the same xy-plane
    separations = [0.1, 0.2, 0.3, 0.5]  # distances in meters
    
    for separation in separations:
        # Receiver coil displaced along x-axis
        rx_coil = WoundCoil(orientation='z', position=(separation, 0, 0), radius=0.01, turns=25)
        
        # Calculate magnetic moment of transmitter coil
        tx_area = np.pi * tx_coil.radius**2
        tx_moment = current * tx_area * tx_coil.normal * tx_coil.turns
        
        # Calculate theoretical dipole field at receiver position
        r_vec = np.array(rx_coil.position) - np.array(tx_coil.position)
        B_dipole_at_rx = dipole_field(tx_moment, r_vec)
        
        # Calculate theoretical flux through receiver coil using dipole approximation
        # Φ = B · A = B_z * π * r² (since both coils are z-oriented)
        rx_area = np.pi * rx_coil.radius**2
        theoretical_flux = B_dipole_at_rx[2] * rx_area * rx_coil.turns
        
        # Calculate actual flux by integrating the simulated field
        # Sample points across the receiver coil area
        n_samples = 20
        simulated_flux = 0.0
        
        # Create sampling points within the receiver coil
        for i in range(n_samples):
            for j in range(n_samples):
                # Sample points in a grid within the coil radius
                x_offset = (i - n_samples/2) * (2 * rx_coil.radius / n_samples)
                y_offset = (j - n_samples/2) * (2 * rx_coil.radius / n_samples)
                
                # Only include points within the circular coil area
                if x_offset**2 + y_offset**2 <= rx_coil.radius**2:
                    sample_point = np.array([
                        rx_coil.position[0] + x_offset,
                        rx_coil.position[1] + y_offset,
                        rx_coil.position[2]
                    ])
                    
                    # Calculate field from transmitter at this point
                    B_sim = tx_coil.b_field(sample_point, current=current)
                    
                    # Add contribution to flux (B·n * dA)
                    # For z-oriented coils, we want the z-component
                    area_element = (2 * rx_coil.radius / n_samples)**2
                    simulated_flux += B_sim[2] * area_element * rx_coil.turns
        
        # Compare simulated flux with theoretical dipole approximation
        rel_error = abs(simulated_flux - theoretical_flux) / abs(theoretical_flux)
        
        print(f"Separation: {separation:.1f}m")
        print(f"  Theoretical flux: {theoretical_flux:.6e} Wb")
        print(f"  Simulated flux:   {simulated_flux:.6e} Wb")
        print(f"  Relative error:   {rel_error:.3f}")
        
        # At larger separations, the dipole approximation should be more accurate
        # Allow larger errors for closer separations due to near-field effects
        if separation >= 0.2:
            assert rel_error < 0.15, f"Flux error too large at {separation}m separation: {rel_error:.3f}"
        else:
            assert rel_error < 0.3, f"Flux error too large at {separation}m separation: {rel_error:.3f}"
    

def test_wire_gauge():
    """
    Test the wire gauge to metric diameter conversion.
    """
    from simulation.utils import wire_gauge_to_metric
    
    # Define a set of wire gauges and their expected diameters in mm
    gauge_diameter_pairs = {
        0: 8.251,
        1: 7.348,
        2: 6.544,
        3: 5.827,
        4: 5.189,
        5: 4.621,
        6: 4.115,
        7: 3.665,
        8: 3.264,
        9: 2.906,
        10: 2.588,
        # Add more if needed
    }
    
    for gauge, expected_diameter in gauge_diameter_pairs.items():
        calculated_diameter = wire_gauge_to_metric(gauge) * 1000  # Convert to mm
        assert np.isclose(calculated_diameter, expected_diameter, rtol=1e-3), \
            f"Gauge {gauge}: expected {expected_diameter}, got {calculated_diameter}"