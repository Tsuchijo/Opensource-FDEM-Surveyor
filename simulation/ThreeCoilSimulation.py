import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq


class Coil():
    """
    A class to represnt a magnetic induction coil. Stores properties such as orientation, position, and physical parameters. 
    """

    def __init__(self, orientation: str, position: tuple, radius: float, turns: int, resistivity: float = 1.68e-8, wire_diameter: float = 0.001, inductance: float = None, resistance: float = None):
        """
        Initialize the Coil with orientation, position, radius, number of turns, and optional resistivity.

        Parameters:
        orientation (str): Orientation of the coil ('x', 'y', or 'z').
        position (tuple): Position of the coil in 3D space (x, y, z).
        radius (float): Radius of the coil.
        turns (int): Number of turns in the coil.
        resistivity (float, optional): Resistivity of the coil material.
        """
        self.orientation = orientation
        self.position = np.array(position)
        self.radius = radius
        self.area = np.pi * radius**2
        self.turns = turns
        self.resistivity = resistivity
        self.wire_diameter = wire_diameter
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
        self.inductance = inductance if inductance is not None else self.calculate_inductance()
        self.resistance = resistance if resistance is not None else self.calculate_resistance()

        # Calculate normal vector based on orientation
        if orientation == 'x':
            self.normal = np.array([1, 0, 0])
        elif orientation == 'y':
            self.normal = np.array([0, 1, 0])
        elif orientation == 'z':
            self.normal = np.array([0, 0, 1])
        else:
            raise ValueError("Orientation must be 'x', 'y', or 'z'.")

    def calculate_inductance(self):
        """
        Calculate the inductance of the coil based on its physical parameters.
        The formula used is a simplified version for a circular coil.
        $$L = \mu_0 a [ln(\frac{8a}{R}) - 1.75]$$
        Returns:
        float: Inductance of the coil in Henrys.
        """
        # Simplified formula for inductance of a circular coil
        return self.mu_0 * self.radius * (np.log(8 * self.radius / (self.wire_diameter / 2)) - 1.75) * self.turns**2
    
    def calculate_resistance(self):
        """
        Calculate the resistance of the coil based on its physical parameters.

        Returns:
        float: Resistance of the coil in Ohms.
        """
        # Calculate the length of the wire
        length = 2 * np.pi * self.radius * self.turns
        # Calculate the cross-sectional area of the wire
        area = np.pi * (self.wire_diameter / 2)**2
        # Calculate resistance using resistivity
        return (self.resistivity * length) / area
    
    def transfer_function(self, freq_array):
        """
        Calculate the transfer function of the coil at a given frequency.

        Parameters:
        frequency (float): Frequency in Hz.

        Returns:
        complex: Transfer function of the coil.
        """
        omega = 2 * np.pi * freq_array
        impedance = complex(self.resistance, omega * self.inductance)
        return 1 / impedance
    
    def transfer_function_loaded(self, freq_array, self_capacitance, R_load, C_load=0):
        """
        Calculate V_meas / V_emf for a coil with self capacitance and measurement loading.

        Parameters:
        freq_array : array of frequencies (Hz)
        self_capacitance : coil self-capacitance (F)
        R_load : measurement device resistance (Ohm)
        C_load : measurement device input capacitance (F)

        Returns:
        complex array: transfer function over frequency
        """
        omega = 2 * np.pi * freq_array
        Z_L = 1j * omega * self.inductance
        Z_R = self.resistance
        Z_Cp = 1 / (1j * omega * self_capacitance)
        Z_Cl = np.inf if C_load == 0 else 1 / (1j * omega * C_load)

        # Parallel combination of coil capacitance, load R, and load C
        def parallel(*Zs):
            Y = 0
            for Z in Zs:
                Y += 1 / Z
            return 1 / Y

        Z_parallel = parallel(Z_Cp, R_load, Z_Cl)
        Z_series = Z_R + Z_L
        H = Z_parallel / (Z_series + Z_parallel)  # Divider ratio
        return H
    
    def b_field(self, point, current=1, resolution=100):
        """
        Calculate the magnetic field at a given point in space due to the coil.
        Uses the coil's normal vector to define the plane and Biot-Savart law for calculation.

        Parameters:
        point (tuple): Point in space (x, y, z) where the magnetic field is calculated.
        current (float): Current flowing through the coil in Amperes.
        resolution (int): number of current element segments to break the coil into.

        Returns:
        np.array: Magnetic field vector at the given point.
        """
        point = np.array(point)
        
        # Create two orthogonal vectors in the plane perpendicular to the normal
        # Find a vector that's not parallel to the normal
        if abs(self.normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create orthogonal basis vectors in the coil plane
        u = np.cross(self.normal, temp)
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(self.normal, u)
        v = v / np.linalg.norm(v)  # Normalize
        
        # Calculate the magnetic field using Biot-Savart law
        b_field = np.zeros(3)
        
        for i in range(resolution):
            # Position of current element on the coil
            theta = 2 * np.pi * i / resolution
            
            # Position of current element in the coil plane
            r_element = (self.position + 
                        self.radius * np.cos(theta) * u + 
                        self.radius * np.sin(theta) * v)
            
            # Current direction (tangent to the circle, perpendicular to radius)
            dl = (self.radius * (-np.sin(theta) * u + np.cos(theta) * v) * 
                  (2 * np.pi / resolution))
            
            # Vector from current element to field point
            r_vec = point - r_element
            r_distance = np.linalg.norm(r_vec)
            
            if r_distance > 1e-12:  # Avoid division by zero
                # Biot-Savart law: dB = (μ₀/4π) * I * (dl × r) / |r|³
                db = (self.mu_0 / (4 * np.pi)) * current * np.cross(dl, r_vec) / (r_distance**3)
                b_field += db * self.turns

        return b_field

    def integrate_flux(self, b_field, coordinates):
        """
        Integrate the magnetic flux through the coil given a magnetic field vector.

        Parameters:
        b_field (np.array): a Nx3 array representing a spacially varying magnetic field.
        coordinates (np.array): a Nx4 array representing the coordinates of the magnetic field points.

        Returns:
        float: Magnetic flux through the coil.
        """
        # Calculate the area vector of the coil
        area_vector = self.area * self.normal
        # Integrate the flux through the coil
        flux = 0
        for i in range(b_field.shape[0]):
            # check if the point is within the coil radius and plane
            if np.linalg.norm(coordinates[i, :3] - self.position) <= self.radius and np.isclose(coordinates[i, 3], self.position[2]):
                flux += np.dot(b_field[i], area_vector)
        return flux


class ThreeCoilSystem:
    def __init__(self, tx_coil, rx_coil, bucking_coil=None, spacing=0.01, resolution_bfield=100):
        self.tx_coil = tx_coil
        self.rx_coil = rx_coil
        self.bucking_coil = bucking_coil
        self.spacing = spacing
        self.resolution_bfield = resolution_bfield

        self._coords_rx = self._coil_plane_coordinates(self.rx_coil, spacing)

        self._M_tx_rx = None
        self._M_bk_rx = None

        # Track coil geometry to detect changes
        self._last_tx_state = None
        self._last_bk_state = None

    def _coil_state(self, coil):
        """Return tuple describing coil geometry for change detection."""
        return (tuple(coil.position), coil.radius, tuple(coil.normal), coil.inductance)

    @property
    def M_tx_rx(self):
        if self._M_tx_rx is None or self._coil_state(self.tx_coil) != self._last_tx_state:
            self._M_tx_rx = self._mutual_inductance(self.tx_coil, self.rx_coil,
                                                    self._coords_rx, self.resolution_bfield)
            self._last_tx_state = self._coil_state(self.tx_coil)
        return self._M_tx_rx

    @property
    def M_bk_rx(self):
        if self.bucking_coil is None:
            return 0.0
        if self._M_bk_rx is None or self._coil_state(self.bucking_coil) != self._last_bk_state:
            self._M_bk_rx = self._mutual_inductance(self.bucking_coil, self.rx_coil,
                                                    self._coords_rx, self.resolution_bfield)
            self._last_bk_state = self._coil_state(self.bucking_coil)
        return self._M_bk_rx

    @property
    def M_total(self):
        return self.M_tx_rx - self.M_bk_rx

    def _coil_plane_coordinates(self, coil, spacing):
        normal = coil.normal.astype(float)
        if abs(normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        coords_list = []
        r_vals = np.arange(-coil.radius, coil.radius + spacing, spacing)
        for ru in r_vals:
            for rv in r_vals:
                if np.sqrt(ru**2 + rv**2) <= coil.radius:
                    pt = coil.position + ru * u + rv * v
                    coords_list.append([pt[0], pt[1], pt[2], pt[2]])
        return np.array(coords_list)

    def _mutual_inductance(self, src_coil, dst_coil, coords_dst, resolution_bfield):
        I_test = 1.0
        b_field_points = []
        for coord in coords_dst:
            point = coord[:3]
            b = src_coil.b_field(point, current=I_test, resolution=resolution_bfield)
            b_field_points.append(b)
        b_field_points = np.array(b_field_points)
        flux = dst_coil.integrate_flux(b_field_points, coords_dst)
        return flux / I_test

    def find_bucking_distance(self, axis="x"):
        """
        Finds bucking coil position along 'axis' where M_total = 0.
        
        Parameters:
        axis (str): Axis to move along ('x', 'y', or 'z')
        bracket (tuple, optional): Search range. If None and constrain_between_coils=True,
                                 uses positions between TX and RX coils
        constrain_between_coils (bool): If True, constrains search between TX and RX coils
        
        Returns:
        float: Optimal position coordinate along the specified axis
        """
        if self.bucking_coil is None:
            raise ValueError("No bucking coil defined.")

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        
        tx_coord = self.tx_coil.position[axis_idx]
        rx_coord = self.rx_coil.position[axis_idx]
        min_coord = min(tx_coord, rx_coord)
        max_coord = max(tx_coord, rx_coord)
        # Add small margins to avoid edge effects
        left_margin = self.tx_coil.radius + self.bucking_coil.radius
        right_margin = self.rx_coil.radius + self.bucking_coil.radius
        bracket = (min_coord + left_margin, max_coord - right_margin)
        print(f"Constraining bucking coil search along {axis}-axis between {bracket[0]:.3f}m and {bracket[1]:.3f}m")

        
        # Store the original bucking coil position
        original_position = self.bucking_coil.position.copy()

        def M_diff(position_coord):
            """Function that returns M_total for a given position coordinate."""
            new_pos = original_position.copy()
            new_pos[axis_idx] = position_coord  # Set absolute position, not offset
            self.bucking_coil.position = new_pos
            self._M_bk_rx = None  # Force recalculation
            return self.M_total

        # Use Brent's method to find the root of M_total = 0
        print("Finding optimal bucking coil position...")
        print(f"Searching in range: {bracket[0]:.3f}m to {bracket[1]:.3f}m")
        print(f"Leftmost value: {M_diff(bracket[0]):.3e}, Rightmost value: {M_diff(bracket[1]):.3e}")

        optimal_position = brentq(M_diff, bracket[0], bracket[-1], xtol=1e-6, rtol=1e-6, maxiter=1000)
        # Set the bucking coil to the optimal position
        optimal_pos = original_position.copy()
        optimal_pos[axis_idx] = optimal_position
        self.bucking_coil.position = optimal_pos
        self._M_bk_rx = None  # Force recalculation
        return optimal_position


    def simulate_primary_response(self, frequencies, v_in,
                                   self_capacitance_rx=0,
                                   R_load=1e6, C_load=0, simulate_bucking=False):
        H_total = np.zeros(len(frequencies), dtype=complex)

        for i, f in enumerate(frequencies):
            omega = 2 * np.pi * f
            Z_tx = self.tx_coil.resistance + 1j * omega * self.tx_coil.inductance
            I_tx = v_in / Z_tx

            if simulate_bucking and self.bucking_coil is not None:
                V_emf_rx = 1j * omega * self.M_bk_rx * I_tx
            else:
                V_emf_rx = 1j * omega * self.M_tx_rx * I_tx
            H_rx_loaded = self.rx_coil.transfer_function_loaded(
                np.array([f]), self_capacitance_rx, R_load, C_load
            )[0]
            V_meas = V_emf_rx * H_rx_loaded
            H_total[i] = V_meas / v_in

        return H_total

    def plot_transfer_function(self, frequencies, v_in=1.0, 
                             self_capacitance_rx=0, R_load=1e6, C_load=0,
                             save_path=None, figsize=(10, 8), dpi=300):
        """
        Plot the transfer function of the three-coil system.
        
        Parameters:
        frequencies (array): Array of frequencies in Hz
        v_in (float): Input voltage amplitude in Volts
        self_capacitance_rx (float): RX coil self-capacitance in F
        R_load (float): Load resistance in Ohms
        C_load (float): Load capacitance in F
        save_path (str, optional): Path to save the plot. If None, plot is not saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution for saved image
        
        Returns:
        tuple: (figure, (ax1, ax2), transfer_function) - matplotlib objects and computed H
        """
        # Calculate transfer function
        H = self.simulate_primary_response(
            frequencies, v_in, self_capacitance_rx, R_load, C_load
        )
        
        # Calculate magnitude and phase
        magnitude_dB = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H) * 180 / np.pi
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Magnitude plot
        ax1.semilogx(frequencies, magnitude_dB, 'b-', linewidth=2, 
                     label=f'|H(f)|, M_total={self.M_total:.2e} H')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Three-Coil System Transfer Function')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(frequencies[0], frequencies[-1])
        ax1.legend()
        
        # Add annotations for key system parameters
        info_text = (f'TX-RX: {self.tx_coil.radius*100:.1f}cm → {self.rx_coil.radius*100:.1f}cm\n'
                    f'Separation: {np.linalg.norm(self.rx_coil.position - self.tx_coil.position)*100:.1f}cm')
        if self.bucking_coil is not None:
            info_text += f'\nBucking: {self.bucking_coil.radius*100:.1f}cm'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Phase plot
        ax2.semilogx(frequencies, phase_deg, 'r-', linewidth=2, label='∠H(f)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(frequencies[0], frequencies[-1])
        ax2.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Transfer function plot saved as '{save_path}'")
        
        # Print summary statistics
        peak_idx = np.argmax(np.abs(H))
        peak_freq = frequencies[peak_idx]
        peak_mag = magnitude_dB[peak_idx]
        
        print(f"\nTransfer Function Summary:")
        print(f"  Frequency range: {frequencies[0]:.0f} Hz to {frequencies[-1]:.0f} Hz")
        print(f"  Peak response: {peak_mag:.1f} dB at {peak_freq:.0f} Hz")
        print(f"  Total mutual inductance: {self.M_total:.6e} H")
        if self.bucking_coil is not None:
            print(f"  Bucking efficiency: {(self.M_bk_rx/self.M_tx_rx)*100:.1f}%")
        
        return fig, (ax1, ax2), H


def test_three_coil_system():
    """
    Test function to demonstrate the three-coil system optimization and 
    transfer function simulation with plotting.
    """
    print("Setting up three-coil system...")
    
    # Define coil parameters
    tx_coil = Coil(
        orientation='z',
        position=(0, 0, 0),
        radius=0.125,      # 12.5 cm radius
        turns=40,
        wire_diameter=0.001,
        resistivity=1.68e-8
    )
    
    rx_coil = Coil(
        orientation='z',
        position=(1.0, 0, 0),  # 30 cm separation
        radius=0.05,           # 5 cm radius
        turns=50,
        wire_diameter=0.0008,
        resistivity=1.68e-8
    )
    
    bucking_coil = Coil(
        orientation='z',
        position=(0.5, 0, 0),  # Initial position (will be optimized)
        radius=0.05,            # 5 cm radius
        turns=5,
        wire_diameter=0.0006,
        resistivity=1.68e-8
    )
    
    # Create the three-coil system
    system = ThreeCoilSystem(tx_coil, rx_coil, bucking_coil, spacing=0.005)
    
    print(f"Initial mutual inductances:")
    print(f"  M_tx_rx: {system.M_tx_rx:.6e} H")
    print(f"  M_bk_rx: {system.M_bk_rx:.6e} H")
    print(f"  M_total: {system.M_total:.6e} H")

    # Optimize bucking coil position
    print("\nOptimizing bucking coil position along X-axis between TX and RX...")

    optimal_x_position = system.find_bucking_distance(
        axis='x'
    )
    print(f"Optimal bucking coil X position: {optimal_x_position:.4f} m")
    print(f"Final bucking coil position: {bucking_coil.position}")
    print(f"Optimized M_total: {system.M_total:.6e} H")
    
    # Calculate the relative position
    tx_x = tx_coil.position[0]
    rx_x = rx_coil.position[0]
    relative_pos = (optimal_x_position - tx_x) / (rx_x - tx_x)
    print(f"Bucking coil is {relative_pos*100:.1f}% of the way from TX to RX")
            

    # Define frequency range for simulation
    frequencies = np.logspace(1, 6.5, 100)  # 100 Hz to 1 MHz
    v_in = 1.0  # 1V input
    
    # Simulate and plot transfer function
    print("\nSimulating and plotting transfer function...")
    fig, axes, H = system.plot_transfer_function(
        frequencies, 
        v_in=v_in,
        self_capacitance_rx=10e-12,  # 10 pF self capacitance
        R_load=1e6,                  # 1 MΩ load resistance
        C_load=50e-12,               # 50 pF load capacitance
        save_path='/home/pidud/FDEM-Scanner-2/three_coil_transfer_function.png'
    )
        
    plt.show()
    
    return system, frequencies, H


if __name__ == "__main__":
    # Run the test when the script is executed directly
    system, freqs, transfer_function = test_three_coil_system()