import numpy as np
from . import ThreeCoilSimulation  # Adjusted import to ensure it's from the current package
from .utils import BaseCoil


class EddyCurrentRing(BaseCoil):
    mu0 = 4*np.pi*1e-7

    def __init__(self, sigma, mu_r, radius, width, thickness, position, orientation, L):
        self.sigma = sigma
        self.mu_r = mu_r
        self.radius = radius
        self.width = width
        self.thickness = thickness
        self.position = np.array(position)
        self.orientation = np.array(orientation) / np.linalg.norm(orientation)
        self.area = np.pi * radius**2
        self.turns = 1
        self.L = L  # inductance 

    def impedance(self, freqs):
        f = np.atleast_1d(freqs).astype(float)
        omega = 2*np.pi*f
        mu = self.mu_r * self.mu0
        r, w, t = self.radius, self.width, self.thickness
        path = 2*np.pi*r

        delta = np.sqrt(2.0/(np.maximum(omega,1e-30)*mu*self.sigma))
        t_eff = np.minimum(t, delta)

        R = (1/self.sigma) * (path/(w*t_eff))

        return R + 1j*omega*self.L

    def mutual_inductance(self, other: ThreeCoilSimulation.WoundCoil):
        return other.b_field(self.position, current=1, use_dipole_approx=True) @ \
               (self.orientation * np.pi*self.radius**2)


class FEMPipe:
    def __init__(self, sigma, mu_r, radius, length, n_rings, position, orientation):
        """
        sigma: conductivity [S/m]
        mu_r: relative permeability
        radius: pipe radius [m]
        width: axial width of each ring [m]
        thickness: wall thickness [m]
        length: pipe length [m]
        n_rings: number of discretized rings
        position: 3-vector center of pipe [m]
        orientation: 3-vector pipe axis direction
        """
        self.sigma = sigma
        self.mu_r = mu_r
        self.radius = radius
        self.thickness = radius
        self.length = length
        self.n_rings = n_rings
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.orientation /= np.linalg.norm(self.orientation)
        self.width = length / n_rings
        self.rings = []
        self._build_rings()

    def _build_rings(self):
        """Discretize the pipe into N rings along orientation axis."""
        # distribute centers evenly along pipe axis
        offsets = np.linspace(-self.length/2, self.length/2, self.n_rings)
        # Use solenoid formula for ring inductance
        mu = self.mu_r * np.pi * 4e-7
        L = mu * self.n_rings * self.radius**2 * np.pi / (self.length)
        self.rings = [
            EddyCurrentRing(
                sigma=self.sigma,
                mu_r=self.mu_r,
                radius=self.radius,
                width=self.width,
                thickness=self.thickness,
                position=self.position + offset * self.orientation,
                orientation=self.orientation,
                L=L,
            )
            for offset in offsets
        ]

    def get_rings(self):
        return self.rings




class PipeSimulation:
    def __init__(self, scanner, pipes):
        """
        scanner: ThreeCoilSystem object
        pipes: FEMPipe object or list of FEMPipe objects
        """
        self.three_coil = scanner
        self.fem_pipes = pipes if isinstance(pipes, list) else [pipes]

    def transfer_function(self, freqs):
        freqs = np.atleast_1d(freqs)
        H = np.zeros_like(freqs, dtype=complex)

        # free-space mutual between Tx and Rx, computer using full Biot-Savart integration
        M_tx_rx = self.three_coil._M_tx_rx

        omega = 2 * np.pi * freqs

        # direct Tx->Rx voltage
        V_direct = 1j * omega * M_tx_rx

        # sum contributions from eddy rings
        H = 0
        for fem_pipe in self.fem_pipes:
            
            rings = fem_pipe.get_rings()
            Z_rings = np.array([ring.impedance(freqs) for ring in rings])  # complex impedance of rings
            M_tx_rings = np.array([self.three_coil.tx_coil.mutual_inductance(ring) for ring in rings])
            M_ring_rx = np.array([ring.mutual_inductance(self.three_coil.rx_coil) for ring in rings])

            M_tx_rings = M_tx_rings[:, np.newaxis]
            M_ring_rx = M_ring_rx[:, np.newaxis]


            I_rings = (-1j * omega[np.newaxis, :] * M_tx_rings) / Z_rings  # induced currents for 1 A Tx
            V_rings = (1j * omega[np.newaxis, :] * M_ring_rx * I_rings).sum(axis=0)

            H += V_rings / V_direct

        return H
    
    def simulate(self, position, freqs):
        """
        Simulate the electromagnetic response at a specific surveyor position.
        
        Args:
            position: 3-element array/tuple (x, y, z) for surveyor position
            freqs: frequency or array of frequencies
            R_load: load resistance (uses system default if None)
            C_load: load capacitance (uses system default if None)
            hs_hp: if True, return HS/HP response, otherwise return transfer function
        
        Returns:
            Complex response array
        """
        position = np.array(position)
                
        # Move system to new position
        self.three_coil.move_system_to(position)
        
        try:
            return self.transfer_function(freqs)
        finally:
            # Restore original positions
            self.three_coil.reset_position()
    
    def grid_survey(self, x_range, y_range, z_height, frequencies):
        """
        Perform a grid survey over the specified area.
        
        Args:
            x_range: tuple (x_min, x_max, num_points) for x-axis grid
            y_range: tuple (y_min, y_max, num_points) for y-axis grid  
            z_height: height above the target to perform the survey
            frequencies: array of frequencies to simulate
        
        Returns:
            tuple: (x_grid, y_grid, real_response, imag_response)
        """
        x_min, x_max, nx = x_range
        y_min, y_max, ny = y_range
        
        x_positions = np.linspace(x_min, x_max, nx)
        y_positions = np.linspace(y_min, y_max, ny)
        
        x_grid, y_grid = np.meshgrid(x_positions, y_positions)
        
        real_response = np.zeros((ny, nx))
        imag_response = np.zeros((ny, nx))
        
        print(f"Running grid search over {nx}x{ny} = {nx*ny} points...")
        
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                position = np.array([x, y, z_height])
                response = self.simulate(position, frequencies)
                
                # Take response at middle frequency for visualization
                mid_freq_idx = len(frequencies) // 2
                complex_resp = response[mid_freq_idx]
                
                real_response[i, j] = np.real(complex_resp)
                imag_response[i, j] = np.imag(complex_resp)
                
            if (i + 1) % 5 == 0:
                print(f"Completed {i+1}/{ny} rows")
        
        return x_grid, y_grid, real_response, imag_response
    
    def line_survey(self, start_point, end_point, num_points, z_height, frequencies):
        """
        Perform a line survey between two points.
        
        Args:
            start_point: (x, y) coordinates of transect start
            end_point: (x, y) coordinates of transect end
            num_points: number of measurement points along the transect
            z_height: height above the target to perform the survey
            frequencies: array of frequencies to simulate
        
        Returns:
            tuple: (distances, real_response, imag_response)
        """
        # Create points along the transect
        x_points = np.linspace(start_point[0], end_point[0], num_points)
        y_points = np.linspace(start_point[1], end_point[1], num_points)
        
        # Calculate distances from start point for x-axis
        distances = np.sqrt((x_points - start_point[0])**2 + (y_points - start_point[1])**2)
        
        real_response = np.zeros(num_points)
        imag_response = np.zeros(num_points)
        
        print(f"Running line survey over {num_points} points...")
        
        for i in range(num_points):
            position = np.array([x_points[i], y_points[i], z_height])
            response = self.simulate(position, frequencies)
            
            # Take response at middle frequency for visualization
            mid_freq_idx = len(frequencies) // 2
            complex_resp = response[mid_freq_idx]
            
            real_response[i] = np.real(complex_resp)
            imag_response[i] = np.imag(complex_resp)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{num_points} points")
        
        return distances, real_response, imag_response
