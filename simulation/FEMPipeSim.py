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
    def __init__(self, three_coil, fem_pipes):
        """
        three_coil: ThreeCoilSystem object
        fem_pipe: FEMPipe object
        """
        self.three_coil = three_coil
        self.fem_pipes = fem_pipes if isinstance(fem_pipes, list) else [fem_pipes]

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
        return self.transfer_function(freqs)
