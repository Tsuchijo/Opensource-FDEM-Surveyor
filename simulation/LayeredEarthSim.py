import numpy as np
from . import ThreeCoilSimulation  # Adjusted import to ensure it's from the current package
import simpeg.electromagnetics.frequency_domain as fdem
import simpeg.maps as maps

class LayeredEarthSim:

    def __init__(self, layer_thicknesses, layer_conductivities, scanner):
        """
        layers: list of dicts with keys 'thickness', 'sigma', 'mu_r'
                thickness in meters (last layer can have thickness=np.inf)
                sigma in S/m
                mu_r relative permeability
        """
        self.layer_thicknesses = layer_thicknesses
        self.layer_conductivities = layer_conductivities

        self.source_location = scanner.tx_coil.position

        # Calculate source orientation by finding which basis vector, X, Y or Z is most aligned with the coil normal
        max_index = np.argmax(np.abs(scanner.tx_coil.normal))
        if max_index == 0:
            self.source_orientation = 'x'
        elif max_index == 1:
            self.source_orientation = 'y'
        else:
            self.source_orientation = 'z'

        self.receiver_location = scanner.rx_coil.position
        max_index_rx = np.argmax(np.abs(scanner.rx_coil.normal))
        if max_index_rx == 0:
            self.receiver_orientation = 'x'
        elif max_index_rx == 1:
            self.receiver_orientation = 'y'
        else:
            self.receiver_orientation = 'z'

    def build_survey(self, frequencies):
        source_list = []  # create empty list for source objects
        data_type = "ppm"

        # loop over all sources
        for freq in frequencies:
            # Define receivers that measure real and imaginary component
            # magnetic field data in ppm.
            receiver_list = []
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    self.receiver_location,
                    orientation=self.receiver_orientation,
                    data_type=data_type,
                    component="real",
                )
            )
            receiver_list.append(
                fdem.receivers.PointMagneticFieldSecondary(
                    self.receiver_location,
                    orientation=self.receiver_orientation,
                    data_type=data_type,
                    component="imag",
                )
            )

            # Define a magnetic dipole source at each frequency
            source_list.append(
                fdem.sources.MagDipole(
                    receiver_list=receiver_list,
                    frequency=freq,
                    location=self.source_location,
                    orientation=self.source_orientation,
                    moment=1.0,
                )
            )

        survey = fdem.survey.Survey(source_list)
        return survey

    def simulate(self, frequencies):
        survey = self.build_survey(frequencies)
        conductivity_map = maps.IdentityMap()

        simulation_conductivity = fdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.layer_thicknesses,
            sigmaMap=conductivity_map,
        )

        earth_response = simulation_conductivity.dpred(self.layer_conductivities)
    
        real = earth_response[::2]
        imag = earth_response[1::2]

        return (real + 1j * imag) * 1e-6 # Convert from ppm to absolute

        
    
