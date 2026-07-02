import numpy as np
import resolution_theory as thc
from utils_microscope import STEMInputs
inp = STEMInputs()

def get_properties():
        # Atomic numbers (Z)
        Z_values = np.array([8, 20, 6, 15, 7, 11, 12])  # O, Ca, C, P, N, Na, Mg

        # Atomic weights (W, g/mol)
        W_values_g = np.array([16.0, 40.0, 12.0, 31.0, 14.0, 23.0, 24.3])

        # Atomic % composition from atom probe tomography
        fractions = np.array([37.8, 26.4, 18.0, 12.3, 2.8, 2.55, 0.15])
        fractions /= fractions.sum()  # normalize

        # Effective properties
        W_bone_gmol = np.average(W_values_g, weights=fractions)  # g/mol
        W_bone = W_bone_gmol * 1e-3  # kg/mol
        Z_bone = np.average(Z_values, weights=fractions)

        # Density of bone (literature ~2.0 g/cm^3)
        rho_bone = 2000.0  # kg/m^3
        rho_bone_gm3 = rho_bone * 1e3  # g/cm^3
        return W_bone_gmol, Z_bone, rho_bone_gm3


