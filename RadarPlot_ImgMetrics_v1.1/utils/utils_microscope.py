import utils_bone
from dataclasses import dataclass

@dataclass
class STEMInputs:
    # Sample params
    matrix: str = "water"
    obj: str = "bone"
    t: float = 1.0e-6           # matrix thickness [m]
    t_SiN: float = 50e-9        # SiN window thickness [m]
    z_rel: float = 1.0          # relative position (0..1) from TOP window (STEM SI)
    x: float = 3.0e-6           # lamella thickness

    # Microscope settings
    U_eV: float = 200_000.0     # accelerating voltage [eV]
    Cs_m: float = 1e-3          # spherical aberration [m]
    alpha_p: float = 8e-3       # probe semi-convergence [rad]
    beta_rad: float = 0.02      # detector opening semi-angle β [rad] (set 0.002–0.03 for BF)
    eD_m2: float = 100 * 1e20   # dose eD [e-/m^2]; 100 e-/Å^2 -> 100*1e20

    # Detector split
    beta_BF_DF_threshold: float = 0.030  # 30 mrad

    # Fundamental constants (use same symbols as SI)
    epsi0: float = 8.854188e-12
    h: float = 6.626e-34
    m0: float = 9.109e-31
    c: float = 2.998e8
    e: float = 1.602e-19
    NA: float = 6.022e23
    aH: float = 5.292e-11
    SNRf: float = 3.0

    # Material properties

    # defaults for gold
    rho_o: float = 19.3e6
    Z_o: float = 79.0
    W_o: float = 197.0

    # defaults for bone
    rho_b: float = 2e6  #g/m^3
    Z_b: float = 11.72
    W_b: float = 23.6  #g/mol

    # defaults for matrix (water)
    rho_m: float = 997000.0
    Z_m: float = 4.7
    W_m: float = 6.0

    # defaults for SiN
    rho_SiN: float = 3.2e6
    Z_SiN: float = 10.6
    W_SiN: float = 20.0