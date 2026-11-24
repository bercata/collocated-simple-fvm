"""Fluid properties for low-Mach natural convection (Le Quere et al.).

Constants for air (T1/T2/T3 benchmark).  Sutherland law for viscosity.
"""
import numpy as np

# Reference (all SI)
P0  = 101325.0
T0  = 600.0
R   = 287.0
gamma = 1.4
cp  = gamma * R / (gamma - 1.0)   # 1004.5
Pr  = 0.71
g   = 9.81

# Sutherland
T_star = 273.0
S_const = 110.5
mu_star = 1.68e-5


def mu_sutherland(T):
    """Viscosity of air (kg/(m·s))."""
    return mu_star * (T / T_star) ** 1.5 * (T_star + S_const) / (T + S_const)


def k_thermal(T, mu_val=None):
    """Thermal conductivity (W/(m·K))."""
    if mu_val is None:
        mu_val = mu_sutherland(T)
    return mu_val * cp / Pr


def mu_constant(T=None):
    """Constant-property viscosity (T1)."""
    return mu_star


def rho_ideal(T, P):
    """Ideal-gas density (kg/m³)."""
    return P / (R * T)


def cavity_length(Ra, eps, mu0):
    """Cavity side length L (m) for given Ra, eps, mu0.
    rho0 = P0/(R*T0), Th = T0*(1+eps), Tc = T0*(1-eps).
    """
    rho0 = P0 / (R * T0)
    Th = T0 * (1.0 + eps)
    Tc = T0 * (1.0 - eps)
    L3 = Ra * T0 * mu0 ** 2 / (Pr * g * rho0 ** 2 * (Th - Tc))
    return L3 ** (1.0 / 3.0)


# Print benchmark values (run once at import for documentation)
if __name__ == "__main__":
    print("Air properties:")
    print(f"  P0={P0} Pa, T0={T0} K, R={R}, gamma={gamma}, cp={cp:.1f}")
    print(f"  Pr={Pr}, g={g}")
    print(f"  mu*(T0=600)={mu_star} (constant for T1)")
    print(f"  k*(T0=600)={mu_star*cp/Pr:.4f}")
    L_t1 = cavity_length(1e6, 0.6, mu_star)
    print(f"  T1 cavity length (Ra=1e6, eps=0.6, mu*): L={L_t1:.5f} m")
    L_t2 = cavity_length(1e6, 0.6, mu_sutherland(T0))
    print(f"  T2 cavity length (Ra=1e6, eps=0.6, Sutherland): L={L_t2:.5f} m")