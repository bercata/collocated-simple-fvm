"""Collocated finite-volume SIMPLE solver (Rhie-Chow) for 2-D incompressible flow."""
from .fields import SimpleFields2D
from .linsys import BCManager, FVLinearSystem
from .mesh import formfunction, setUpMesh, unit_square
from .operators import FVM_div, FVM_grad, FVM_laplacian, cell_area
from .simple import SimpleSolver
from .solvers import (TDMA, ConvergenceMonitor, MomentumEquationSolver, PressureVelocityCorrector,
                      RhieChowPressureCorrection, SparseDirect)
from .stencil import Coordicate2D, Stencil2D

__all__ = [
    "SimpleFields2D", "BCManager", "FVLinearSystem", "formfunction", "setUpMesh", "unit_square",
    "FVM_div", "FVM_grad", "FVM_laplacian", "cell_area", "SimpleSolver", "TDMA",
    "ConvergenceMonitor", "MomentumEquationSolver", "PressureVelocityCorrector",
    "RhieChowPressureCorrection", "SparseDirect", "Coordicate2D", "Stencil2D",
]
__version__ = "1.0.0"
