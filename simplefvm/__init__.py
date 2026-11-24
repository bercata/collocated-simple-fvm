"""Collocated finite-volume SIMPLE solver (Rhie-Chow) for 2-D incompressible flow."""
from .energy import EnergySolver
from .fields import SimpleFields2D
from .linsys import BCManager, FVLinearSystem
from .mesh import formfunction, setUpMesh, unit_square
from .operators import FVM_div, FVM_grad, FVM_laplacian, cell_area, deferred_correction_source
from .simple import NaturalConvectionSolver, SimpleSolver
from .solvers import (TDMA, ConvergenceMonitor, MomentumEquationSolver, PressureVelocityCorrector,
                      RhieChowPressureCorrection, SparseDirect)
from .stencil import Coordicate2D, Stencil2D

__all__ = [
    "SimpleFields2D", "BCManager", "FVLinearSystem", "formfunction", "setUpMesh", "unit_square",
    "FVM_div", "FVM_grad", "FVM_laplacian", "cell_area", "deferred_correction_source",
    "SimpleSolver", "NaturalConvectionSolver", "TDMA",
    "ConvergenceMonitor", "MomentumEquationSolver", "PressureVelocityCorrector",
    "RhieChowPressureCorrection", "SparseDirect", "Coordicate2D", "Stencil2D",
    "EnergySolver",
]
__version__ = "1.0.0"
