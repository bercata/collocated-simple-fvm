# Changes from the first version

The first version of the solver was a single notebook. It is kept unchanged in
[`original/`](../original). The class layout and the SIMPLE sequence are the same, so code
written against the first version still reads the same way:

```
MomentumEquationSolver.solve(u) -> solve(v) -> RhieChowPressureCorrection.solve
  -> PressureVelocityCorrector.apply -> convergence check
```

## Corrections

| # | Where | First version | Now |
|---|---|---|---|
| 1 | `RhieChowPressureCorrection`, north faces | The direct pressure difference was taken in **x** (`P[i+1,j+1] - P[i,j+1]`) | `p_N - p_P` across the face, in y |
| 2 | `RhieChowPressureCorrection`, wall faces | Wall coefficients were not zeroed and the ghost p' stayed 0, so p' = 0 was imposed on every wall | Wall coefficients are zero (zero-gradient p'); one cell is pinned to remove the constant null space |
| 3 | Wall mass fluxes | `Con_we[0, 1:M]`, `Con_ns[1:M, 0]` missed the first row/column, so a small flux crossed the wall there | All wall faces carry exactly zero mass flux |
| 4 | South-face area in the p' matrix | Reused the variable `S_n` from the previous loop | Each face uses its own geometry (no effect on a uniform mesh, wrong on a stretched or curved one) |
| 5 | Convective fluxes in the momentum equation | Recomputed from averaged cell velocities, which do not satisfy continuity per cell | The Rhie-Chow face mass fluxes, corrected with p', are stored and reused, so every cell is conservative. Transported scalars (for example a flamelet mixture fraction) need this to stay bounded |
| 6 | Velocity correction at walls | Face p' at a wall was the average of p' and a ghost value of 0 | Ghost p' copies the adjacent cell (consistent with the zero-gradient condition) |
| 7 | `BCManager._outward_dirs` | Mapped `i = 0` to "north" although `i` is the x index | West/east/south/north follow the package convention |
| 8 | Default `alpha_p` in the run cell | 0.8 with `alpha_u` = 0.7 | `alpha_p = 1 - alpha_u = 0.3`, the usual SIMPLE pairing |
| 9 | Run cell | 5 outer iterations, never converged, Re = 1, 2:1 cavity | Runs to a scaled residual of 1e-6 at Re = 100, 400, 1000 on a square cavity |

## Generalisations (identical on a rectangular mesh)

- Face fluxes use the face normal vector `(Sx, Sy)` instead of only `u` on east faces and `v` on
  north faces. On the curved-lid meshes the north faces are inclined, so the old form lost the
  `u * Sx` part of the flux.
- The gradient uses the Gauss theorem over the four faces. On a uniform rectangle it is the same
  central difference as before.
- The moving lid follows the local wall tangent (for a flat lid: `u = U_lid`, `v = 0` as before).

## Performance

- All face lengths, normals, centre distances and volumes are computed once in `Stencil2D`
  instead of inside every loop of every operator on every iteration.
- All operators are NumPy array expressions; there are no per-cell Python loops left in the hot path.
- `TDMA.solve_line` solves all x-lines at once and then all y-lines (vectorised alternating
  direction), instead of one line at a time.
- The momentum equations get 2 TDMA sweeps per outer iteration instead of up to 550. The
  coefficients change at the next outer iteration, so converging the inner system fully is
  wasted work.
- The pressure-correction equation is solved directly with SciPy's sparse LU (`SparseDirect`).
  The TDMA path is still available with `linear_solver="tdma"`.

Measured cost of one outer iteration on the same 20 x 20 mesh: see `scripts/benchmark_original.py`
and the README.

## Not changed on purpose

- First-order upwind convection (`max(-F, 0)` coefficients) and central diffusion.
- Collocated arrangement with Rhie-Chow interpolation, one ghost layer, `(M+2, M+2)` arrays and
  the `i -> x`, `j -> y` index convention.
- `SimpleFields2D.vars` / `add_field` for extra scalars.
