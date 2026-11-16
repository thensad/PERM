# Verification

PERM v32 (`perm.py`, SHA-256 `b48f42c48fec7bfc…`). The runs below were made with the file before its last edit (SHA-256 `57eba7c49afd0e02…`), which differs only in one comment and one docstring; the Python code is identical. All runs were made on one machine: Ubuntu 24.04.5 LTS, x86_64, without a graphics card (software OpenGL under Xvfb for the interface test), memory limit 5.8 GB. **Windows and macOS were not tested.**

## Blender versions

| Blender | Python | NumPy (Blender's) | SciPy (installed by the button) |
|---|---|---|---|
| 4.2.23 LTS | 3.11.7 | 1.24.3 | 1.15.3 |
| 4.5.15 LTS | 3.11.15 | 1.26.4 | 1.17.1 |
| 5.3.0 | 3.13.13 | 2.3.4 | 1.18.1 |

## Summary

| Test | 4.2.23 LTS | 4.5.15 LTS | 5.3.0 |
|---|---|---|---|
| `test_commands.py` | 65 of 65 commands ok, 29 of 29 analyses ok, 0 deviations | 65 of 65 commands ok, 29 of 29 analyses ok, 0 deviations | 65 of 65 commands ok, 29 of 29 analyses ok, 0 deviations |
| `test_benchmarks.py` | 27 of 27 checks pass | 27 of 27 checks pass | 27 of 27 checks pass |
| `test_ui.py` | PASS; 0 tracebacks | PASS; 0 tracebacks | PASS; 0 tracebacks |
| Install SciPy button, then solve | installed SciPy 1.15.3, solved | installed SciPy 1.17.1, solved | installed SciPy 1.18.1, solved |

## Every command

`test_commands.py` calls all 65 operators of the add-on, 259 calls in total, and runs 29 analyses. A command passes when it finishes, or cancels with a message where that is the right answer (nothing selected, no supports, no mesh), without a Python error. Besides that the script compares what was stored and computed with what was set: sums of loads and reactions, the surface-load resultant, the per-node forces, the exported JSON, the face tags after *Clear ALL*, the files of the mesh export.

| Analysis | Nodes | Largest displacement (m) |
|---|---|---|
| A plate, point + line + surface load + self-weight, all result maps | 153 | 0.001254 |
| A plate, self-weight with g = (2, -1, -9.81) | 153 | 0.0009643 |
| B plate with offset faces | 153 | 0.0009635 |
| C plate with full-shell and plate-only faces | 153 | 0.0009643 |
| C vertical wall, membrane-only faces, in-plane load | 45 | 3.284e-05 |
| D plate + truss bars on one edge + beams on the other | 153 | 0.0009844 |
| E triangle mesh as QUAD4 | 121 | 2.413e-05 |
| E triangle mesh as TRI3 | 121 | 2.304e-05 |
| F mixed mesh with one n-gon, quadrilateral topology (the n-gon gives no element; solver warns) | 13 | 0.001598 |
| F same mesh, triangular topology | 13 | 0.0009934 |
| G barrel roof under self-weight | 289 | 0.3434 |
| H two plates, FIXED joint | 85 | 0.004219 |
| H two plates, joint added by hand | 85 | 0.004219 |
| I monkey as QUAD4 (500 faces) | 505 | 0.0002472 |
| I monkey as TRI3 (500 faces) | 505 | 0.0001907 |
| I uv_sphere as QUAD4 (288 faces) | 266 | 5.997e-06 |
| I uv_sphere as TRI3 (288 faces) | 266 | 5.776e-06 |
| I ico_sphere as QUAD4 (320 faces) | 162 | 5.314e-06 |
| I ico_sphere as TRI3 (320 faces) | 162 | 4.965e-06 |
| I torus as QUAD4 (384 faces) | 384 | 6.333e-06 |
| I torus as TRI3 (384 faces) | 384 | 5.656e-06 |
| I cone as QUAD4 (25 faces) | 25 | 1.495e-06 |
| I cone as TRI3 (25 faces) | 25 | 1.495e-06 |
| I cube_scaled as QUAD4 (216 faces) | 218 | 0.0012 |
| I cube_scaled as TRI3 (216 faces) | 218 | 0.0008012 |
| K supports only (default material and thickness) | 9 | 1.4e-05 |
| L girder, details off | 441 | 0.001356 |
| L girder, bridge preset | 441 | 0.001242 |
| N example slab (surface load 5 kN/m2, 20 kN point load, self-weight, steel beam) | 425 | 0.001035 |

The meshes from Blender's *Add* menu (monkey, spheres, torus, cone, box) are there to show that arbitrary meshes are accepted and solved in both topologies; their values have no reference solution. A uniform pressure on the closed sphere, torus and scaled box gives a resultant below 1e-3 N, which checks the face normals under rotation and non-uniform scale.

## Benchmarks

Set up with the panel's commands (`test_benchmarks.py`). "Reference" is a closed-form value computed in the script, or the published value named in the row. The tolerance is the discretisation error accepted for that mesh. Values are from Blender 4.5.15 LTS; the other two versions give the same numbers (largest relative difference between versions 6.2e-12).

Displacements are in m, forces in N, stresses in Pa.

| Case | PERM | Reference | Difference | Tolerance |
|---|---|---|---|---|
| 1a cantilever strip 40x4 quads, tip deflection vs PL^3/3EI + PL/(kGA) (nu = 0) | 0.0213378 | 0.0213413 | 0.016 % | 0.5 % |
| 1b same strip, each quad split into two triangles (TRI3) | 0.0213378 | 0.0213413 | 0.016 % | 2 % |
| 1c strip on edge (in-plane bending, membrane action), tip deflection | 0.00134065 | 0.00134133 | 0.051 % | 1 % |
| 2a patch test, distorted quads: end displacement = sigma L / E | 0.000142857 | 0.000142857 | 0.000 % | 0.0001 % |
| 2b patch test: lateral contraction = -nu sigma Ly / E | -1.78571e-05 | -1.78571e-05 | 0.000 % | 0.0001 % |
| 2c patch test: global sigma_xx of every element (the one furthest from sigma) | 5e+06 | 5e+06 | 0.000 % | 0.0001 % |
| 2d patch test: largest \|sigma_yy\| and \|tau_xy\| of any element (zero) | 0.018149 | 0 | 0.018 (absolute) | 5 |
| 3a SS plate 16x16 quads, pressure (along the normal): centre deflection vs Navier | 0.000443931 | 0.000443609 | 0.073 % | 1 % |
| 3b SS plate: total applied force = q a^2 | 4000 | 4000 | 0.000 % | 1e-07 % |
| 3l same quad plate and pressure, solved as Triangles: total applied force = q a^2 | 4000 | 4000 | 0.000 % | 1e-07 % |
| 3m same: centre deflection | 0.000431524 | 0.000443609 | 2.724 % | 3 % |
| 3c same, surface load in the global direction (0, 0, -q) | 0.000443931 | 0.000443609 | 0.073 % | 1 % |
| 3d same plate under self-weight (rho g t per area) | 0.000683729 | 0.000683233 | 0.073 % | 1 % |
| 3e SS plate 16x16, central point load vs Navier | 0.000316998 | 0.000316698 | 0.095 % | 2 % |
| 3f SS plate 32x32 quads, pressure (mesh refinement) | 0.000444953 | 0.000443609 | 0.303 % | 0.5 % |
| 3g SS plate, 16x16x2 triangles solved as TRI3 | 0.000431524 | 0.000443609 | 2.724 % | 3 % |
| 3h same triangle mesh, topology QUAD4 (triangles joined to quads) | 0.000443931 | 0.000443609 | 0.073 % | 1 % |
| 3j same plate drawn in mm, object scaled by 0.001, rotated 25 degrees and moved (scale not applied) | 0.000443931 | 0.000443609 | 0.073 % | 1 % |
| 3k its surface-load resultant = q a^2 | 4000 | 4000 | 0.000 % | 0.001 % |
| 3i quarter plate 8x8 with two symmetry planes (CUSTOM: u_n and the two in-plane rotations) | 0.000443931 | 0.000443609 | 0.073 % | 1 % |
| 4a clamped plate 16x16, pressure: centre deflection vs 0.00126 q a^4 / D (Timoshenko's table) | 0.000138045 | 0.000137592 | 0.329 % | 2 % |
| 5a Scordelis-Lo roof, 16x16 over the whole roof, self-weight: free-edge midspan deflection vs 0.3024 | 0.30385 | 0.3024 | 0.480 % | 3 % |
| 5b same roof 32x32, load as a surface load of 90 N/m2 | 0.301655 | 0.3024 | 0.246 % | 2 % |
| 6a hemisphere with 18 degree hole, 16 elements per quadrant side: radial displacement under the load vs 0.0940 | 0.0928225 | 0.094 | 1.253 % | 5 % |
| 7a beam on 12 edges (no faces), axial extension = P L / E A | 0.000125 | 0.000125 | 0.000 % | 0.0001 % |
| 7b same beam, tip deflection = P L^3 / 3 E I + P L / (k G A), k = 5/6 | 0.200016 | 0.200016 | 0.000 % | 0.1 % |
| 7c truss on 12 edges, extension = P L / E A (A from the section) | 0.000125 | 0.000125 | 0.000 % | 0.0001 % |

Reference values: the simply supported plate is the Navier double sine series (Kirchhoff plate theory), summed in the script; it reproduces the coefficients 0.00406 (uniform load) and 0.01160 (central load) of Timoshenko and Woinowsky-Krieger, *Theory of Plates and Shells*. The clamped plate uses their coefficient 0.00126. 0.3024 (Scordelis-Lo roof) and 0.0940 (hemisphere with an 18° hole) are the values commonly used for these two problems of the shell test set of MacNeal and Harder (1985). PERM's shell is a Reissner-Mindlin element, so on a simply supported edge it is slightly more flexible than the Kirchhoff reference as the mesh is refined (3a against 3f).

## Not covered

- Windows and macOS; Blender older than 4.2.
- *Open Results Folder*: the command ran, but the test machine has no file manager, so no folder window opened.
- The SciPy installer where pip is blocked or needs a proxy.
- A real display: the interface test draws with software OpenGL and calls the operators with `INVOKE_DEFAULT`; no mouse click on a button is simulated.
- Nonlinear, dynamic and buckling behaviour (not in the add-on).
- An independent check of surface loads and custom restraints beyond the benchmarks above.

