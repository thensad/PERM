"""PERM against closed-form and published reference solutions, set up through the panel's commands.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_benchmarks.py -- --addon perm.py [--site <site-packages with SciPy>] [--out report.json]

Each case builds a mesh, assigns material, section, supports and loads with the same operators the panel calls,
solves, and compares one displacement with the reference. The tolerance of each check is the discretisation error
expected of that mesh, stated with the check. Exit code 1 if a check fails.
"""
import os, sys, math, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy
import perm_testlib as T
from perm_testlib import call, scenario, select_verts, select_faces, check

perm = T.load_addon()
scn = bpy.context.scene
scn.perm_results_dir = tempfile.mkdtemp(prefix="perm_bench_")
OUT = T.opt("--out")


def model(name, V, F, E, nu, rho, t, tri_edges=(), edges=()):
    T.clear_scene()
    o = T.new_mesh(name, V, F, edges)
    call("perm.add_material", name="M", youngs_modulus=E, poissons_ratio=nu, density=rho)
    call("perm.add_section", name="S", thickness=t)
    scn.perm_material_index = 0; scn.perm_section_index = 0
    if F:
        select_faces(o, None)
        call("perm.assign_material_to_faces"); call("perm.assign_section_to_faces")
        T.obj_mode()
    call("perm.reset_details")
    scn.perm_gravity_x = 0.0; scn.perm_gravity_y = 0.0; scn.perm_gravity_z = -9.81
    scn.perm_mesh_topology = 'QUAD4'
    return o


def sup(o, fn, name, kind, dofs=""):
    idx = [v.index for v in o.data.vertices if fn(v.co)]
    assert idx, f"support {name}: no vertices"
    select_verts(o, idx)
    kw = {("fix_" + d): (d in dofs.split()) for d in perm.PERM_DOF_NAMES} if kind == 'CUSTOM' else {}
    call("perm.assign_bc", group_name=name, bc_type=kind, **kw)
    T.obj_mode()
    return idx


def pload(o, fn, name, fx=0.0, fy=0.0, fz=0.0):
    idx = [v.index for v in o.data.vertices if fn(v.co)]
    assert idx, f"load {name}: no vertices"
    select_verts(o, idx)
    call("perm.assign_load_per_node", group_name=name, fx=fx, fy=fy, fz=fz)
    T.obj_mode()
    return idx


def surface(o, mode, q=(0.0, 0.0, 0.0), p=0.0, faces=None):
    select_faces(o, faces)
    scn.perm_sl_mode = mode; scn.perm_sl_vector = q; scn.perm_sl_pressure = p; scn.perm_sl_add = False
    call("perm.set_surface_load")
    T.obj_mode()


def u_at(s, xyz, tol=1e-6):
    d = T.node_disp(s, xyz, tol)
    assert len(d) == 1, f"{len(d)} nodes at {xyz}"
    return d[0]


def near(a, b=0.0, tol=1e-6):
    return abs(a - b) < tol


def navier_ss_plate(a, b, D, q=0.0, P=0.0, terms=199):
    """Centre deflection of a simply supported thin plate (Kirchhoff), Navier double sine series:
    uniform load q and / or a central point load P."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            den = (math.pi ** 4) * D * ((m / a) ** 2 + (n / b) ** 2) ** 2
            s = math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            w += (16.0 * q / (math.pi ** 2 * m * n)) / den * s          # q_mn = 16 q / (pi^2 m n)
            w += (4.0 * P / (a * b)) * s / den * s                       # P_mn = 4 P / (a b) sin sin
    return w


# ═══════════════════════════════════════════════════════════ 1. cantilever strip, tip load (beam theory)
scenario("1 cantilever strip")
L, b, t, E = 2.0, 0.2, 0.05, 30e9
V, F = T.grid(L, b, 40, 4)
o = model("Cantilever", V, F, E, 0.0, 0.0, t)
sup(o, lambda c: near(c.x), "root", 'FIXED')
tip = pload(o, lambda c: near(c.x, L), "tip", fz=-100.0)            # 5 nodes x 100 N
s = T.solve(o)
Ptot = 500.0; I = b * t ** 3 / 12.0; G = E / 2.0
ref = Ptot * L ** 3 / (3 * E * I) + Ptot * L / (5.0 / 6.0 * G * b * t)
check("1a cantilever strip 40x4 quads, tip deflection vs PL^3/3EI + PL/(kGA) (nu = 0)", -u_at(s, (L, b / 2, 0))[2], ref, rtol=0.005)
call("perm.clear_results"); T.activate(o)
scn.perm_mesh_topology = 'TRI3'
s = T.solve(o)
check("1b same strip, each quad split into two triangles (TRI3)", -u_at(s, (L, b / 2, 0))[2], ref, rtol=0.02)
scn.perm_mesh_topology = 'QUAD4'
# in-plane (membrane) bending of the same strip set on edge: QM6 incompatible modes
V2 = [(x, 0.0, y) for x, y, _ in V]
o = model("CantileverOnEdge", V2, F, E, 0.0, 0.0, t)
sup(o, lambda c: near(c.x), "root", 'FIXED')
pload(o, lambda c: near(c.x, L), "tip", fz=-100.0)
s = T.solve(o)
I2 = t * b ** 3 / 12.0
ref2 = Ptot * L ** 3 / (3 * E * I2) + Ptot * L / (5.0 / 6.0 * G * b * t)
check("1c strip on edge (in-plane bending, membrane action), tip deflection", -u_at(s, (L, 0, b / 2))[2], ref2, rtol=0.01)

# ═══════════════════════════════════════════════════════════ 2. membrane patch test on a distorted mesh
scenario("2 membrane patch test")
Lx, Ly, t, E, nu = 2.0, 1.0, 0.1, 70e9, 0.25
V, F = T.grid(Lx, Ly, 4, 2)
rng = np.random.default_rng(3)
inside = lambda x, y: 1e-9 < x < Lx - 1e-9 and 1e-9 < y < Ly - 1e-9          # move the interior nodes only
V = [((x + rng.uniform(-0.12, 0.12), y + rng.uniform(-0.12, 0.12), 0.0) if inside(x, y) else (x, y, 0.0)) for x, y, _ in V]
o = model("Patch", V, F, E, nu, 0.0, t)
sup(o, lambda c: near(c.x), "left", 'CUSTOM', "ux uz")
sup(o, lambda c: near(c.x) and near(c.y), "corner", 'CUSTOM', "uy")
sup(o, lambda c: near(c.x, Lx) and near(c.y), "far", 'CUSTOM', "uz")
sig = 5.0e6                                             # uniform tension on the right edge, consistent nodal forces
pload(o, lambda c: near(c.x, Lx) and (near(c.y) or near(c.y, Ly)), "end_corners", fx=sig * t * Ly / 4)
pload(o, lambda c: near(c.x, Lx) and near(c.y, Ly / 2), "end_mid", fx=sig * t * Ly / 2)
s = T.solve(o)
check("2a patch test, distorted quads: end displacement = sigma L / E", u_at(s, (Lx, Ly, 0))[0], sig * Lx / E, rtol=1e-6)
check("2b patch test: lateral contraction = -nu sigma Ly / E", u_at(s, (Lx, Ly, 0))[1], -nu * sig * Ly / E, rtol=1e-6)
G_ = np.array(list(s.results.stresses_global.values()))       # [s_xx, s_yy, s_xy] in the global axes
check("2c patch test: global sigma_xx of every element (the one furthest from sigma)", float(G_[np.argmax(np.abs(G_[:, 0] - sig)), 0]), sig, rtol=1e-6)
check("2d patch test: largest |sigma_yy| and |tau_xy| of any element (zero)", float(np.abs(G_[:, 1:]).max()), 0.0, rtol=0.0, atol=sig * 1e-6)

# ═══════════════════════════════════════════════════════════ 3. simply supported square plate (Navier)
scenario("3 simply supported plate")
a, t, E, nu = 2.0, 0.02, 200e9, 0.3
D = E * t ** 3 / (12 * (1 - nu ** 2))
q = 1000.0
w_q = navier_ss_plate(a, a, D, q=q)
print(f"Navier series: w_centre = {w_q:.6e} m = {w_q * D / (q * a ** 4):.5f} q a^4 / D   (Timoshenko's table: 0.00406)")
edge = lambda c: near(c.x) or near(c.x, a) or near(c.y) or near(c.y, a)


def ss_plate(n, tri=False):
    V, F = T.grid(a, a, n, n, tri=tri)
    o = model("SSPlate", V, F, E, nu, 7850.0, t)
    sup(o, edge, "edges", 'CUSTOM', "uz")
    sup(o, lambda c: near(c.x) and near(c.y), "c00", 'CUSTOM', "ux uy")
    sup(o, lambda c: near(c.x, a) and near(c.y), "c10", 'CUSTOM', "uy")
    return o


o = ss_plate(16)
surface(o, 'NORMAL', p=q)                                # face normals +z: the pressure pushes down
s = T.solve(o)
check("3a SS plate 16x16 quads, pressure (along the normal): centre deflection vs Navier", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.01)
Fa = np.asarray(s.results.applied_total)
check("3b SS plate: total applied force = q a^2", -Fa[2], q * a * a, rtol=1e-9)
call("perm.clear_results"); T.activate(o)
scn.perm_mesh_topology = 'TRI3'                           # the loaded quads are split: the load must follow the faces
s = T.solve(o)
check("3l same quad plate and pressure, solved as Triangles: total applied force = q a^2", -np.asarray(s.results.applied_total)[2], q * a * a, rtol=1e-9)
check("3m same: centre deflection", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.03)
scn.perm_mesh_topology = 'QUAD4'
call("perm.clear_results"); T.activate(o)
surface(o, 'GLOBAL', q=(0.0, 0.0, -q))
s = T.solve(o)
check("3c same, surface load in the global direction (0, 0, -q)", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.01)
call("perm.clear_results"); T.activate(o)
call("perm.clear_surface_load", all_faces=True)
s = T.solve(o, gravity=True)                              # self-weight = rho g t per area
w_g = navier_ss_plate(a, a, D, q=7850.0 * 9.81 * t)
check("3d same plate under self-weight (rho g t per area)", -u_at(s, (a / 2, a / 2, 0))[2], w_g, rtol=0.01)
call("perm.clear_results"); T.activate(o)
P0 = 1000.0
pload(o, lambda c: near(c.x, a / 2) and near(c.y, a / 2), "P", fz=-P0)
s = T.solve(o, gravity=False)
w_p = navier_ss_plate(a, a, D, P=P0)
print(f"Navier series, central point load: w = {w_p * D / (P0 * a ** 2):.5f} P a^2 / D   (Timoshenko's table: 0.01160)")
check("3e SS plate 16x16, central point load vs Navier", -u_at(s, (a / 2, a / 2, 0))[2], w_p, rtol=0.02)
o = ss_plate(32)
surface(o, 'NORMAL', p=q)
s = T.solve(o)
check("3f SS plate 32x32 quads, pressure (mesh refinement)", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.005)
o = ss_plate(16, tri=True)
scn.perm_mesh_topology = 'TRI3'
surface(o, 'NORMAL', p=q)
s = T.solve(o)
check("3g SS plate, 16x16x2 triangles solved as TRI3", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.03)
call("perm.clear_results"); T.activate(o)
scn.perm_mesh_topology = 'QUAD4'
s = T.solve(o)
check("3h same triangle mesh, topology QUAD4 (triangles joined to quads)", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.01)
# the same plate drawn in millimetres and brought to metres by the OBJECT scale (not applied): PERM reads world coordinates
V, F = T.grid(a * 1000.0, a * 1000.0, 16, 16)
o = model("SSPlate_mm", V, F, E, nu, 0.0, t)
o.scale = (0.001, 0.001, 0.001); o.location = (3.0, -1.0, 0.5); o.rotation_euler = (0.0, 0.0, math.radians(25.0))
bpy.context.view_layer.update()
edge_mm = lambda c: near(c.x, 0, 1e-3) or near(c.x, a * 1000, 1e-3) or near(c.y, 0, 1e-3) or near(c.y, a * 1000, 1e-3)
sup(o, edge_mm, "edges", 'CUSTOM', "uz")
sup(o, lambda c: near(c.x, 0, 1e-3) and near(c.y, 0, 1e-3), "c00", 'CUSTOM', "ux uy")
sup(o, lambda c: near(c.x, a * 1000, 1e-3) and near(c.y, 0, 1e-3), "c10", 'CUSTOM', "ux uy")
surface(o, 'NORMAL', p=q)
s = T.solve(o)
centre = o.matrix_world @ o.data.vertices[8 * 17 + 8].co
check("3j same plate drawn in mm, object scaled by 0.001, rotated 25 degrees and moved (scale not applied)", -u_at(s, tuple(centre), tol=1e-5)[2], w_q, rtol=0.01)
check("3k its surface-load resultant = q a^2", -np.asarray(s.results.applied_total)[2], q * a * a, rtol=1e-5)
# quarter plate with symmetry planes (CUSTOM restraints with rotations)
V, F = T.grid(a / 2, a / 2, 8, 8)
o = model("SSQuarter", V, F, E, nu, 0.0, t)
sup(o, lambda c: near(c.x) or near(c.y), "edges", 'CUSTOM', "uz")
sup(o, lambda c: near(c.x, a / 2), "symX", 'CUSTOM', "ux ry rz")      # plane x = a/2, normal X
sup(o, lambda c: near(c.y, a / 2), "symY", 'CUSTOM', "uy rx rz")      # plane y = a/2, normal Y
surface(o, 'NORMAL', p=q)
s = T.solve(o)
check("3i quarter plate 8x8 with two symmetry planes (CUSTOM: u_n and the two in-plane rotations)", -u_at(s, (a / 2, a / 2, 0))[2], w_q, rtol=0.01)

# ═══════════════════════════════════════════════════════════ 4. clamped square plate
scenario("4 clamped plate")
V, F = T.grid(a, a, 16, 16)
o = model("ClampedPlate", V, F, E, nu, 0.0, t)
sup(o, edge, "edges", 'FIXED')
surface(o, 'NORMAL', p=q)
s = T.solve(o)
check("4a clamped plate 16x16, pressure: centre deflection vs 0.00126 q a^4 / D (Timoshenko's table)", -u_at(s, (a / 2, a / 2, 0))[2], 0.00126 * q * a ** 4 / D, rtol=0.02)

# ═══════════════════════════════════════════════════════════ 5. Scordelis-Lo barrel roof
scenario("5 Scordelis-Lo roof")
R, Lr, tr = 25.0, 50.0, 0.25


def roof(n, surface_load):
    V, F = T.cylinder_patch(R, Lr, 40.0, n, n)
    o = model("Roof", V, F, 4.32e8, 0.0, 360.0 / 9.81, tr)
    sup(o, lambda c: near(c.y) or near(c.y, Lr), "diaphragms", 'CUSTOM', "ux uz")
    sup(o, lambda c: near(c.y) and near(c.x), "axial", 'CUSTOM', "uy")
    if surface_load:
        surface(o, 'GLOBAL', q=(0.0, 0.0, -90.0))
        s = T.solve(o, gravity=False)
    else:
        s = T.solve(o, gravity=True)                      # 360 / 9.81 kg/m3 x 9.81 x 0.25 m = 90 N/m2
    xe = R * math.sin(math.radians(40.0)); ze = R * math.cos(math.radians(40.0))
    return -u_at(s, (xe, Lr / 2, ze), tol=1e-5)[2]


check("5a Scordelis-Lo roof, 16x16 over the whole roof, self-weight: free-edge midspan deflection vs 0.3024", roof(16, False), 0.3024, rtol=0.03)
check("5b same roof 32x32, load as a surface load of 90 N/m2", roof(32, True), 0.3024, rtol=0.02)

# ═══════════════════════════════════════════════════════════ 6. pinched hemisphere with an 18 degree hole
scenario("6 pinched hemisphere")
Rh, th = 10.0, 0.04


def hemisphere(n):
    """Whole hemisphere, 4n elements around, n from the equator to the 18 degree hole."""
    V = []; nphi = 4 * n
    for j in range(n + 1):
        lat = math.radians(72.0 * j / n)
        for i in range(nphi):
            lon = 2 * math.pi * i / nphi
            V.append((Rh * math.cos(lat) * math.cos(lon), Rh * math.cos(lat) * math.sin(lon), Rh * math.sin(lat)))
    F = [(j * nphi + i, j * nphi + (i + 1) % nphi, (j + 1) * nphi + (i + 1) % nphi, (j + 1) * nphi + i) for j in range(n) for i in range(nphi)]
    o = model("Hemisphere", V, F, 6.825e7, 0.3, 0.0, th)
    # two pairs of opposite forces on the equator (self-equilibrated); six DOFs held at three load points
    sup(o, lambda c: near(c.x, Rh, 1e-4) and near(c.z), "A", 'CUSTOM', "uy uz")
    sup(o, lambda c: near(c.x, -Rh, 1e-4) and near(c.z), "B", 'CUSTOM', "uy uz")
    sup(o, lambda c: near(c.y, Rh, 1e-4) and near(c.z), "C", 'CUSTOM', "ux uz")
    pload(o, lambda c: near(c.x, Rh, 1e-4) and near(c.z), "px+", fx=2.0)
    pload(o, lambda c: near(c.x, -Rh, 1e-4) and near(c.z), "px-", fx=-2.0)
    pload(o, lambda c: near(c.y, Rh, 1e-4) and near(c.z), "py+", fy=-2.0)
    pload(o, lambda c: near(c.y, -Rh, 1e-4) and near(c.z), "py-", fy=2.0)
    s = T.solve(o)
    return u_at(s, (Rh, 0, 0), tol=1e-4)[0]


check("6a hemisphere with 18 degree hole, 16 elements per quadrant side: radial displacement under the load vs 0.0940", hemisphere(16), 0.0940, rtol=0.05)

# ═══════════════════════════════════════════════════════════ 7. bars on edges: truss and beam
scenario("7 truss and beam on edges")
Lb, n = 3.0, 12
V = [(Lb * i / n, 0.0, 0.0) for i in range(n + 1)]
Ed = [(i, i + 1) for i in range(n)]


def bar_model(kind):
    T.clear_scene()
    o = T.new_mesh("Bar", V, [], Ed)
    call("perm.add_material", name="Steel", youngs_modulus=200e9, poissons_ratio=0.3, density=0.0)
    call("perm.add_section", name="Bar", thickness=0.01, area=6.0e-4)
    scn.perm_material_index = 0; scn.perm_section_index = 0
    sec = scn.perm_sections[0]
    sec.beam_shape = 'RECT'; sec.dim_b = 0.02; sec.dim_h = 0.03
    call("perm.add_element_type", name="E", kind=kind)
    scn.perm_element_type_index = 0
    select_verts(o, range(n + 1), 'EDGE')
    call("perm.assign_element_type", replace_existing=True)
    T.obj_mode()
    call("perm.reset_details")
    return o


o = bar_model('BEAM_B2')
sup(o, lambda c: near(c.x), "root", 'FIXED')
pload(o, lambda c: near(c.x, Lb), "tip", fx=5000.0, fz=-200.0)
s = T.solve(o)
Eb, A, Iy = 200e9, 0.02 * 0.03, 0.02 * 0.03 ** 3 / 12.0
Gb = Eb / 2.6
u = u_at(s, (Lb, 0, 0))
check("7a beam on 12 edges (no faces), axial extension = P L / E A", u[0], 5000.0 * Lb / (Eb * A), rtol=1e-6)
check("7b same beam, tip deflection = P L^3 / 3 E I + P L / (k G A), k = 5/6", -u[2], 200.0 * Lb ** 3 / (3 * Eb * Iy) + 200.0 * Lb / (5.0 / 6.0 * Gb * A), rtol=1e-3)
o = bar_model('TRUSS_T2')
sup(o, lambda c: near(c.x), "root", 'PINNED')
sup(o, lambda c: c.x > 1e-6, "guide", 'CUSTOM', "uy uz")
pload(o, lambda c: near(c.x, Lb), "tip", fx=5000.0)
s = T.solve(o)
check("7c truss on 12 edges, extension = P L / E A (A from the section)", u_at(s, (Lb, 0, 0))[0], 5000.0 * Lb / (200e9 * 6.0e-4), rtol=1e-6)

bad = [c for c in T.CHECKS if not c['ok']]
print(f"\n================ benchmarks: {len(T.CHECKS) - len(bad)} of {len(T.CHECKS)} checks pass ================")
for c in T.CHECKS:
    if c['expected'] != 0:
        d = f"difference {100 * c['rel_err']:.3f} % (tolerance {100 * c['tol']:g} %)"
    else:
        d = f"difference {c['abs_err']:.3g} (tolerance {c['atol']:.3g})"
    print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}: PERM {c['got']:.6g}, reference {c['expected']:.6g}, {d}")
ops_bad = T.summary(sorted(T.LOG))
if OUT:
    json.dump(dict(checks=T.CHECKS, blender=bpy.app.version_string), open(OUT, "w"), indent=1)
print(f"\nRESULT: {'PASS' if not bad and not ops_bad else 'FAIL'}", flush=True)
sys.exit(0 if not bad and not ops_bad else 1)
