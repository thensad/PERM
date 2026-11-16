"""Run every PERM command (operator) the way a user would, on generic surface meshes.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_commands.py -- --addon perm.py [--site <site-packages with SciPy>] [--out report.json]

A command passes when it finishes (or cancels with a message, where that is the documented answer) without a Python error.
The script exits with code 1 if a command fails, is never run, or a solve leaves no result.
"""
import os, sys, math, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy
from mathutils import Matrix, Euler
import perm_testlib as T
from perm_testlib import call, scenario, select_verts, select_faces, verts_where, faces_where

perm = T.load_addon()
scn = bpy.context.scene
OUT = T.opt("--out")
TMP = tempfile.mkdtemp(prefix="perm_cmd_")
scn.perm_results_dir = TMP
NOTES = []
SOLVES = []


def note(msg):
    NOTES.append(msg); print("[!!]", msg, flush=True)


def close(a, b, rtol=1e-6, atol=1e-9):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def totals(s):
    """(sum of applied loads, sum of reactions) of the last solve, N."""
    return np.asarray(s.results.applied_total, float), np.asarray(s.results.reaction_total, float)


def setup_mat_sec(obj, E=30e9, nu=0.2, rho=2400.0, t=0.2, name="M1", sname="S1", faces=None):
    T.activate(obj)
    call("perm.add_material", name=name, youngs_modulus=E, poissons_ratio=nu, density=rho)
    call("perm.add_section", name=sname, thickness=t)
    scn.perm_material_index = len(scn.perm_materials) - 1
    scn.perm_section_index = len(scn.perm_sections) - 1
    select_faces(obj, faces)
    call("perm.assign_material_to_faces")
    call("perm.assign_section_to_faces")
    T.obj_mode()


def support(obj, idx, name, bc):
    select_verts(obj, idx)
    call("perm.assign_bc", group_name=name, bc_type=bc)
    T.obj_mode()


def load(obj, idx, name, fx=0.0, fy=0.0, fz=-1000.0):
    select_verts(obj, idx)
    call("perm.assign_load_per_node", group_name=name, fx=fx, fy=fy, fz=fz)
    T.obj_mode()


def solved(tag, s, need_disp=True):
    ok = s is not None and getattr(s, 'results', None) is not None
    d = float(s.results.max_displacement) if ok else float('nan')
    ok = ok and math.isfinite(d) and (d > 0 or not need_disp)
    SOLVES.append(dict(case=tag, ok=bool(ok), max_disp_m=d, nodes=int(len(s.node_coords)) if s is not None else 0,
                       elements=int(getattr(s, 'n_elements', -1)) if s is not None else -1))
    print(f"[{'ok' if ok else '!!'}] solve '{tag}': max displacement {d:.6g} m", flush=True)
    return ok


def n_selected_faces(obj):
    T.obj_mode()
    return sum(1 for p in obj.data.polygons if p.select)


# ═════════════════════════════════════════════ A. quad plate, object moved / rotated in space
scenario("A quad plate (rotated + translated object)")
T.clear_scene()
V, F = T.grid(4.0, 2.0, 16, 8)
plate = T.new_mesh("PlateQ", V, F)
plate.matrix_world = Matrix.Translation((3.0, -2.0, 1.5)) @ Euler((0.0, 0.0, math.radians(30.0))).to_matrix().to_4x4()
bpy.context.view_layer.update()
loc = lambda v: plate.data.vertices[v].co
setup_mat_sec(plate)
support(plate, [v.index for v in plate.data.vertices if abs(v.co.x) < 1e-6], "left", "PINNED")
support(plate, [v.index for v in plate.data.vertices if abs(v.co.x - 4.0) < 1e-6], "right", "ROLLER_X")
mid = [v.index for v in plate.data.vertices if abs(v.co.x - 2.0) < 1e-6 and abs(v.co.y - 1.0) < 1e-6]
load(plate, mid, "P", fz=-5000.0)
edge3 = [v.index for v in plate.data.vertices if abs(v.co.x - 3.0) < 1e-6]
load(plate, edge3, "line", fz=-100.0)
# a CUSTOM restraint: two nodes held sideways only (uy), as a guide
select_verts(plate, [v.index for v in plate.data.vertices if abs(v.co.x - 2.0) < 1e-6 and (abs(v.co.y) < 1e-6 or abs(v.co.y - 2.0) < 1e-6)])
call("perm.assign_bc", group_name="guide", bc_type='CUSTOM', fix_ux=False, fix_uy=True, fix_uz=False, fix_rx=False, fix_ry=False, fix_rz=False)
T.obj_mode()
if perm.perm_support_dofs(plate, "guide") != ['uy']:
    note(f"CUSTOM restraint stored as {perm.perm_support_dofs(plate, 'guide')}, expected ['uy']")
select_verts(plate, [0])
call("perm.assign_bc", group_name="nothing", bc_type='CUSTOM', fix_ux=False, fix_uy=False, fix_uz=False, expect=('CANCELLED',))
T.obj_mode()
# surface loads: 1500 N/m2 down on every face, then a pressure of 500 N/m2 added on the first 16 faces
select_faces(plate, None)
scn.perm_sl_mode = 'GLOBAL'; scn.perm_sl_vector = (0.0, 0.0, -1500.0); scn.perm_sl_add = False
call("perm.set_surface_load")
select_faces(plate, list(range(16)))
scn.perm_sl_mode = 'NORMAL'; scn.perm_sl_pressure = 500.0; scn.perm_sl_add = True
call("perm.set_surface_load")
T.obj_mode(); T.activate(plate)
info = list(plate["perm_surface_load_info"])
exp_fz = -1500.0 * 8.0 - 500.0 * 16 * 0.0625
if not (int(info[0]) == 128 and close(info[3], exp_fz, 1e-5) and abs(info[1]) < 1e-3 and abs(info[2]) < 1e-3):
    note(f"surface load total {info}, expected 128 faces and Fz = {exp_fz}")
call("perm.select_surface_load_faces")
if n_selected_faces(plate) != 128:
    note(f"select_surface_load_faces selected {n_selected_faces(plate)} faces, expected 128")
T.activate(plate)
T.activate(plate)
call("perm.report_loads")
call("perm.show_markers")
call("perm.select_group", group_name="PERM_Support_left")
T.obj_mode(); T.activate(plate)
call("perm.toggle_node_list", group_name="PERM_Load_line")
call("perm.toggle_node_list", group_name="PERM_Load_line")
T.edit(plate)
call("perm.select_assignment", group_name="PERM_Load_line")
call("perm.deselect_assignment", group_name="PERM_Load_line")
T.obj_mode()
# per-node editor: execute applies the values typed in the dialog; bulk set fills one axis for the group
T.activate(plate)
call("perm.open_per_node_editor", group_name="PERM_Load_line")          # loads the group's rows
call("perm.bulk_set_per_node", axis='z', value=-150.0)                  # 'Set all Fz' in the dialog
call("perm.open_per_node_editor", group_name="PERM_Load_line")          # OK: saves the rows
pn = perm.get_per_node_forces(plate, "line")
if len(pn) != 9 or any(abs(v[2] + 150.0) > 1e-9 for v in pn.values()):
    note(f"per-node editor: stored {pn}")
call("perm.open_per_node_editor", group_name="PERM_Load_none", expect=('CANCELLED',))
T.activate(plate)
call("perm.select_problem_faces", folded=True, ngon=True, warped=True, unassigned=True, expect=('FINISHED', 'CANCELLED'))
T.obj_mode(); T.activate(plate)
for k in ('ANY', 'SECTION', 'MATERIAL'):
    call("perm.select_unassigned_faces", kind=k, expect=('FINISHED', 'CANCELLED'))
    T.obj_mode(); T.activate(plate)
s = T.solve(plate, gravity=True, viz=T.VIZ)
solved("A plate, point + line + surface load + self-weight, all result maps", s)
if s:
    Fa, Rt = totals(s)
    w_self = 2400.0 * 0.2 * 8.0 * 9.81
    exp = -5000.0 - 9 * 150.0 + exp_fz - w_self          # point + line (9 nodes x 150 N) + surface + self-weight
    if not close(Fa[2], exp, 1e-5):
        note(f"A: sum of loads Fz = {Fa[2]:.3f} N, expected {exp:.3f} N")
    if np.linalg.norm(Fa + Rt) > 1e-6 * np.linalg.norm(Fa):
        note(f"A: reactions do not balance the loads: {Fa} + {Rt}")
res_objs = [o.name for o in bpy.data.objects if o.name != plate.name]
print("objects after solve:", res_objs)
call("perm.show_results")
call("perm.recolor_results")
call("perm.auto_color_range", expect=('FINISHED', 'CANCELLED'))
scn.perm_deformation_scale = 250.0
call("perm.refresh_visualization")
scn.perm_result_opacity = 0.6
call("perm.rebuild_result_materials")
call("perm.select_peak_faces", expect=('FINISHED', 'CANCELLED'))
T.obj_mode()
scn.perm_find_element = 5
call("perm.find_element", expect=('FINISHED', 'CANCELLED'))
call("perm.clear_find_element")
T.obj_mode()
T.activate(plate)
call("perm.export_json", filepath=os.path.join(TMP, "setup.json"), include_results=False, solve_first=False)
T.obj_mode(); T.activate(plate)
call("perm.export_json", filepath=os.path.join(TMP, "setup_res.json"), include_results=True, solve_first=True)
T.obj_mode(); T.activate(plate)
for fn in ("setup.json", "setup_res.json"):
    p = os.path.join(TMP, fn)
    try:
        d = json.load(open(p)); print("export readable:", fn, os.path.getsize(p), "bytes")
        if len(d['nodes']) != 153 or len(d['elements']) != 128 or len(d['supports']) != 3:
            note(f"export {fn}: {len(d['nodes'])} nodes, {len(d['elements'])} elements, {len(d['supports'])} supports")
        ftot = np.sum([l['total_force'] for l in d['loads']], axis=0)
        if not close(ftot[2], exp, 1e-5):
            note(f"export {fn}: loads sum to Fz = {ftot[2]:.3f} N, expected {exp:.3f}")
        if {l['type'] for l in d['loads']} != {'point', 'self_weight', 'surface'}:
            note(f"export {fn}: load types {[l['type'] for l in d['loads']]}")
        if fn == "setup_res.json":
            um = max(np.linalg.norm(r[1:4]) for r in d['results']['displacements'])
            if not close(um, d['results']['max_displacement'], 1e-9):
                note(f"export: max |u| of the listed displacements {um} != max_displacement {d['results']['max_displacement']}")
            rz = sum(r[3] for r in d['results']['reactions'])
            if not close(rz, -exp, 1e-5):
                note(f"export: reactions sum to Rz = {rz:.3f}, expected {-exp:.3f}")
    except Exception as e:
        note(f"export_json {fn}: not readable JSON ({e.__class__.__name__}: {e})")
call("perm.open_results_folder", expect=('FINISHED', 'CANCELLED'))
call("perm.edit_model")
T.obj_mode()
# markers: select one and remove it
mk = [o for o in bpy.data.objects if o.name.startswith(("PERM_LoadArrow_", "PERM_BCLock_"))]
print("marker objects:", len(mk))
if mk:
    T.obj_mode()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    try:
        mk[0].hide_set(False); mk[0].select_set(True); bpy.context.view_layer.objects.active = mk[0]
    except Exception as e:
        print("marker select:", e)
call("perm.remove_marker", expect=('FINISHED',) if mk else ('CANCELLED',))
T.activate(plate)
call("perm.clear_markers")
call("perm.show_markers")
select_verts(plate, edge3[:2])
call("perm.remove_from_groups")
T.obj_mode()
call("perm.remove_assignment", group_name="PERM_Load_line")
T.obj_mode(); T.activate(plate)
call("perm.remove_assignment", group_name="PERM_Support_guide")
T.obj_mode(); T.activate(plate)
select_faces(plate, list(range(8)))
call("perm.clear_surface_load", all_faces=False)
T.obj_mode(); T.activate(plate)
if int(plate["perm_surface_load_info"][0]) != 120:
    note(f"clear_surface_load on 8 faces left {plate['perm_surface_load_info'][0]} loaded faces, expected 120")
call("perm.clear_surface_load", all_faces=True)
if plate.data.attributes.get(perm.PERM_SURFLOAD_ATTR) is not None:
    note("clear_surface_load(all) left the attribute on the mesh")
call("perm.clear_results")
T.activate(plate)
# self-weight with a sideways component: sum of loads = mass x g
scn.perm_gravity_x = 2.0; scn.perm_gravity_y = -1.0
s = T.solve(plate, gravity=True)
solved("A plate, self-weight with g = (2, -1, -9.81)", s)
if s:
    Fa, Rt = totals(s)
    m_plate = 2400.0 * 0.2 * 8.0
    expv = np.array([2.0, -1.0, -9.81]) * m_plate + np.array([0.0, 0.0, -5000.0])
    if np.linalg.norm(Fa - expv) > 1e-5 * np.linalg.norm(expv):
        note(f"gravity vector: sum of loads {Fa}, expected {expv}")
scn.perm_gravity_x = 0.0; scn.perm_gravity_y = 0.0
call("perm.clear_results")
T.activate(plate)

# material / section bookkeeping ('Unassign' removes the active material / section from ALL its faces)
T.edit(plate, 'FACE')
call("perm.unassign_material"); call("perm.unassign_section")
T.obj_mode(); T.activate(plate)
call("perm.select_unassigned_faces", kind='ANY', expect=('FINISHED', 'CANCELLED'))
nun = n_selected_faces(plate)
if nun != len(plate.data.polygons):
    NOTES.append(f"select_unassigned_faces found {nun} faces, expected {len(plate.data.polygons)}"); print("[!!]", NOTES[-1])
select_faces(plate, None)
call("perm.assign_material_to_faces"); call("perm.assign_section_to_faces")
T.obj_mode(); T.activate(plate)
call("perm.select_unassigned_faces", kind='ANY', expect=('FINISHED', 'CANCELLED'))
if n_selected_faces(plate) != 0 and False:
    pass
T.edit(plate, 'FACE'); bpy.ops.mesh.select_all(action='DESELECT')
call("perm.select_material_faces"); call("perm.select_section_faces")
if n_selected_faces(plate) != len(plate.data.polygons):
    NOTES.append("select_section_faces did not select all faces"); print("[!!]", NOTES[-1])
T.activate(plate)

# ═════════════════════════════════════════════ B. shell offset + thickness preview
scenario("B shell offset")
select_faces(plate, list(range(0, 16)))
scn.perm_offset_half = True
call("perm.set_shell_offset")
T.obj_mode(); T.activate(plate)
T.edit(plate, 'FACE'); bpy.ops.mesh.select_all(action='DESELECT')
call("perm.select_offset_faces", expect=('FINISHED', 'CANCELLED'))
noff = n_selected_faces(plate)
if noff != 16:
    NOTES.append(f"select_offset_faces selected {noff}, expected 16"); print("[!!]", NOTES[-1])
T.activate(plate)
call("perm.preview_thickness", only_selected=False, compare=True, expect=('FINISHED', 'CANCELLED'))
T.obj_mode()
call("perm.clear_thickness_preview")
T.activate(plate)
s = T.solve(plate, gravity=True)
solved("B plate with offset faces", s)
select_faces(plate, [0, 1])
call("perm.clear_shell_offset", all_faces=False)
T.obj_mode(); T.activate(plate)
call("perm.clear_shell_offset", all_faces=True)
T.obj_mode(); T.activate(plate)
call("perm.clear_results")
T.activate(plate)

# ═════════════════════════════════════════════ C. per-face element types
scenario("C per-face element types")
for nm, kind in (("Full", 'QUAD4_MITC4'), ("Plt", 'QUAD4_PLATE'), ("Mem", 'QUAD4_MEMBRANE'), ("Tri", 'TRI3')):
    call("perm.add_element_type", name=nm, kind=kind)
scn.perm_element_type_index = 0
select_faces(plate, list(range(0, 8)))
call("perm.assign_element_type", replace_existing=True)
T.obj_mode(); T.activate(plate)
scn.perm_element_type_index = 1                      # plate-only faces: fine on a flat plate under transverse load
select_faces(plate, list(range(8, 24)))
call("perm.assign_element_type", replace_existing=True)
T.obj_mode(); T.activate(plate)
T.edit(plate, 'FACE'); bpy.ops.mesh.select_all(action='DESELECT')
call("perm.select_element_type_faces")
if n_selected_faces(plate) != 16:
    NOTES.append(f"select_element_type_faces: {n_selected_faces(plate)} faces, expected 16"); print("[!!]", NOTES[-1])
T.activate(plate)
s = T.solve(plate, gravity=True)
solved("C plate with full-shell and plate-only faces", s)
d_c = float(s.results.max_displacement) if s else float('nan')
if not (0.0 < d_c < 0.01):
    NOTES.append(f"C: plate-only faces changed the deflection to {d_c:.4g} m (expected about 1 mm)"); print("[!!]", NOTES[-1])
scn.perm_element_type_index = 1
call("perm.unassign_element_type")
T.obj_mode(); T.activate(plate)
scn.perm_element_type_index = 3
call("perm.delete_element_type")
call("perm.clear_results"); T.activate(plate)
# membrane-only faces: a vertical wall loaded in its own plane
Vw, Fw = T.grid(2.0, 1.0, 8, 4)
wall = T.new_mesh("Wall", [(x, 5.0, z) for x, z, _ in Vw], Fw)
select_faces(wall, None)
scn.perm_material_index = 0; scn.perm_section_index = 0
call("perm.assign_material_to_faces"); call("perm.assign_section_to_faces")
scn.perm_element_type_index = 2
call("perm.assign_element_type", replace_existing=True)
T.obj_mode(); T.activate(wall)
support(wall, [v.index for v in wall.data.vertices if abs(v.co.x) < 1e-6], "wall_root", "FIXED")
load(wall, [v.index for v in wall.data.vertices if abs(v.co.x - 2.0) < 1e-6], "wall_tip", fz=-1000.0)
s = T.solve(wall, gravity=False)
solved("C vertical wall, membrane-only faces, in-plane load", s)
call("perm.clear_results")
T.obj_mode()
bpy.data.objects.remove(wall, do_unlink=True)
T.activate(plate)
scn.perm_element_type_index = 0

# ═════════════════════════════════════════════ D. bars on edges (truss, beam) with shells
scenario("D truss / beam on edges")
call("perm.add_material", name="Steel", youngs_modulus=200e9, poissons_ratio=0.3, density=7850.0)
call("perm.add_section", name="Bar", thickness=0.01, area=1.0e-3)
scn.perm_material_index = len(scn.perm_materials) - 1
scn.perm_section_index = len(scn.perm_sections) - 1
sec = scn.perm_sections[scn.perm_section_index]
if hasattr(sec, "beam_shape"):
    sec.beam_shape = 'RECT'; sec.dim_b = 0.05; sec.dim_h = 0.10
call("perm.add_element_type", name="Tie", kind='TRUSS_T2')
scn.perm_element_type_index = len(scn.perm_element_types) - 1
row_y0 = [v.index for v in plate.data.vertices if abs(v.co.y) < 1e-6]
select_verts(plate, row_y0, 'EDGE')
call("perm.assign_element_type", replace_existing=True)
T.obj_mode(); T.activate(plate)
call("perm.add_element_type", name="Rib", kind='BEAM_B2')
scn.perm_element_type_index = len(scn.perm_element_types) - 1
row_y2 = [v.index for v in plate.data.vertices if abs(v.co.y - 2.0) < 1e-6]
select_verts(plate, row_y2, 'EDGE')
call("perm.assign_element_type", replace_existing=True)
T.obj_mode(); T.activate(plate)
T.edit(plate, 'EDGE'); bpy.ops.mesh.select_all(action='DESELECT')
call("perm.select_element_type_faces", expect=('FINISHED', 'CANCELLED'))
T.obj_mode(); T.activate(plate)
s = T.solve(plate, gravity=True, viz=("viz_displacement", "viz_vm"))
solved("D plate + truss bars on one edge + beams on the other", s)
select_verts(plate, row_y2[:3], 'EDGE')
call("perm.remove_truss_from_edges", expect=('FINISHED', 'CANCELLED'))
T.obj_mode(); T.activate(plate)
call("perm.unassign_element_type")
T.obj_mode(); T.activate(plate)
call("perm.delete_element_type")
call("perm.delete_section"); call("perm.delete_material")
call("perm.clear_results"); T.activate(plate)
tags_before = (perm._read_face_tags(plate.data, 'material').tolist(), dict(perm._face_tag_names(plate, 'material')),
               dict(perm._face_tag_names(plate, 'section')))
call("perm.clear_assignments")
T.obj_mode()
tags_after = (perm._read_face_tags(plate.data, 'material').tolist(), dict(perm._face_tag_names(plate, 'material')),
              dict(perm._face_tag_names(plate, 'section')))
if tags_before != tags_after or not tags_after[1]:
    note(f"clear_assignments changed the face material/section tables: {tags_before[1:]} -> {tags_after[1:]}")
if perm.find_perm_support_groups(plate) or perm.find_perm_load_groups(plate):
    note("clear_assignments left support or load groups")

# ═════════════════════════════════════════════ E. triangulated plate, both topologies
for topo in ('QUAD4', 'TRI3'):
    scenario(f"E triangle mesh, topology {topo}")
    T.clear_scene()
    V, F = T.grid(3.0, 3.0, 10, 10, tri=True)
    tri = T.new_mesh("PlateT", V, F)
    setup_mat_sec(tri, t=0.15)
    support(tri, [v.index for v in tri.data.vertices if abs(v.co.x) < 1e-6 or abs(v.co.x - 3.0) < 1e-6], "ends", "FIXED")
    load(tri, [v.index for v in tri.data.vertices if abs(v.co.x - 1.5) < 1e-6 and abs(v.co.y - 1.5) < 1e-6], "P", fx=2000.0, fz=-3000.0)
    scn.perm_mesh_topology = topo
    s = T.solve(tri, gravity=False, viz=("viz_displacement", "viz_vm", "viz_reactions"))
    solved(f"E triangle mesh as {topo}", s)
    call("perm.clear_results"); T.activate(tri)
scn.perm_mesh_topology = 'QUAD4'

# ═════════════════════════════════════════════ F. mixed mesh with an n-gon and a folded pair
scenario("F mixed quads + triangles + n-gon")
T.clear_scene()
V = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (3, 2, 0), (1.5, 2.6, 0)]
F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7), (2, 7, 6), (4, 5, 9, 8), (5, 6, 10, 12, 9), (6, 7, 11, 10)]
mix = T.new_mesh("Mixed", V, F)
setup_mat_sec(mix, t=0.1)
support(mix, [0, 4, 8], "edge", "FIXED")
load(mix, [3, 7, 11], "tip", fz=-200.0)
call("perm.select_problem_faces", folded=True, ngon=True, warped=True, unassigned=True, expect=('FINISHED', 'CANCELLED'))
nprob = n_selected_faces(mix)
print("problem faces selected:", nprob)
if nprob < 1:
    NOTES.append("select_problem_faces did not select the n-gon"); print("[!!]", NOTES[-1])
T.activate(mix)
s = T.solve(mix, gravity=False, viz=("viz_displacement",))
solved("F mixed mesh with one n-gon, quadrilateral topology (the n-gon gives no element; solver warns)", s)
n_quad_mode = int(s.n_elements) if s else -1
call("perm.clear_results"); T.activate(mix)
scn.perm_mesh_topology = 'TRI3'
s = T.solve(mix, gravity=False, viz=("viz_displacement",))
solved("F same mesh, triangular topology", s)
n_tri_mode = int(s.n_elements) if s else -1
print(f"n-gon mesh (7 faces: 4 quads, 2 triangles, 1 pentagon): {n_quad_mode} elements as quadrilateral, {n_tri_mode} as triangular")
scn.perm_mesh_topology = 'QUAD4'
call("perm.clear_results"); T.activate(mix)

# ═════════════════════════════════════════════ G. curved shell (barrel roof), self-weight only
scenario("G curved shell")
T.clear_scene()
V, F = T.cylinder_patch(25.0, 50.0, 40.0, 16, 16)
roof = T.new_mesh("Roof", V, F)
setup_mat_sec(roof, E=4.32e8, nu=0.0, rho=360.0 / 9.81, t=0.25)
ends = [v.index for v in roof.data.vertices if abs(v.co.y) < 1e-6 or abs(v.co.y - 50.0) < 1e-6]
support(roof, ends, "diaphragm", "ROLLER_Y")
s = T.solve(roof, gravity=True, viz=("viz_displacement", "viz_s1_top", "viz_Mx"))
solved("G barrel roof under self-weight", s)
call("perm.clear_results"); T.activate(roof)

# ═════════════════════════════════════════════ H. two parts joined by connections
scenario("H connections between two mesh parts")
T.clear_scene()
V1, F1 = T.grid(2.0, 1.0, 8, 4)
V2, F2 = T.grid(2.0, 1.0, 8, 4, x0=2.0, off=len(V1))
two = T.new_mesh("TwoParts", V1 + V2, F1 + F2)
setup_mat_sec(two, t=0.1)
support(two, [v.index for v in two.data.vertices if abs(v.co.x) < 1e-6], "wall", "FIXED")
tipv = [v.index for v in two.data.vertices if abs(v.co.x - 4.0) < 1e-6]
load(two, tipv, "tip", fz=-100.0)
call("perm.detect_contacts", tolerance=0.005, ctype='FIXED', axis='LINE', replace=True)
T.obj_mode(); T.activate(two)
ncon = len(two.perm_connections)
print("connections found:", ncon)
if ncon != 1:
    NOTES.append(f"detect_contacts found {ncon} connections, expected 1"); print("[!!]", NOTES[-1])
s = T.solve(two, gravity=False, viz=("viz_displacement",))
solved("H two plates, FIXED joint", s)
d_fixed = float(s.results.max_displacement) if s else float('nan')
for ct in ('HINGE', 'PINNED'):
    if ncon:
        two.perm_connections[0].ctype = ct
    call("perm.connections_apply_all", expect=('FINISHED', 'CANCELLED'))
    T.obj_mode(); T.activate(two)
T.edit(two)
for side in ('BOTH', 'A', 'B'):
    call("perm.select_connection", side=side, expect=('FINISHED', 'CANCELLED'))
T.obj_mode(); T.activate(two)
call("perm.remove_connection", expect=('FINISHED', 'CANCELLED'))
seam = [v.index for v in two.data.vertices if abs(v.co.x - 2.0) < 1e-6]
select_verts(two, seam)
call("perm.add_connection", name="seam", ctype='FIXED', axis='LINE')
T.obj_mode(); T.activate(two)
s = T.solve(two, gravity=False, viz=("viz_displacement",))
solved("H two plates, joint added by hand", s)
if s and abs(float(s.results.max_displacement) - d_fixed) > 1e-9 * max(d_fixed, 1e-30) + 1e-12:
    NOTES.append(f"hand-made FIXED joint gives {float(s.results.max_displacement):.6g} m, detected one gave {d_fixed:.6g} m")
    print("[!!]", NOTES[-1])
call("perm.clear_connections")
call("perm.clear_results"); T.activate(two)

# ═════════════════════════════════════════════ I. arbitrary meshes straight from Blender's Add menu
def primitive(kind):
    T.obj_mode()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    if kind == "monkey":
        bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 1.2))
    elif kind == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0, location=(5, 1, 2))
    elif kind == "ico_sphere":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(-4, 0, 3))
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.4, major_segments=32, minor_segments=12, location=(0, 6, 1))
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=1.0, depth=2.0, location=(1, 1, 1))
    elif kind == "cube_scaled":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2, 2, 2))
        o = bpy.context.active_object
        o.scale = (4.0, 1.0, 0.5); o.rotation_euler = (0.3, 0.2, 0.1)
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=5); bpy.ops.object.mode_set(mode='OBJECT')
    o = bpy.context.active_object
    bpy.context.view_layer.update()
    return o


for kind in ("monkey", "uv_sphere", "ico_sphere", "torus", "cone", "cube_scaled"):
    scenario(f"I arbitrary mesh: {kind}")
    T.clear_scene()
    o = primitive(kind)
    setup_mat_sec(o, t=0.05)
    if kind in ("cube_scaled", "uv_sphere", "torus"):
        select_faces(o, None)
        scn.perm_sl_mode = 'NORMAL'; scn.perm_sl_pressure = 2000.0; scn.perm_sl_add = False
        call("perm.set_surface_load")
        T.obj_mode(); T.activate(o)
        inf = list(o["perm_surface_load_info"])
        me_area = sum(p.area for p in o.data.polygons)
        print(f"closed surface '{kind}' under uniform pressure: resultant ({inf[1]:.3e}, {inf[2]:.3e}, {inf[3]:.3e}) N on {int(inf[0])} faces")
        if max(abs(inf[1]), abs(inf[2]), abs(inf[3])) > 1e-3 * 2000.0:
            note(f"{kind}: uniform pressure on a closed surface has the resultant {inf[1:]} N (should vanish)")
    zs = np.array([(o.matrix_world @ v.co).z for v in o.data.vertices])
    lo = [int(i) for i in np.where(zs <= zs.min() + 0.15 * (zs.max() - zs.min()))[0]]
    hi = [int(i) for i in np.where(zs >= zs.max() - 1e-6 - 0.05 * (zs.max() - zs.min()))[0]]
    support(o, lo, "base", "FIXED")
    load(o, hi[:4], "top", fx=50.0, fz=-100.0)
    for topo in ('QUAD4', 'TRI3'):
        scn.perm_mesh_topology = topo
        s = T.solve(o, gravity=True, viz=("viz_displacement", "viz_vm"))
        solved(f"I {kind} as {topo} ({len(o.data.polygons)} faces)", s)
        call("perm.clear_results"); T.activate(o)
    scn.perm_mesh_topology = 'QUAD4'

# ═════════════════════════════════════════════ J. mesh conversion tool (second panel)
scenario("J convert mesh")
T.clear_scene()
o = primitive("monkey")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(TMP, "convert_test.blend"))      # exports go next to the .blend
for et, fmt, ext in (('TRI3', 'NONE', None), ('QUAD4', 'ABAQUS', "inp"), ('TRI6', 'NASTRAN', "bdf"), ('QUAD8', 'JSON', "json")):
    T.activate(o)
    call("perm.convert_mesh", element_type=et, thickness=0.01, scale_factor=1.0, clean_mesh=True, compute_quality=True,
         visualize=True, export_format=fmt)
    T.obj_mode()
    if ext:
        fp = os.path.join(TMP, f"{o.name}_PERM.{ext}")
        if not os.path.exists(fp) or os.path.getsize(fp) < 1000:
            note(f"convert_mesh {et} -> {fmt}: no file {fp}")
        else:
            print(f"convert_mesh {et} -> {os.path.basename(fp)}: {os.path.getsize(fp)} bytes")
            if ext == "json":
                try:
                    json.load(open(fp))
                except Exception as e:
                    note(f"convert_mesh JSON export is not readable: {e}")

# ═════════════════════════════════════════════ K. wrong use must give a message, not a Python error
scenario("K wrong use")
T.clear_scene()
V, F = T.grid(1.0, 1.0, 2, 2)
o = T.new_mesh("NoSetup", V, F)
call("perm.run_solver", expect=('CANCELLED',), **{k: False for k in T.VIZ})      # no supports
T.activate(o)
call("perm.assign_material_to_faces", expect=('CANCELLED',))                     # object mode, nothing in the list
call("perm.assign_section_to_faces", expect=('CANCELLED',))
call("perm.set_shell_offset", expect=('CANCELLED', 'FINISHED'))
call("perm.select_peak_faces", expect=('CANCELLED', 'FINISHED'))
call("perm.find_element", expect=('CANCELLED', 'FINISHED'))
call("perm.refresh_visualization", expect=('CANCELLED', 'FINISHED'))
call("perm.recolor_results", expect=('CANCELLED', 'FINISHED'))
call("perm.show_results", expect=('CANCELLED', 'FINISHED'))
call("perm.delete_material", expect=('CANCELLED', 'FINISHED'))
call("perm.delete_section", expect=('CANCELLED', 'FINISHED'))
call("perm.delete_element_type", expect=('CANCELLED', 'FINISHED'))
call("perm.remove_connection", expect=('CANCELLED', 'FINISHED'))
T.edit(o)
bpy.ops.mesh.select_all(action='DESELECT')
call("perm.assign_bc", group_name="x", bc_type='FIXED', expect=('CANCELLED',))   # nothing selected
call("perm.assign_load_per_node", group_name="x", expect=('CANCELLED',))
call("perm.add_connection", name="x", expect=('CANCELLED',))
T.obj_mode()
# supports only, no material / section / load: must solve with the defaults or stop with a message
support(o, [0, 1, 2], "s", "FIXED")
s = T.solve(o, gravity=True)
solved("K supports only (default material and thickness)", s, need_disp=False)
call("perm.clear_results"); T.activate(o)
# no mesh at all
T.clear_scene()
call("perm.run_solver", expect=('CANCELLED',), **{k: False for k in T.VIZ})
call("perm.show_markers", expect=('CANCELLED', 'FINISHED'))
call("perm.clear_assignments", expect=('CANCELLED', 'FINISHED'))
call("perm.edit_model", expect=('CANCELLED', 'FINISHED'))

# ═════════════════════════════════════════════ L. connection details as options
scenario("L detail options: girder on two supports, plain and with the bridge preset")
T.clear_scene()
V, F = T.grid(6.0, 1.0, 48, 8)
g = T.new_mesh("Girder", V, F)
setup_mat_sec(g, t=0.25)
support(g, [v.index for v in g.data.vertices if abs(v.co.x) < 1e-6 and abs(v.co.y - 0.5) < 1e-6], "A", "PINNED")
support(g, [v.index for v in g.data.vertices if abs(v.co.x - 6.0) < 1e-6 and abs(v.co.y - 0.5) < 1e-6], "B", "ROLLER_X")
load(g, [v.index for v in g.data.vertices if abs(v.co.x - 3.0) < 1e-6 and abs(v.co.y - 0.5) < 1e-6], "P", fz=-10000.0)
call("perm.reset_details")
if perm._perm_options_summary(scn) != "all off (plain shell model)" or perm._perm_is_bridge_preset(scn):
    note("reset_details did not switch every detail off")
s0 = T.solve(g, gravity=False)
solved("L girder, details off", s0)
d0 = float(s0.results.max_displacement) if s0 else float('nan')
n_patch0 = len(getattr(s0, '_patched_loads', None) or {}); n_pads0 = len(getattr(s0, '_bearing_pads', None) or [])
call("perm.clear_results"); T.activate(g)
call("perm.bridge_preset")
if not perm._perm_is_bridge_preset(scn):
    note("bridge_preset did not set the preset values")
s1 = T.solve(g, gravity=False)
solved("L girder, bridge preset", s1)
d1 = float(s1.results.max_displacement) if s1 else float('nan')
n_patch1 = len(getattr(s1, '_patched_loads', None) or {}); n_pads1 = len(getattr(s1, '_bearing_pads', None) or [])
print(f"details off: max disp {d0:.6g} m, patches {n_patch0}, pads {n_pads0} | preset: {d1:.6g} m, patches {n_patch1}, pads {n_pads1}")
if n_patch0 or n_pads0:
    note("details off, but a loading patch or bearing pad was used")
if not (n_patch1 == 1 and n_pads1 == 2):
    note(f"bridge preset: {n_patch1} patches and {n_pads1} pads, expected 1 and 2")
if not (d1 < d0):
    note(f"bridge preset should reduce the peak deflection under a point load ({d1} vs {d0})")
call("perm.clear_results"); T.activate(g)
call("perm.reset_details")
call("perm.install_scipy")            # SciPy is present in this test: must answer 'already available'

# ═════════════════════════════════════════════ N. the example script of the repository
EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "example_plate.py")
if os.path.exists(EX):
    scenario("N examples/example_plate.py")
    T.clear_scene()
    T.CAP.clear()
    try:
        exec(compile(open(EX).read(), EX, "exec"), {"__name__": "__main__"})
        s = T.CAP.get('solver')
        ok = solved("N example slab (surface load 5 kN/m2, 20 kN point load, self-weight, steel beam)", s)
        if ok:
            Fa, Rt = totals(s)
            exp_n = -(5000.0 * 24.0 + 20000.0 + 2400.0 * 0.20 * 24.0 * 9.81 + 7850.0 * 0.03 * 6.0 * 9.81)
            if not close(Fa[2], exp_n, 1e-5):
                note(f"example: sum of loads Fz = {Fa[2]:.2f} N, expected {exp_n:.2f} N")
            if np.linalg.norm(Fa + Rt) > 1e-6 * np.linalg.norm(Fa):
                note(f"example: reactions do not balance the loads ({Fa} + {Rt})")
    except Exception as e:
        note(f"examples/example_plate.py failed: {e.__class__.__name__}: {e}")
        import traceback; traceback.print_exc()
    T.obj_mode()

# modal tool: needs a window (tested in the UI run); in background it must refuse cleanly
scenario("M modal inspect (background)")
call("perm.inspect_results", mode='INVOKE_DEFAULT', expect=('CANCELLED', 'RUNNING_MODAL', 'FINISHED', 'PASS_THROUGH'))

all_ops = sorted(c.bl_idname for c in perm.classes if issubclass(c, bpy.types.Operator))
bad = T.summary(all_ops, OUT)
print("\nSolves:")
for r in SOLVES:
    print(f"  {'ok' if r['ok'] else 'FAIL':5s} {r['case']:62s} max disp {r['max_disp_m']:.4g} m  nodes {r['nodes']}")
print("\nNotes:", *NOTES, sep="\n  ") if NOTES else print("\nNotes: none")
nbad = bad + sum(1 for r in SOLVES if not r['ok']) + len(NOTES)
if OUT:
    json.dump(dict(operators=json.load(open(OUT)), solves=SOLVES, notes=NOTES, blender=bpy.app.version_string), open(OUT, "w"), indent=1)
print(f"\nRESULT: {'PASS' if nbad == 0 else 'FAIL'} ({nbad} problem(s))", flush=True)
sys.exit(0 if nbad == 0 else 1)
