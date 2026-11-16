"""PERM example: a 6 m x 4 m concrete slab, simply supported on its four edges, with a steel beam along its
middle line, under self-weight, a surface load of 5 kN/m2 and a 20 kN point load.

Run it in Blender with PERM enabled and SciPy installed:
    Scripting workspace > Text > Open > this file > Run Script
or from a terminal:
    blender --python examples/example_plate.py

It uses only the commands of the PERM panel, in the order you would click them, so it also shows how to drive
PERM from a script. After it has run: View3D > Sidebar (N) > PERM.
"""
import bpy, bmesh

SOLVE = True                    # False: build and assign only, then click RUN ANALYSIS yourself
LX, LY, NX, NY = 6.0, 4.0, 24, 16

if not hasattr(bpy.ops, "perm") or not hasattr(bpy.types.Scene, "perm_materials"):
    raise RuntimeError("Enable the PERM add-on first (Edit > Preferences > Add-ons > Install from Disk > perm.py)")
scn = bpy.context.scene

# ---- the mesh: any mesh object works; here a grid of quads, 0.25 m elements
if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
verts = [(LX * i / NX, LY * j / NY, 0.0) for i in range(NX + 1) for j in range(NY + 1)]
vid = lambda i, j: i * (NY + 1) + j
faces = [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)) for i in range(NX) for j in range(NY)]
me = bpy.data.meshes.new("PERM example slab")
me.from_pydata(verts, [], faces); me.update()
obj = bpy.data.objects.new("PERM example slab", me)
bpy.context.collection.objects.link(obj)
for o in bpy.context.view_layer.objects:
    o.select_set(False)
bpy.context.view_layer.objects.active = obj; obj.select_set(True)


def select(kind, test):
    """Edit Mode selection of the vertices ('VERT'), edges ('EDGE') or faces ('FACE') that pass test(co)."""
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type=kind)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    if kind == 'FACE':
        for f in bm.faces:
            f.select_set(test(f.calc_center_median()))
    elif kind == 'EDGE':
        for e in bm.edges:
            e.select_set(test(e.verts[0].co) and test(e.verts[1].co))
    else:
        for v in bm.verts:
            v.select_set(test(v.co))
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


near = lambda a, b: abs(a - b) < 1e-6

# ---- 1. materials (E in Pa, density in kg/m3) and 2. cross-sections (thickness in m)
bpy.ops.perm.add_material(name="Concrete", youngs_modulus=30e9, poissons_ratio=0.2, density=2400.0)
bpy.ops.perm.add_material(name="Steel", youngs_modulus=200e9, poissons_ratio=0.3, density=7850.0)
bpy.ops.perm.add_section(name="Slab 200", thickness=0.20)
bpy.ops.perm.add_section(name="Steel beam", thickness=0.01, area=0.03)
beam = scn.perm_sections[len(scn.perm_sections) - 1]
beam.beam_shape = 'RECT'; beam.dim_b = 0.10; beam.dim_h = 0.30          # 100 x 300 mm solid steel section

mat = {m.name: i for i, m in enumerate(scn.perm_materials)}
sec = {s.name: i for i, s in enumerate(scn.perm_sections)}
scn.perm_material_index = mat["Concrete"]; scn.perm_section_index = sec["Slab 200"]
select('FACE', lambda c: True)
bpy.ops.perm.assign_material_to_faces()
bpy.ops.perm.assign_section_to_faces()

# ---- 3. element types: a beam on the mesh edges along the middle line y = LY / 2 (active material + section)
bpy.ops.perm.add_element_type(name="Rib", kind='BEAM_B2')
scn.perm_element_type_index = len(scn.perm_element_types) - 1
scn.perm_material_index = mat["Steel"]; scn.perm_section_index = sec["Steel beam"]
select('EDGE', lambda c: near(c.y, LY / 2))
bpy.ops.perm.assign_element_type(replace_existing=True)

# ---- 4. restraints: the four edges held vertically (CUSTOM: uz), two corners hold the slab in its plane
select('VERT', lambda c: near(c.x, 0) or near(c.x, LX) or near(c.y, 0) or near(c.y, LY))
bpy.ops.perm.assign_bc(group_name="edges", bc_type='CUSTOM', fix_ux=False, fix_uy=False, fix_uz=True,
                       fix_rx=False, fix_ry=False, fix_rz=False)
select('VERT', lambda c: near(c.x, 0) and near(c.y, 0))
bpy.ops.perm.assign_bc(group_name="corner A", bc_type='PINNED')
select('VERT', lambda c: near(c.x, LX) and near(c.y, 0))
bpy.ops.perm.assign_bc(group_name="corner B", bc_type='ROLLER_X')

# ---- 5. loads: a 20 kN point load, 5 kN/m2 on the whole slab, self-weight
select('VERT', lambda c: near(c.x, LX / 4) and near(c.y, LY / 4))
bpy.ops.perm.assign_load_per_node(group_name="P 20 kN", fx=0.0, fy=0.0, fz=-20000.0)
select('FACE', lambda c: True)
scn.perm_sl_mode = 'GLOBAL'; scn.perm_sl_vector = (0.0, 0.0, -5000.0); scn.perm_sl_add = False
bpy.ops.perm.set_surface_load()
bpy.ops.object.mode_set(mode='OBJECT')
scn.perm_apply_gravity = True
scn.perm_gravity_x, scn.perm_gravity_y, scn.perm_gravity_z = 0.0, 0.0, -9.81

# ---- 8. analysis: the connection details of the bridge study stay off
bpy.ops.perm.reset_details()
scn.perm_case_name = "example slab"
if SOLVE:
    bpy.ops.perm.run_solver(viz_displacement=True, viz_vm=True, viz_vm_top=False, viz_vm_bot=False,
                            viz_s1_top=False, viz_s2_top=False, viz_s1_bot=True, viz_s2_bot=False,
                            viz_sx=False, viz_sy=False, viz_txy=False, viz_sx_top=False, viz_sy_top=False,
                            viz_txy_top=False, viz_sx_bot=False, viz_sy_bot=False, viz_txy_bot=False,
                            viz_Mx=True, viz_My=True, viz_Mxy=False, viz_reactions=True)
    print("PERM example: report written to", scn.get("perm_last_report"))
