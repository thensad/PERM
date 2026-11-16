"""Helpers for the PERM tests (run inside Blender: blender -b --factory-startup --python tests/<test>.py -- --addon perm.py).

load_addon(path)                 import the single-file add-on from a path and register it
new_mesh / grid / cylinder ...   small generic surface meshes
select_verts / select_faces      Edit Mode selections, as a user would make them
call(idname, **props)            run one PERM operator (EXEC_DEFAULT), record the outcome per operator
"""
import os, sys, importlib.util, traceback, math, json
import numpy as np
import bpy, bmesh

perm = None
LOG = {}          # operator idname -> list of (scenario, status, message)
SCENARIO = ["-"]
CAP = {}


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default=None):
    a = argv_after_dashes()
    return a[a.index(name) + 1] if name in a else default


def load_addon(path=None, site=None):
    """Import the add-on file and register it. `site`: extra site-packages folder that holds SciPy."""
    global perm
    site = site or opt("--site")
    if site and site not in sys.path:
        sys.path.insert(0, site)
    path = path or opt("--addon")
    spec = importlib.util.spec_from_file_location("perm_addon_under_test", path)
    perm = importlib.util.module_from_spec(spec); sys.modules[spec.name] = perm
    spec.loader.exec_module(perm)
    perm.register()
    oi = perm.PERMSolver.__init__

    def spy(self, c):
        CAP['solver'] = self
        oi(self, c)
    perm.PERMSolver.__init__ = spy
    return perm


def scenario(name):
    SCENARIO[0] = name
    print(f"\n===== scenario: {name} =====", flush=True)


def call(idname, mode='EXEC_DEFAULT', expect=('FINISHED',), **props):
    """Run bpy.ops.<idname>; record FINISHED / CANCELLED / exception. Returns the result set (or None)."""
    cat, name = idname.split(".")
    op = getattr(getattr(bpy.ops, cat), name)
    try:
        r = op(mode, **props)
        status = next(iter(r)) if r else "NONE"
        msg = ""
    except RuntimeError as e:
        # an operator that reports {'ERROR'} and cancels reaches Python as RuntimeError("Error: <message>");
        # a Python error INSIDE the operator carries its traceback in the text
        txt = str(e).strip()
        if "Traceback" in txt or "Python:" in txt or not txt.startswith("Error:"):
            last = txt.splitlines()[-1] if txt else ""
            LOG.setdefault(idname, []).append((SCENARIO[0], "ERROR", last))
            print(f"[!!] {idname} raised a Python error: {last}", flush=True)
            print(txt, flush=True)
            return None
        status, msg = "CANCELLED", txt.splitlines()[0]
        r = {'CANCELLED'}
    except Exception as e:
        msg = f"{e.__class__.__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}"
        LOG.setdefault(idname, []).append((SCENARIO[0], "ERROR", msg))
        print(f"[!!] {idname} raised {msg}", flush=True)
        traceback.print_exc()
        return None
    ok = status in expect
    LOG.setdefault(idname, []).append((SCENARIO[0], status if ok else f"UNEXPECTED {status}", msg))
    if not ok:
        print(f"[!!] {idname} -> {status} {msg} (expected {expect})", flush=True)
    return r


def summary(all_ops=None, out_json=None):
    """Print one line per operator; return the number of operators that failed or were never run."""
    all_ops = all_ops or sorted(c.bl_idname for c in perm.classes if issubclass(c, bpy.types.Operator))
    bad = 0; rows = []
    print("\n================ PERM commands: outcome per operator ================")
    for idn in all_ops:
        runs = LOG.get(idn, [])
        n_ok = sum(1 for _, s, _ in runs if s in ("FINISHED", "CANCELLED", "RUNNING_MODAL", "PASS_THROUGH", "INTERFACE"))
        errs = [(sc, s, m) for sc, s, m in runs if s == "ERROR" or s.startswith("UNEXPECTED")]
        state = "NOT RUN" if not runs else ("FAIL" if errs else "ok")
        bad += state != "ok"
        rows.append(dict(operator=idn, state=state, runs=len(runs), problems=[f"{sc}: {s} {m}" for sc, s, m in errs]))
        print(f"  {state:8s} {idn:36s} runs {len(runs):3d}" + ("" if not errs else "   <- " + "; ".join(f"{sc}: {s} {m}" for sc, s, m in errs[:3])))
    print(f"================ {len(all_ops) - bad} of {len(all_ops)} operators ok ================", flush=True)
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1)
    return bad


# ───────────────────────────── scene / meshes
def clear_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)
    scn = bpy.context.scene
    for k in ("perm_materials", "perm_sections", "perm_element_types"):
        getattr(scn, k).clear()


def new_mesh(name, verts, faces, edges=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [tuple(e) for e in edges], [tuple(f) for f in faces])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    activate(obj)
    return obj


def activate(obj):
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def grid(lx, ly, nx, ny, z=0.0, x0=0.0, y0=0.0, tri=False, off=0):
    V = [(x0 + lx * i / nx, y0 + ly * j / ny, z) for i in range(nx + 1) for j in range(ny + 1)]
    vid = lambda i, j: off + i * (ny + 1) + j
    F = []
    for i in range(nx):
        for j in range(ny):
            q = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))
            F += [(q[0], q[1], q[2]), (q[0], q[2], q[3])] if tri else [q]
    return V, F


def cylinder_patch(R, L, phi_deg, nphi, nx, tri=False):
    """Cylindrical roof: axis along Y, apex at z = R; angle from -phi to +phi about the vertical."""
    V = []
    for i in range(nphi + 1):
        a = math.radians(-phi_deg + 2 * phi_deg * i / nphi)
        for j in range(nx + 1):
            V.append((R * math.sin(a), L * j / nx, R * math.cos(a)))
    vid = lambda i, j: i * (nx + 1) + j
    F = []
    for i in range(nphi):
        for j in range(nx):
            q = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))
            F += [(q[0], q[1], q[2]), (q[0], q[2], q[3])] if tri else [q]
    return V, F


def edit(obj, sel='VERT'):
    activate(obj) if bpy.context.view_layer.objects.active is not obj else None
    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type=sel)


def obj_mode():
    o = bpy.context.active_object
    if o and o.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select_verts(obj, idx, sel='VERT'):
    edit(obj, sel)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data); bm.verts.ensure_lookup_table()
    s = set(int(i) for i in idx)
    for v in bm.verts:
        v.select = v.index in s
    bm.select_flush(True)
    bmesh.update_edit_mesh(obj.data)


def select_faces(obj, fidx=None):
    edit(obj, 'FACE')
    bpy.ops.mesh.select_all(action='DESELECT' if fidx is not None else 'SELECT')
    if fidx is None:
        return
    bm = bmesh.from_edit_mesh(obj.data); bm.faces.ensure_lookup_table()
    s = set(int(i) for i in fidx)
    for f in bm.faces:
        f.select_set(f.index in s)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def verts_where(obj, fn):
    return [v.index for v in obj.data.vertices if fn(obj.matrix_world @ v.co)]


def faces_where(obj, fn):
    return [p.index for p in obj.data.polygons if fn(obj.matrix_world @ p.center)]


VIZ = ["viz_displacement", "viz_vm", "viz_vm_top", "viz_vm_bot", "viz_s1_top", "viz_s2_top", "viz_s1_bot", "viz_s2_bot",
       "viz_sx", "viz_sy", "viz_txy", "viz_sx_top", "viz_sy_top", "viz_txy_top", "viz_sx_bot", "viz_sy_bot", "viz_txy_bot",
       "viz_Mx", "viz_My", "viz_Mxy", "viz_reactions"]


def solve(obj, gravity=False, viz=(), record=True, **kw):
    """Run the solver on obj, as the RUN ANALYSIS button does: self-weight and mesh topology are read from the
    panel (scene). Returns the PERMSolver instance (results in .results) or None."""
    activate(obj)
    scn = bpy.context.scene
    scn.perm_apply_gravity = bool(gravity)
    args = {k: (k in viz) for k in VIZ if hasattr(perm.PERM_OT_RunSolver, "__annotations__") and k in perm.PERM_OT_RunSolver.__annotations__}
    args.update(kw)
    CAP.clear()
    if record:
        call("perm.run_solver", **args)
    else:
        bpy.ops.perm.run_solver('EXEC_DEFAULT', **args)
    return CAP.get('solver')


def node_disp(solver, xyz, tol=1e-6):
    P = np.asarray(solver.node_coords); d = np.linalg.norm(P - np.asarray(xyz, float), axis=1)
    return solver.results.displacements[d <= tol]


CHECKS = []


def check(name, got, exp, rtol=0.02, atol=0.0):
    ok = abs(got - exp) <= max(rtol * abs(exp), atol)
    rel = float(abs(got - exp) / abs(exp)) if exp != 0 else float('nan')        # no relative error against zero
    CHECKS.append(dict(name=name, got=float(got), expected=float(exp), abs_err=float(abs(got - exp)), rel_err=rel,
                       tol=rtol, atol=atol, ok=bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {got:.6g} vs {exp:.6g} (difference {abs(got - exp):.3e}, tolerance "
          f"{max(rtol * abs(exp), atol):.3e})", flush=True)
    return ok
