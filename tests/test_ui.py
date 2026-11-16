"""PERM in Blender's interface: install the add-on from disk, draw the panel in every state, open every dialog,
run the hover inspector. Needs a display (on a server: Xvfb) and --enable-event-simulate.

    blender --enable-event-simulate --python-exit-code 1 --python tests/test_ui.py -- --addon perm.py [--site <SciPy folder>] [--shots <folder>]

Blender's console output is the record: the script prints 'UI RESULT: PASS' and quits with exit code 0, or prints
what failed. A Python error inside a panel or dialog draw is printed by Blender as a traceback; run_ui_tests.sh
greps the log for it.
"""
import os, sys, math, traceback, tempfile, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import perm_testlib as T

ARGS = T.argv_after_dashes()
SHOTS = T.opt("--shots")
SITE = T.opt("--site")
ADDON = os.path.abspath(T.opt("--addon"))
FAIL = []
STATE = {"i": 0}
if SITE and SITE not in sys.path:
    sys.path.insert(0, SITE)


def fail(msg):
    FAIL.append(msg); print("UI FAIL:", msg, flush=True)


def win():
    return bpy.context.window_manager.windows[0]


def v3d():
    w = win(); scr = w.screen
    a = max((a for a in scr.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
    return w, scr, a, [r for r in a.regions if r.type == 'WINDOW'][0], [r for r in a.regions if r.type == 'UI'][0]


def ctx():
    w, scr, a, r, ui = v3d()
    return bpy.context.temp_override(window=w, screen=scr, area=a, region=r)


def redraw():
    w, scr, a, r, ui = v3d()
    for ar in scr.areas:
        ar.tag_redraw()
    try:
        with ctx():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception as e:
        print("redraw:", e)


def shot(name):
    redraw()
    if not SHOTS:
        return
    os.makedirs(SHOTS, exist_ok=True)
    path = os.path.join(SHOTS, name + ".png")
    import shutil, subprocess
    try:
        if shutil.which("import") and os.environ.get("DISPLAY"):
            # ImageMagick grabs the X display: Blender's own screenshot is black with software OpenGL (Xvfb)
            subprocess.run(["import", "-window", "root", path], check=False, timeout=30)
        else:
            with ctx():
                bpy.ops.screen.screenshot(filepath=path)
    except Exception as e:
        print("screenshot:", e)


def scroll_panel(pages):
    w, scr, a, r, ui = v3d()
    try:
        with bpy.context.temp_override(window=w, screen=scr, area=a, region=ui):
            for _ in range(abs(pages)):
                (bpy.ops.view2d.scroll_down if pages > 0 else bpy.ops.view2d.scroll_up)(page=True)
    except Exception as e:
        print("scroll:", e)


def esc():
    win().event_simulate(type='ESC', value='PRESS'); win().event_simulate(type='ESC', value='RELEASE')


def op(idname, mode='EXEC_DEFAULT', **kw):
    with ctx():
        return T.call(idname, mode=mode, **kw)


# ───────────────────────────── steps (each runs from a timer, so the interface redraws in between)
def s_install():
    p = bpy.context.preferences
    p.view.show_splash = False
    mod = os.path.splitext(os.path.basename(ADDON))[0]
    try:
        bpy.ops.preferences.addon_install(filepath=ADDON, overwrite=True)
        bpy.ops.preferences.addon_enable(module=mod)
    except Exception as e:
        fail(f"install from disk: {e.__class__.__name__}: {e}"); traceback.print_exc(); return
    import addon_utils
    en = {m.__name__ for m in addon_utils.modules() if addon_utils.check(m.__name__)[1]}
    if mod not in en:
        fail(f"add-on '{mod}' is not enabled after addon_install + addon_enable"); return
    T.perm = sys.modules[mod]
    print(f"UI: installed and enabled '{mod}' from {T.perm.__file__}; version {T.perm.PERM_VERSION}; SciPy {T.perm.HAS_SCIPY}", flush=True)
    if not hasattr(bpy.types.Scene, "perm_materials") or not hasattr(bpy.ops.perm, "run_solver"):
        fail("add-on enabled but its properties / operators are missing")
    w, scr, a, r, ui = v3d()
    a.spaces.active.show_region_ui = True
    bpy.context.scene.perm_results_dir = tempfile.mkdtemp(prefix="perm_ui_")


def s_full():
    with ctx():
        bpy.ops.screen.screen_full_area()               # the 3D viewport alone, as large as the window


def s_tab():
    w, scr, a, r, ui = v3d()
    a.spaces.active.show_region_ui = True
    try:
        ui.active_panel_category = 'PERM'
    except Exception as e:
        fail(f"could not open the PERM tab of the sidebar: {e}")
    print("UI: sidebar tab:", getattr(ui, "active_panel_category", "?"), flush=True)
    for o in list(bpy.data.objects):                     # state 1: no mesh at all
        bpy.data.objects.remove(o, do_unlink=True)
    shot("01_no_mesh")


def s_no_scipy():
    T.perm.HAS_SCIPY = False                             # state 2: the SciPy notice and the scene-units notice
    bpy.context.scene.unit_settings.scale_length = 0.001
    V, F = T.grid(4.0, 2.0, 16, 8)
    STATE["obj"] = T.new_mesh("Plate", V, F)
    with ctx():
        bpy.ops.view3d.view_all()
    shot("02_scipy_notice")


def s_model():
    bpy.context.scene.unit_settings.scale_length = 1.0
    T.perm._perm_try_import_scipy()
    if not T.perm.HAS_SCIPY:
        fail("SciPy not importable in this Blender (pass --site or install it)"); return
    scn = bpy.context.scene; o = STATE["obj"]
    op("perm.add_material", name="Concrete", youngs_modulus=30e9, poissons_ratio=0.2, density=2400.0)
    op("perm.add_material", name="Steel", youngs_modulus=200e9, poissons_ratio=0.3, density=7850.0)
    op("perm.add_section", name="Slab", thickness=0.2)
    scn.perm_sections[0].rebar_ratio = 0.005
    op("perm.add_section", name="Bar", thickness=0.01, area=1e-3)
    scn.perm_sections[1].beam_shape = 'RHS'
    scn.perm_material_index = 0; scn.perm_section_index = 0
    T.select_faces(o, None)
    op("perm.assign_material_to_faces"); op("perm.assign_section_to_faces")
    T.select_faces(o, list(range(16)))
    scn.perm_offset_half = True
    op("perm.set_shell_offset")
    T.obj_mode()
    for nm, kind in (("Shell", 'QUAD4_MITC4'), ("Rib", 'BEAM_B2')):
        op("perm.add_element_type", name=nm, kind=kind)
    scn.perm_element_type_index = 1; scn.perm_material_index = 1; scn.perm_section_index = 1
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.y) < 1e-6], 'EDGE')
    op("perm.assign_element_type", replace_existing=True)
    T.obj_mode()
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.x) < 1e-6]); op("perm.assign_bc", group_name="left", bc_type='PINNED'); T.obj_mode()
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.x - 4) < 1e-6]); op("perm.assign_bc", group_name="right", bc_type='ROLLER_X'); T.obj_mode()
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.x - 2) < 1e-6 and abs(v.co.y - 2) < 1e-6])
    op("perm.assign_bc", group_name="guide", bc_type='CUSTOM', fix_ux=False, fix_uy=True, fix_uz=False); T.obj_mode()
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.x - 2) < 1e-6 and abs(v.co.y - 1) < 1e-6])
    op("perm.assign_load_per_node", group_name="P", fz=-5000.0); T.obj_mode()
    T.select_verts(o, [v.index for v in o.data.vertices if abs(v.co.x - 3) < 1e-6])
    op("perm.assign_load_per_node", group_name="line", fx=10.0, fz=-100.0); T.obj_mode()
    T.activate(o)
    op("perm.toggle_node_list", group_name="PERM_Load_line")
    T.select_faces(o, None)
    scn.perm_sl_mode = 'NORMAL'; scn.perm_sl_pressure = 1500.0
    op("perm.set_surface_load")
    T.obj_mode(); T.activate(o)
    scn.perm_apply_gravity = True
    scn.perm_case_name = "ui_test"
    op("perm.bridge_preset")
    op("perm.reset_details")
    scn.perm_opt_show = True
    scn.perm_opt_patch_loads = True; scn.perm_opt_post_plates = True; scn.perm_opt_tie_layers = True
    scn.perm_opt_bearing_pads = True; scn.perm_opt_local_zone = 0.25
    shot("03_model_top")


def s_model_scroll1():
    scroll_panel(1); shot("04_model_scroll1")


def s_model_scroll2():
    scroll_panel(1); shot("05_model_scroll2")


def s_model_scroll3():
    scroll_panel(1); shot("06_model_scroll3")


def s_model_scroll4():
    scroll_panel(1); shot("07_model_scroll4")


def s_solve():
    scn = bpy.context.scene
    op("perm.reset_details")
    scroll_panel(-6)
    T.activate(STATE["obj"])
    with ctx():
        T.solve(STATE["obj"], gravity=True, viz=("viz_displacement", "viz_vm", "viz_s1_bot", "viz_reactions"))
    s = T.perm._perm_last_solver_for(STATE["obj"])
    if s is None or not (s.results.max_displacement > 0):
        fail("solve in the interface gave no result")
    scn.perm_find_element = 10
    op("perm.find_element", expect=('FINISHED', 'CANCELLED'))
    shot("08_results")


def s_results_scroll():
    scroll_panel(4); shot("09_results_panel")


def s_results_scroll2():
    scroll_panel(1); shot("10_results_panel2")


def s_range():
    scn = bpy.context.scene
    op("perm.show_results", expect=('FINISHED', 'CANCELLED'))
    scn.perm_viz_filter = 'CONCRETE'; scn.perm_color_range = 'MANUAL'
    try:
        scn.perm_range_max = 1.0
    except Exception as e:
        fail(f"typing a colour range: {e}")
    scn.perm_principal_colormap = 'DIVERGING'
    scn.perm_deformation_scale = 300.0
    scn.perm_result_opacity = 0.7
    scn.perm_reaction_scale = 2.0
    shot("11_filter_manual_range")
    scn.perm_viz_filter = 'ALL'; scn.perm_color_range = 'AUTO'; scn.perm_result_opacity = 1.0


# dialogs: open with INVOKE_DEFAULT (their draw code runs), look, press Esc
DIALOGS = [
    ("perm.run_solver", {}, None),
    ("perm.export_json", {}, None),
    ("perm.add_material", {}, None),
    ("perm.add_section", {}, None),
    ("perm.add_element_type", {}, None),
    ("perm.assign_bc", {"bc_type": 'CUSTOM'}, 'VERT'),
    ("perm.assign_load_per_node", {}, 'VERT'),
    ("perm.add_connection", {}, 'VERT'),
    ("perm.detect_contacts", {}, None),
    ("perm.select_problem_faces", {}, None),
    ("perm.open_per_node_editor", {"group_name": "PERM_Load_line"}, None),
    ("perm.bulk_set_per_node", {}, None),
    ("perm.convert_mesh", {}, None),
]


def make_dialog_step(idname, kw, edit):
    def step():
        o = STATE["obj"]
        T.obj_mode(); T.activate(o)
        try:
            o.hide_set(False)
        except Exception:
            pass
        if edit:
            T.select_verts(o, [0, 1, 2])
        with ctx():
            r = T.call(idname, mode='INVOKE_DEFAULT', expect=('RUNNING_MODAL', 'FINISHED', 'INTERFACE', 'CANCELLED', 'PASS_THROUGH'), **kw)
        print(f"UI: dialog {idname} -> {r}", flush=True)
        STATE["dlg"] = idname
    return step


def s_dialog_look():
    shot("dlg_" + STATE.get("dlg", "x").split(".")[1])
    esc()


def s_inspect_start():
    o = STATE["obj"]
    T.obj_mode()
    op("perm.show_results", expect=('FINISHED', 'CANCELLED'))
    with ctx():
        r = T.call("perm.inspect_results", mode='INVOKE_DEFAULT', expect=('RUNNING_MODAL',))
    print("UI: inspect ->", r, "running flag", bpy.context.scene.perm_inspect_running, flush=True)
    if not bpy.context.scene.perm_inspect_running:
        fail("inspect_results did not start")
    w, scr, a, r3, ui = v3d()
    STATE["mx"], STATE["my"] = r3.x + r3.width // 2, r3.y + r3.height // 2


def s_inspect_move():
    w = win()
    for k in range(6):
        w.event_simulate(type='MOUSEMOVE', value='NOTHING', x=STATE["mx"] + 12 * k, y=STATE["my"] + 5 * k)


def s_inspect_look():
    shot("12_inspect_hover")


def s_inspect_stop():
    esc()


def s_inspect_check():
    if bpy.context.scene.perm_inspect_running:
        with ctx():
            T.call("perm.inspect_results", mode='INVOKE_DEFAULT', expect=('RUNNING_MODAL', 'FINISHED', 'CANCELLED'))
    print("UI: inspector running after stop:", bpy.context.scene.perm_inspect_running, flush=True)
    if bpy.context.scene.perm_inspect_running:
        fail("inspect_results did not stop")


def s_cleanup():
    T.obj_mode()
    op("perm.clear_find_element")
    op("perm.clear_results")
    T.activate(STATE["obj"])
    op("perm.clear_assignments")
    mod = os.path.splitext(os.path.basename(ADDON))[0]
    try:
        bpy.ops.preferences.addon_disable(module=mod)
        if hasattr(bpy.types.Scene, "perm_materials") or hasattr(bpy.types.Scene, "perm_opt_patch_loads"):
            fail("add-on disabled but scene properties are still registered")
        bpy.ops.preferences.addon_enable(module=mod)
        if not hasattr(bpy.types.Scene, "perm_materials"):
            fail("add-on did not come back after disable + enable")
        print("UI: disable + enable ok", flush=True)
    except Exception as e:
        fail(f"disable / enable: {e.__class__.__name__}: {e}"); traceback.print_exc()


def s_done():
    bad = [(k, v) for k, v in T.LOG.items() if any(s == "ERROR" or s.startswith("UNEXPECTED") for _, s, _ in v)]
    for k, v in bad:
        fail(f"{k}: {[x for x in v if x[1] == 'ERROR' or x[1].startswith('UNEXPECTED')]}")
    print(f"UI: operators called through the interface: {len(T.LOG)}", flush=True)
    print("UI RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)}): " + " | ".join(FAIL), flush=True)
    sys.stdout.flush()
    os._exit(0 if not FAIL else 1)


STEPS = [s_install, s_full, s_tab, s_no_scipy, s_model, s_model_scroll1, s_model_scroll2, s_model_scroll3, s_model_scroll4,
         s_solve, s_results_scroll, s_results_scroll2, s_range]
for _id, _kw, _edit in DIALOGS:
    STEPS += [make_dialog_step(_id, _kw, _edit), s_dialog_look]
STEPS += [s_inspect_start, s_inspect_move, s_inspect_look, s_inspect_stop, s_inspect_check, s_cleanup, s_done]


def runner():
    i = STATE["i"]
    if i >= len(STEPS):
        return None
    st = STEPS[i]; STATE["i"] = i + 1
    try:
        print(f"UI step {i + 1}/{len(STEPS)}: {st.__name__}", flush=True)
        st()
    except Exception as e:
        fail(f"step {st.__name__}: {e.__class__.__name__}: {e}"); traceback.print_exc()
        if st is s_install:
            s_done()
    return 1.2


bpy.app.timers.register(runner, first_interval=3.0)
