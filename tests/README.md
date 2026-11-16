# PERM tests

All three scripts run inside Blender and drive PERM through the commands of the panel.
`<blender>` is the Blender executable, `<site>` a folder that holds SciPy for that Blender's Python (after
**Install SciPy** it is Blender's user `scripts/modules` folder; with `--factory-startup` Blender does not look
there by itself, PERM does).

```
<blender> -b --factory-startup --python-exit-code 1 --python tests/test_commands.py   -- --addon perm.py [--site <site>] [--out commands.json]
<blender> -b --factory-startup --python-exit-code 1 --python tests/test_benchmarks.py -- --addon perm.py [--site <site>] [--out benchmarks.json]
bash tests/run_ui_test.sh <blender> perm.py [<site>] [<screenshot folder>]
```

Each prints `RESULT: PASS` (or `UI RESULT: PASS`) and ends with exit code 0, or lists what failed.

| Script | What it checks |
|---|---|
| `test_commands.py` | Every operator of the add-on is run at least once, most of them in several situations: a rotated and translated quad plate, a triangle mesh in both topologies, a mesh with an n-gon, a curved shell, two parts joined by connections, bars on edges, and meshes straight from Blender's Add menu (monkey, UV sphere, icosphere, torus, cone, a scaled and rotated box). Sums of loads and reactions, the exported JSON and the stored assignments are compared with what was set. Wrong use (nothing selected, no supports, no mesh) must end with a message, not a Python error. |
| `test_benchmarks.py` | Displacements against closed-form and published solutions; see `docs/VERIFICATION.md`. |
| `test_ui.py` | The add-on is installed from disk and enabled; the panel is drawn without a mesh, with the SciPy notice, with a full model and with results; every dialog is opened; the hover inspector is started, moved and stopped; the add-on is disabled and enabled again. Needs a display: `run_ui_test.sh` starts Xvfb and fails if Blender prints a Python traceback. |

`perm_testlib.py` holds the helpers (mesh builders, Edit Mode selections, one `call()` that records the outcome
of every operator).

The interface test presses no real buttons: it calls the operators with `INVOKE_DEFAULT`, which runs the same
`invoke` and `draw` code as a click.
