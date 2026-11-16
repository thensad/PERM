# Changelog

## v32 — public release

PERM v31 was written for one reality-captured girder bridge. v32 keeps that analysis available and makes the
add-on usable on any surface mesh.

### Changed behaviour

- **Bridge details are options, off by default.** Loading patches, bearing pads, post base plates and ties
  between offset layers were always on in v27–v31. They are now set per scene in *8b. Connection details*, all
  off for a new model. **Bridge preset (paper settings)** switches them on with the v27–v31 values; the results
  are then those of v31. *A file made with v27–v31 needs the preset once to reproduce its earlier results.*
- **Girder checks are an option, off by default.** The support advice in the console (clamping, thrust), and in
  the report the span, the positions a/L, the deflection per girder line, the 0.1–0.9 L stress lines and the
  midspan section, appear only with *Girder checks and span report* (part of the preset). Without it the report
  lists supports, reactions, loads, displacements, stress extremes per material and the equilibrium checks.
- **Local zone radius** (report line that leaves out elements near supports, point loads and material joints)
  is an option, 0 = off by default, 0.25 m in the preset.
- **A bare `bpy.ops.perm.run_solver()` follows the panel.** Called from a script without arguments, it applied
  self-weight whatever the panel said; it now reads self-weight and mesh topology from the panel unless they
  are passed explicitly.

### New

- **Surface loads** (*5b*): force per unit area on selected faces, in a global direction or as a pressure along
  the face normal; consistent nodal forces; listed in the report and the JSON export.
- **Self-weight direction**: acceleration components gx, gy, gz (was gz only).
- **Custom restraints**: any set of the six degrees of freedom, for symmetry planes and guides.
- **Install SciPy** button, shown while SciPy is missing; every command that needs SciPy says so instead of
  failing with a Python error.
- **Export FEA JSON** writes the model as analysed: nodes, elements with material, thickness and offset,
  supports, each load case as nodal forces, optionally the results (format `PERM_FEA_v2`).
- A note in the panel and in the console when the scene's Unit Scale is not 1 or the unit system is Imperial
  (PERM reads one Blender unit as 1 m).
- Tests: every command on generic meshes, benchmarks against reference solutions, and the interface
  (`tests/`).

### Fixed

- **Clear ALL** removed every `perm_*` property of the object, including the tables that map the face tags to
  material, section and element-type names. The faces kept their numbers, and the next assignment gave number 1
  to another name, so faces could change material or section without a message. It now removes supports and
  loads only (and is labelled so).
- **Export FEA JSON** only worked with the legacy `PERM_Supports` vertex group, which the panel no longer
  creates; on a model made with the panel it stopped with "Assign supports first".
- **Post base plates** stopped with a broadcast error on a post that had no inner node before its first
  bottom-layer node.
- The per-node force editor, run without its dialog (script, Redo), saved an empty or foreign list and wiped
  the group's forces.
- *Refresh* after the analysed mesh was deleted gave a raw Python message; it now asks to run the analysis again.
- The panel said "Triangles have NO bending". Triangles have had the MITC3 bending element since v26.

### Interface

- Wording for any structure (no "bridge", "deck", "concrete" in the general sections); the colour filter
  buttons are named by what they select (E < 100 GPa, E ≥ 100 GPa).
- Labels wrapped and button texts shortened so that the panel reads at the sidebar's default width.
- *Unassign* buttons, which remove a material, section or element type from all its faces, are labelled *Clear*.

### Code

- Code that nothing called was removed: the disabled DKT triangle, an older QM6 membrane
  routine, unused node-set and vertex-colour helpers, and the export of the classes to
  `bpy.app.driver_namespace`. The assembled matrices and the results are unchanged (see `docs/VERIFICATION.md`).
- `run_solver` no longer takes `gravity_z`: it had no effect. Self-weight is set by the scene properties
  `perm_apply_gravity` and `perm_gravity_x`, `perm_gravity_y`, `perm_gravity_z`.
- Console messages are prefixed `[PERM]` (they carried the number of the version that introduced them).

## v31 and earlier

These versions were not published. In short:

- **v31**: BEAM_B2, a 2-node 3D Timoshenko beam on mesh edges (sections by shape or by A, Iy, Iz, J);
  connections between mesh parts (fixed, pinned, hinge) with *Detect Contacts*.
- **v30**: point loads made with *Assign Load (per-node)* are spread over the loading patch as well.
- **v28**: railing-post bases built as steel base plates of shell elements: the hole in the deck's top layer
  under each post is closed, a steel plate is bonded on the deck, the scanned flare at the tube foot is taken
  out and the tube walls are carried down to the plate.
- **v27**: connection details: point loads on a loading patch, support nodes as bearing pads, railing posts on
  rigid base plates, the two overhang layers tied through the thickness; stiffness assembly in typed arrays.
- **v26**: formulation changes: warped quads linked rigidly to their mean plane; MITC4 transverse shear
  transformed with J⁻¹; the Hughes-Brezzi drilling penalty condensed with the QM6 incompatible modes, with
  γ = 0.1; TRI3 as CST + MITC3 + drilling; a rigid-body-consistent stabiliser for membrane-only elements;
  consistent nodal shares of gravity and pressure on QUAD4; force and moment closure of the reactions in the
  report.
- **v25**: local zones in the results report. **v23**: colour filter and colour range. **v22**: find element
  by number. **v21**: linear solver with a symmetric ordering. **v19**: colour scales, legend, results report. **v18**: shell
  offset.
