# SPDX-License-Identifier: GPL-3.0-or-later
# PERM - Physics-Encoded Reality Modeling.
# Copyright (C) 2026 Sadig Nuraliyev and Furkan Luleci, Department of Civil and Environmental Engineering, Louisiana State University.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file. Results of an analysis are the user's
# responsibility: check them against an independent calculation before relying on them.
"""
PERM - Physics-Encoded Reality Modeling

A Blender add-on that runs a linear static finite element analysis on a surface mesh. This file is the whole
add-on.

Stage 1, mesh to finite element model (PERMMeshConverter)
  * works on a duplicate of the mesh, never on the source object
  * cleans it: merges doubles, drops loose geometry, dissolves degenerate faces, makes normals consistent
  * triangulates or quadrangulates the faces where asked
  * turns vertices into nodes and faces into elements (midside nodes for TRI6 / QUAD8), finds the boundary,
    carries the material slots over
  * mesh quality: aspect ratio, skewness, warpage, scaled Jacobian
  * mesh export: Abaqus .inp, Nastran .bdf, JSON

Stage 2, analysis (PERMSolver)
  * QUAD4 shell: QM6 incompatible-mode membrane (Taylor 1976, with the Jacobian correction), MITC4 assumed
    transverse shear (Dvorkin-Bathe 1984) and Hughes-Brezzi drilling (1989), condensed with the QM6 modes
  * TRI3 shell: CST membrane, MITC3 bending and shear (Lee-Bathe 2004), Hughes-Brezzi drilling
  * TRI6 / QUAD8: membrane only, no validated bending formulation
  * TRUSS_T2, a 2-node axial bar, and BEAM_B2, a 2-node 3D Timoshenko beam, on mesh edges
  * warped quads: rigid links from the corners to the mean plane, so that rigid-body motion is exact
  * shell offset; reinforced concrete through a linear transformed section
  * supports: fixed, pinned, rollers, or any set of the six DOFs
  * loads: point forces, self-weight with a free direction, surface loads on faces
  * joints between mesh parts: fixed, pinned and hinged, with contact detection
  * connection details, all off for a new model: point loads on loading patches, supports on bearing pads,
    posts of a stiff material on base plates, offset layers tied through the thickness
    (panel section '8b. Connection details')
  * 6 DOF per node (ux, uy, uz, theta_x, theta_y, theta_z), sparse CSR assembly, direct solve of K u = F
  * stress and strain per element at the top and bottom fibre, von Mises and principal stresses
  * results drawn on the mesh through colour attributes; a text report per analysis

Units: SI throughout (m, N, Pa, kg). One Blender unit = 1 m.
Requires NumPy (ships with Blender) and SciPy (not shipped: the panel has an 'Install SciPy' button).
"""
# Add-on metadata: the file installs through Edit > Preferences > Add-ons > Install from Disk.
bl_info = {
    "name": "PERM - Physics-Encoded Reality Modeling",
    "author": "Sadig Nuraliyev, Furkan Luleci",
    "description": "Linear finite element analysis on any surface mesh: shells, trusses and beams, "
                   "joints between mesh parts, results drawn on the mesh",
    "version": (32, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > PERM",
    "category": "Object",
}
import bpy
import bmesh
import math
import re
import time
import numpy as np
from mathutils import Vector, Matrix
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto
from collections import defaultdict, Counter

# SciPy for sparse matrix assembly and solving. Blender does not ship SciPy: the PERM panel shows an
# 'Install SciPy' button when it is missing (PERM_OT_InstallSciPy), and every command that needs it says so.
PERM_SCIPY_ERROR = ""
try:
    from scipy import sparse
    from scipy.sparse import linalg as spla
    HAS_SCIPY = True
except Exception as _e_scipy:               # ImportError, or a SciPy built for another NumPy
    HAS_SCIPY = False
    PERM_SCIPY_ERROR = f"{_e_scipy.__class__.__name__}: {_e_scipy}"

PERM_NO_SCIPY_MSG = ("SciPy is not installed in Blender's Python - the solver needs it. "
                     "Click 'Install SciPy' at the top of the PERM panel, then run again.")


def _perm_scipy_target() -> str:
    """Folder the installer puts SciPy in: Blender's user 'scripts/modules' folder (on sys.path, writable
    without administrator rights, one per Blender version)."""
    d = bpy.utils.user_resource('SCRIPTS', path="modules", create=True)
    if not d:
        raise RuntimeError("Blender gave no user scripts folder")
    return d


def _perm_scipy_command(target: str) -> List[str]:
    """pip command line. NumPy is pinned to the version Blender ships, so pip picks a SciPy built for it
    and never replaces Blender's own NumPy."""
    import sys as _sys
    return [_sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
            "--upgrade", "--target", target, "scipy", f"numpy=={np.__version__}"]


def _perm_try_import_scipy() -> bool:
    """(Re)import SciPy into this module. True if the solver can run."""
    global HAS_SCIPY, PERM_SCIPY_ERROR, sparse, spla
    import importlib
    import sys as _sys, os as _os
    try:
        # the installer's folder is on sys.path after a normal start, but not with --factory-startup
        _t = bpy.utils.user_resource('SCRIPTS', path="modules")
        if _t and _os.path.isdir(_os.path.join(_t, "scipy")) and _t not in _sys.path:
            _sys.path.append(_t)
    except Exception:
        pass
    importlib.invalidate_caches()
    try:
        from scipy import sparse as _sp
        from scipy.sparse import linalg as _spla
        sparse, spla = _sp, _spla
        HAS_SCIPY, PERM_SCIPY_ERROR = True, ""
    except Exception as e:
        HAS_SCIPY, PERM_SCIPY_ERROR = False, f"{e.__class__.__name__}: {e}"
    return HAS_SCIPY


if not HAS_SCIPY and not _perm_try_import_scipy():
    print(f"[PERM] SciPy is not available ({PERM_SCIPY_ERROR}). The solver needs it: "
          "use 'Install SciPy' at the top of the PERM panel (View3D > Sidebar > PERM).")


def _perm_need_scipy(op) -> bool:
    """True (and an error in the status bar) when SciPy is missing. Call first in commands that need it."""
    if HAS_SCIPY or _perm_try_import_scipy():
        return False
    op.report({'ERROR'}, PERM_NO_SCIPY_MSG)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR SOLVER
# ─────────────────────────────────────────────────────────────────────────────
# K is symmetric positive definite (supports are penalty springs on the
# diagonal and solve() symmetrises K), so the LU factorisation uses a symmetric
# fill-reducing ordering (MMD on K + K^T) and diagonal pivots. That keeps the
# factors much smaller than the general-purpose spsolve default (COLAMD): same
# answer to round-off, several times faster, less memory. If it fails, the
# plain spsolve call is used.


def _perm_linear_solve(K, F):
    """Solve K·u = F. Returns (u, method name, relative residual |K·u - F| / |F|)."""
    F = np.asarray(F, dtype=float)
    nF = float(np.linalg.norm(F)) or 1.0

    def _res(u):
        return float(np.linalg.norm(K @ u - F)) / nF

    try:
        lu = spla.splu(sparse.csc_matrix(K), permc_spec="MMD_AT_PLUS_A",
                       diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        u = lu.solve(F)
        del lu
        r = _res(u)
        if np.all(np.isfinite(u)) and r < 1e-6:
            return u, "SciPy SuperLU, symmetric ordering", r
        print(f"[PERM Solver] Symmetric LU residual {r:.1e} is too large; solving again with spsolve")
    except Exception as e:
        print(f"[PERM Solver] Symmetric LU failed ({e.__class__.__name__}: {e}); solving again with spsolve")
    u = spla.spsolve(K, F)
    return u, "SciPy spsolve (COLAMD)", _res(u)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import json


class ElementType(Enum):
    """Element topologies PERM supports. The converter and the solver share this one enum, so members can be
    compared with `is`."""
    TRI3 = auto()      # 3-node linear triangle     - CST + MITC3 shell
    TRI6 = auto()      # 6-node quadratic triangle  - membrane only
    QUAD4 = auto()     # 4-node linear quad         - the verified shell:
                       #   QM6 membrane + MITC4 bending + Hughes-Brezzi drilling
    QUAD8 = auto()     # 8-node quadratic quad      - membrane only
    TRUSS_T2 = auto()  # 2-node linear truss bar    - axial only
    BEAM_B2 = auto()   # 2-node 3D Timoshenko beam  - axial, shear, torsion, bending


_PERM_LINE_TYPES = ('TRUSS_T2', 'BEAM_B2')     # line elements (bars on mesh edges), never shells


def _perm_is_line(e) -> bool:
    return getattr(getattr(e, 'element_type', None), 'name', '') in _PERM_LINE_TYPES


@dataclass
class FENode:
    id: int
    x: float
    y: float
    z: float
    boundary: bool = False
    groups: Set[str] = field(default_factory=set)

    @property
    def position(self):
        return Vector((self.x, self.y, self.z))

@dataclass
class FEElement:
    id: int
    element_type: ElementType
    node_ids: List[int]
    material_id: int = 0
    thickness: float = 1.0
    quality: Dict[str, float] = field(default_factory=dict)
    groups: Set[str] = field(default_factory=set)
    # Truss-specific: section id referring to converter.sections dict.
    # For shells this is None and thickness is used instead.
    section_id: Optional[str] = None
    # Truss-specific: material NAME chosen when the bar was created. Resolved to
    # material_id once the scene materials are loaded. Trusses never inherit a
    # face material -- a strand along a concrete web must still be steel.
    material_name: Optional[str] = None
    # source-mesh vertex indices of a bar's two ends (to know which mesh part it belongs to)
    src_verts: Optional[Tuple[int, int]] = None
    # Marker: set to True if this element should NOT contribute to assembly
    # (e.g., shell faces that have been replaced by truss elements but kept
    # for visualization purposes only)
    viz_only: bool = False
    # Marker: set to True when the USER deliberately assigned a membrane-only
    # element type to this face (Element Types panel -> "Membrane shell").
    # The face then carries in-plane load only and acts as a HINGE out of plane.
    # Distinct from an element type that HAS no validated bending formulation
    # (TRI6/QUAD8): that is a limitation of the element, this is a modelling
    # choice. assemble_global_stiffness counts and reports the two separately.
    membrane_only: bool = False
    # Marker: set to True when the USER deliberately assigned a PLATE element
    # type to this face (Element Types -> "Plate, MITC4 only"). The face then
    # carries out-of-plane load only: MITC4 bending and transverse shear plus
    # Hughes-Brezzi drilling, with NO membrane stiffness whatsoever.
    #
    # Measured consequence, free-free single element: 11 near-zero eigenvalues
    # instead of the 6 rigid-body modes. The five extra are the membrane
    # deformation modes (stretch x, stretch y, in-plane shear and the two
    # in-plane bending modes) left with zero energy. The element is therefore
    # only meaningful where nothing loads it in its own plane -- a flat plate
    # under transverse load. On a folded assembly (a box girder, a bridge deck
    # with webs) global bending is carried as MEMBRANE action of the flanges,
    # so this element removes the very load path that carries the structure.
    #
    # The exact complement of membrane_only: that one drops MITC4 and keeps
    # QM6, this one drops QM6 and keeps MITC4. Both keep drilling, because a
    # 6-DOF assembly needs a theta_z term or K is singular at every node.
    bending_only: bool = False
    # ── Reinforced concrete (linear transformed section) ────────────────────
    # Smeared reinforcement for shell elements, taken from the assigned Section
    # (NOT the material). Defaults describe plain concrete: with rebar_ratio == 0
    # or steel_modulus == 0 the element behaves exactly as before.
    rebar_ratio: float = 0.0      # A_s / (b·t), total over top+bottom mats
    rebar_cover: float = 0.0      # m, face → bar-centroid distance (each mat)
    steel_modulus: float = 0.0    # E_s (Pa); 0 = ignore rebar
    # ── Shell offset ──────────────────────────────────────────────────
    # Signed distance (m) from the nodes to the shell's MID-surface, along the
    # element normal (local +z). 0 = mid-surface on the nodes (classic shell).
    # Lets a scanned OUTER surface carry concrete that lies inside the member.
    # Filled from the face attribute 'perm_offset' by _perm_apply_shell_offsets.
    offset: float = 0.0

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


@dataclass
class MaterialProperty:
    id: int
    name: str
    youngs_modulus: float = 30e9
    poissons_ratio: float = 0.2
    density: float = 2400.0
    yield_strength: float = 250e6
    thermal_expansion: float = 12e-6

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'name': self.name,
            'E': self.youngs_modulus, 'nu': self.poissons_ratio,
            'rho': self.density, 'sigma_y': self.yield_strength,
            'alpha': self.thermal_expansion
        }


# ─────────────────────────────────────────────────────────────────────────────
# MESH QUALITY VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

class MeshQualityMetric(Enum):
    """Quality metrics for articulated data assessment.

    The member names are the public vocabulary; METRIC_KEYS maps each to the
    key it occupies in FEElement.quality. Anything that accepts a metric
    accepts either form -- see _resolve_metric().
    """
    ASPECT_RATIO = auto()
    SKEWNESS = auto()
    JACOBIAN = auto()
    MIN_ANGLE = auto()
    MAX_ANGLE = auto()
    WARPAGE = auto()


METRIC_KEYS = {
    MeshQualityMetric.ASPECT_RATIO: 'aspect_ratio',
    MeshQualityMetric.SKEWNESS:     'skewness',
    MeshQualityMetric.JACOBIAN:     'jacobian',
    MeshQualityMetric.MIN_ANGLE:    'min_angle',
    MeshQualityMetric.MAX_ANGLE:    'max_angle',
    MeshQualityMetric.WARPAGE:      'warpage',
}


def _resolve_metric(metric) -> str:
    """Accept a MeshQualityMetric or the raw dict key; return the dict key."""
    if isinstance(metric, MeshQualityMetric):
        return METRIC_KEYS[metric]
    key = str(metric)
    if key not in METRIC_KEYS.values():
        raise ValueError(f"Unknown quality metric: {metric!r}. "
                         f"Expected one of {sorted(METRIC_KEYS.values())}")
    return key


@dataclass
class MeshQualityReport:
    """Summary of articulated data quality analysis."""
    total_elements: int = 0
    total_nodes: int = 0
    tri_count: int = 0
    quad_count: int = 0
    min_aspect_ratio: float = float('inf')
    max_aspect_ratio: float = 0.0
    avg_aspect_ratio: float = 0.0
    min_skewness: float = float('inf')
    max_skewness: float = 0.0
    avg_skewness: float = 0.0
    max_warpage: float = 0.0
    avg_warpage: float = 0.0
    min_angle: float = 180.0
    max_angle: float = 0.0
    min_jacobian: float = 1.0     # scaled Jacobian; <= 0 means degenerate
    avg_jacobian: float = 0.0
    inverted_elements: int = 0    # scaled Jacobian < 0: corner folds through
    poor_elements: int = 0        # elements failing quality thresholds
    quality_threshold: float = 0.3  # skewness threshold for "poor"



# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: REPRESENTATION → ARTICULATED DATA
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MODEL vs. RESULT COPIES
# ─────────────────────────────────────────────────────────────────────────────
# A run hides the user's model and works on a copy ("<model>_FEA_Base"); every
# result mesh is a further copy. All copies carry perm_source_name, so whatever
# the user clicks afterwards -- a result mesh, the camera, nothing -- the panel
# and RUN ANALYSIS still resolve to the MODEL, never to a stale copy.

def _perm_is_result(o) -> bool:
    return bool(o.get("perm_result_viz") or o.get("perm_disp_viz") or o.get("perm_reaction_viz"))


def _perm_is_copy(o) -> bool:
    """Only RESULT meshes count as copies. A mesh the user selects and works on
    is the model, even if it began life as a working copy and still carries
    perm_working_copy / perm_source_name tags from that (duplicates and renamed
    copies inherit custom properties)."""
    return bool(o is not None and _perm_is_result(o))


def _perm_resolve_source(context):
    """The analysis MODEL: the active mesh if it is one, the model a result copy
    came from, or the last model analysed. None if there is none."""
    obj = context.active_object
    if obj is not None and obj.type == 'MESH':
        if not _perm_is_result(obj):
            return obj                       # the mesh you are working on IS the model
        src = obj.get("perm_source_name")    # a result mesh -> the model it came from
        if src and src in bpy.data.objects:
            return bpy.data.objects[src]
    last = context.scene.get("perm_last_source")
    if last and last in bpy.data.objects and bpy.data.objects[last].type == 'MESH':
        return bpy.data.objects[last]
    return None


def _perm_reject_copy(op, obj) -> bool:
    """Assignments on a result/working copy never reach the analysis. Refuse them."""
    if _perm_is_copy(obj):
        src = obj.get("perm_source_name", "the model")
        op.report({'ERROR'}, f"'{obj.name}' is a result copy, not the model - assignments "
                             f"here are ignored. Click 'Edit Model' in the PERM panel "
                             f"(it selects '{src}').")
        return True
    return False


def _perm_remove_object(o) -> None:
    data = o.data if o.type == 'MESH' else None
    bpy.data.objects.remove(o, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.meshes.remove(data)


class PERMMeshConverter:
    """
    Converts Blender representations (from photogrammetry/LiDAR)
    into FEA articulated data.
    """

    def __init__(self, blender_object: bpy.types.Object):
        if blender_object.type != 'MESH':
            raise TypeError(f"Object '{blender_object.name}' is not a representation (type: {blender_object.type})")

        self.source_object = blender_object
        self.source_name = blender_object.name
        self.working_object = None

        # FE Model data
        self.nodes: Dict[int, FENode] = {}
        self.elements: Dict[int, FEElement] = {}
        self.materials: Dict[int, MaterialProperty] = {}
        self.node_sets: Dict[str, Set[int]] = defaultdict(set)
        self.element_sets: Dict[str, Set[int]] = defaultdict(set)

        self._vert_to_node: Dict[int, int] = {}
        # FE node id -> vertex index. The solver needs this direction to map an
        # element back to the faces/vertex groups it came from; build it here so
        # it can never go missing.
        self._node_to_vert: Dict[int, int] = {}
        # Blender polygon index -> FE element id. NEVER assume elem_id ==
        # poly.index + 1: any skipped face breaks that correspondence, and
        # everything downstream that assumed it silently mislabels every face
        # after the first skip.
        self._poly_to_elem: Dict[int, int] = {}
        # ...and its inverse. The solver reads this one to map an element back
        # to the face it came from (truss viz-only marking). It was read behind
        # a hasattr() guard that nothing ever satisfied, so that lookup ALWAYS
        # fell through to a much slower per-vertex-group scan.
        self._element_to_poly: Dict[int, int] = {}
        # Polygon indices that produced NO element (n-gons). These are holes in
        # the structure, not cosmetic omissions -- see build_fe_model.
        self.skipped_ngons: List[int] = []
        self._mid_node_cache: Dict[Tuple[int, int], int] = {}
        self._next_node_id: int = 1
        self._next_elem_id: int = 1

        self.quality_report: Optional[MeshQualityReport] = None
        self.materials[0] = MaterialProperty(id=0, name="Default_Concrete")
        self.scale_factor: float = 1.0

    def _ensure_working_copy(self):
        """Creates a safe duplicate of the representation to prevent destructive operations."""
        if not self.working_object:
            # One working copy per model: drop the one from the previous run
            # (tagged, or by its legacy name) instead of stacking another.
            for o in list(bpy.data.objects):
                if o is self.source_object or o.select_get():
                    continue
                if (o.get("perm_working_copy") and o.get("perm_source_name") == self.source_name
                        and not o.visible_get()):
                    _perm_remove_object(o)       # our own hidden copy from the last run
            obj_copy = self.source_object.copy()
            obj_copy.data = self.source_object.data.copy()
            obj_copy.name = f"{self.source_name}_FEA_Base"
            obj_copy["perm_working_copy"] = True
            obj_copy["perm_source_name"] = self.source_name
            bpy.context.collection.objects.link(obj_copy)
            obj_copy.select_set(False)
            self.working_object = obj_copy
            # Hide the original to avoid viewport confusion
            self.source_object.hide_set(True)

    # ── STEP 1: REPRESENTATION CLEANING ──────────────────────────────────

    def clean_mesh(self,
                   merge_distance: float = 0.0001,
                   remove_loose: bool = True,
                   fix_normals: bool = True,
                   remove_doubles: bool = True,
                   remove_degenerate: bool = True) -> dict:
        """Clean the representation for FEA suitability safely."""
        self._ensure_working_copy()
        obj = self.working_object
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        stats = {
            'verts_before': len(obj.data.vertices),
            'faces_before': len(obj.data.polygons),
        }

        if remove_doubles:
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)

        if remove_degenerate:
            # remove_doubles only collapses faces whose VERTICES coincide.
            # Zero-area faces with distinct but collinear vertices survive it,
            # and those are exactly the elements whose Jacobian is singular in
            # _compute_element_stiffness. Dissolve them explicitly.
            bpy.ops.mesh.dissolve_degenerate(threshold=merge_distance)

        if remove_loose:
            bpy.ops.mesh.delete_loose(
                use_verts=True,
                use_edges=True,
                use_faces=False
            )

        if fix_normals:
            bpy.ops.mesh.normals_make_consistent(inside=False)

        bpy.ops.object.mode_set(mode='OBJECT')

        stats['verts_after'] = len(obj.data.vertices)
        stats['faces_after'] = len(obj.data.polygons)
        stats['verts_removed'] = stats['verts_before'] - stats['verts_after']
        stats['faces_removed'] = stats['faces_before'] - stats['faces_after']

        # `faces_removed` is a NET face-count change -- doubles, loose geometry
        # and degenerates together -- not a count of degenerate faces. Naming it
        # as one made the report claim something it never measured.
        print(f"[PERM] Representation cleaned: {stats['verts_removed']} vertices and "
              f"{stats['faces_removed']} faces removed in total "
              f"(merge + loose + degenerate)")

        return stats

    # ── STEP 2: ELEMENT TYPE CONVERSION ──────────────────────────────────

    def convert_to_triangles(self) -> None:
        """Convert all faces to triangles (for Tri3/Tri6 elements)."""
        self._ensure_working_copy()
        obj = self.working_object
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"[PERM] Converted to triangles: {len(obj.data.polygons)} elements")

    def convert_to_quads(self, max_angle: float = 40.0) -> None:
        """Convert triangles to quads where possible (for Quad4/Quad8 elements)."""
        self._ensure_working_copy()
        obj = self.working_object
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=math.radians(max_angle),
            shape_threshold=math.radians(max_angle)
        )
        bpy.ops.object.mode_set(mode='OBJECT')

        tri_count = sum(1 for p in obj.data.polygons if len(p.vertices) == 3)
        quad_count = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        print(f"[PERM] Converted to quads: {quad_count} quads, {tri_count} remaining tris")

    def _materials_from_slots(self, obj: bpy.types.Object) -> Dict[int, int]:
        """Map Blender material slot index -> converter material id.

        Per-slot FEA properties are read from CUSTOM PROPERTIES on the Blender
        material ('youngs_modulus', 'poissons_ratio', 'density',
        'yield_strength'); a slot without them inherits the concrete defaults
        of MaterialProperty. Slot materials are numbered from 1 so that id 0
        stays reserved as the fallback -- a caller that owns material
        assignment itself (PERMSolver does) can reset every element to 0 and
        know exactly which material it is getting.

        Returns {} when the object has no usable slots, leaving every element
        on material_id = 0.
        """
        slots = [sl.material for sl in obj.material_slots] if obj.material_slots else []
        if not any(slots):
            return {}

        default = MaterialProperty(id=0, name="Default_Concrete")
        slot_to_matid: Dict[int, int] = {}
        next_id = 1
        for slot_index, mat in enumerate(slots):
            if mat is None:
                slot_to_matid[slot_index] = 0      # empty slot -> fallback
                continue
            self.materials[next_id] = MaterialProperty(
                id=next_id,
                name=mat.name,
                youngs_modulus=float(mat.get("youngs_modulus", default.youngs_modulus)),
                poissons_ratio=float(mat.get("poissons_ratio", default.poissons_ratio)),
                density=float(mat.get("density", default.density)),
                yield_strength=float(mat.get("yield_strength", default.yield_strength)),
            )
            slot_to_matid[slot_index] = next_id
            next_id += 1

        print(f"[PERM] Read {next_id - 1} material(s) from object slots "
              f"(id 0 remains the fallback)")
        return slot_to_matid

    # ── STEP 3: BUILD FINITE ELEMENT DATA ────────────────────────────────

    def build_fe_model(self,
                       target_type: ElementType = ElementType.TRI3,
                       thickness: float = 1.0,
                       scale: float = 1.0,
                       use_material_slots: bool = True) -> None:
        """Build the articulated data from the Blender representation.

        thickness           in SOURCE mesh units; scaled by `scale` along with
                            the coordinates so the model is ONE unit system.
        use_material_slots  read per-element material_id from the object's
                            Blender material slots. A caller that assigns
                            materials itself should pass False, or reset
                            material_id afterwards.
        """
        self.scale_factor = scale
        self.nodes.clear()
        self.elements.clear()
        self._vert_to_node.clear()
        self._node_to_vert.clear()
        self._poly_to_elem.clear()
        self._element_to_poly.clear()
        self.skipped_ngons.clear()
        self._mid_node_cache.clear()
        self._next_node_id = 1
        self._next_elem_id = 1

        self._ensure_working_copy()

        # Prepare representation topology
        if target_type in (ElementType.TRI3, ElementType.TRI6):
            self.convert_to_triangles()
        elif target_type in (ElementType.QUAD4, ElementType.QUAD8):
            self.convert_to_quads()

        obj = self.working_object
        mesh = obj.data
        world_matrix = obj.matrix_world

        # ── Create FE Nodes from vertices ──
        for vert in mesh.vertices:
            world_co = world_matrix @ vert.co
            node = FENode(
                id=self._next_node_id,
                x=world_co.x * scale,
                y=world_co.y * scale,
                z=world_co.z * scale
            )
            self.nodes[node.id] = node
            self._vert_to_node[vert.index] = node.id
            self._node_to_vert[node.id] = vert.index
            self._next_node_id += 1

        # ── Material slots → converter materials ──
        # Without this every element keeps material_id = 0, so a multi-material
        # model built by the converter alone is analysed and EXPORTED as
        # single-material.
        # PERM_OT_RunSolver overrides this wholesale when scene Materials exist,
        # so this governs direct use of the converter and the .inp/.bdf/JSON
        # exports.
        slot_to_matid = self._materials_from_slots(obj) if use_material_slots else {}

        # ── Create FE Elements from faces ──
        for poly in mesh.polygons:
            vert_indices = list(poly.vertices)
            n = len(vert_indices)

            fe_node_ids = [self._vert_to_node[vi] for vi in vert_indices]

            if n == 3:
                if target_type == ElementType.TRI6:
                    mid_nodes = self._create_midside_nodes(vert_indices, mesh, world_matrix, scale)
                    fe_node_ids.extend(mid_nodes)
                    elem_type = ElementType.TRI6
                else:
                    elem_type = ElementType.TRI3

            elif n == 4:
                if target_type == ElementType.QUAD8:
                    mid_nodes = self._create_midside_nodes(vert_indices, mesh, world_matrix, scale)
                    fe_node_ids.extend(mid_nodes)
                    elem_type = ElementType.QUAD8
                else:
                    elem_type = ElementType.QUAD4
            else:
                # NOT a cosmetic omission. This face contributes no element, so
                # the structure has a HOLE there: no stiffness, no mass, no load
                # path across it. convert_to_quads() merges triangles into quads
                # but does NOT dissolve n-gons, so this is reachable on the
                # recommended QUAD4 path. Recorded and reported below.
                self.skipped_ngons.append(poly.index)
                continue

            element = FEElement(
                id=self._next_elem_id,
                element_type=elem_type,
                node_ids=fe_node_ids,
                # Thickness must live in the SAME unit system as the
                # coordinates, which were multiplied by `scale` above. Leaving
                # it unscaled put membrane rigidity Et out by `scale` and
                # flexural rigidity Et³/12 out by `scale³` -- a factor of 1e9
                # for a millimetre-to-metre conversion, with nothing to detect
                # it. `thickness` is therefore given in SOURCE mesh units.
                thickness=thickness * scale,
                material_id=slot_to_matid.get(poly.material_index, 0)
            )
            self.elements[element.id] = element
            self._poly_to_elem[poly.index] = element.id
            self._element_to_poly[element.id] = poly.index
            self._next_elem_id += 1

        self._identify_boundary_nodes(mesh)

        if self.skipped_ngons:
            n_ng = len(self.skipped_ngons)
            print(f"[PERM] {'!' * 60}")
            print(f"[PERM] WARNING: {n_ng} n-gon face(s) produced NO element.")
            print("[PERM]   The model has HOLES there: no stiffness, no mass,")
            print("[PERM]   no load path. Results near them are meaningless.")
            preview = ", ".join(str(i) for i in self.skipped_ngons[:12])
            more = f" ... (+{n_ng - 12} more)" if n_ng > 12 else ""
            print(f"[PERM]   polygon indices: {preview}{more}")
            print("[PERM]   Fix: triangulate or quadrangulate those faces, then")
            print("[PERM]   rebuild. Ctrl+T in Edit Mode triangulates a selection.")
            print(f"[PERM] {'!' * 60}")

        print(f"[PERM] Articulated data built: {len(self.nodes)} nodes, {len(self.elements)} elements")

    def _create_midside_nodes(self,
                              vert_indices: List[int],
                              mesh: bpy.types.Mesh,
                              world_matrix: Matrix,
                              scale: float) -> List[int]:
        """Create mid-side nodes sequentially for Abaqus/Nastran compliance."""
        mid_node_ids = []
        n = len(vert_indices)

        for i in range(n):
            v1_idx = vert_indices[i]
            v2_idx = vert_indices[(i + 1) % n]

            edge_key = tuple(sorted((v1_idx, v2_idx)))

            if edge_key in self._mid_node_cache:
                mid_node_ids.append(self._mid_node_cache[edge_key])
            else:
                co1 = world_matrix @ mesh.vertices[v1_idx].co
                co2 = world_matrix @ mesh.vertices[v2_idx].co
                mid_co = (co1 + co2) / 2.0

                node = FENode(
                    id=self._next_node_id,
                    x=mid_co.x * scale,
                    y=mid_co.y * scale,
                    z=mid_co.z * scale
                )
                self.nodes[node.id] = node
                self._mid_node_cache[edge_key] = node.id
                mid_node_ids.append(node.id)
                self._next_node_id += 1

        return mid_node_ids

    def _identify_boundary_nodes(self, mesh: bpy.types.Mesh) -> None:
        """Identify boundary nodes on open edges."""
        edge_face_count: Dict[Tuple[int, int], int] = defaultdict(int)

        for poly in mesh.polygons:
            verts = list(poly.vertices)
            n = len(verts)
            for i in range(n):
                edge = tuple(sorted((verts[i], verts[(i + 1) % n])))
                edge_face_count[edge] += 1

        boundary_verts = set()
        for edge, count in edge_face_count.items():
            if count == 1:
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])

        boundary_node_set = set()
        for vert_idx in boundary_verts:
            if vert_idx in self._vert_to_node:
                node_id = self._vert_to_node[vert_idx]
                self.nodes[node_id].boundary = True
                boundary_node_set.add(node_id)

        self.node_sets['boundary'] = boundary_node_set
        print(f"[PERM] Identified {len(boundary_node_set)} boundary nodes")

    # ── STEP 4: ARTICULATED DATA QUALITY ANALYSIS ────────────────────────

    def compute_quality_metrics(self, skewness_threshold: float = 0.3) -> MeshQualityReport:
        """Compute quality metrics including warpage for shell elements."""
        report = MeshQualityReport(
            total_elements=len(self.elements),
            total_nodes=len(self.nodes),
            quality_threshold=skewness_threshold
        )

        aspect_ratios = []
        skewness_vals = []
        warpage_vals = []
        jacobians = []
        all_angles = []

        for elem in self.elements.values():
            coords = [self.nodes[nid].position for nid in elem.node_ids[:elem.node_count]]
            warpage_ratio = 0.0

            if elem.element_type in (ElementType.TRI3, ElementType.TRI6):
                report.tri_count += 1
                p0, p1, p2 = coords[0], coords[1], coords[2]

                edges = [
                    (p1 - p0).length,
                    (p2 - p1).length,
                    (p0 - p2).length
                ]

                ar = max(edges) / max(min(edges), 1e-15)
                angles = self._triangle_angles(p0, p1, p2)

                ideal_angle = 60.0
                max_dev = max(abs(a - ideal_angle) for a in angles)
                skew = max_dev / (180.0 - ideal_angle) if (180.0 - ideal_angle) > 0 else 0

            elif elem.element_type in (ElementType.QUAD4, ElementType.QUAD8):
                report.quad_count += 1
                p0, p1, p2, p3 = coords[0], coords[1], coords[2], coords[3]

                edges = [
                    (p1 - p0).length,
                    (p2 - p1).length,
                    (p3 - p2).length,
                    (p0 - p3).length
                ]

                ar = max(edges) / max(min(edges), 1e-15)
                angles = self._quad_angles(p0, p1, p2, p3)

                ideal_angle = 90.0
                max_dev = max(abs(a - ideal_angle) for a in angles)
                skew = max_dev / 90.0

                # Quad Warpage Check
                n_vec = (p1 - p0).cross(p2 - p0)
                if n_vec.length > 1e-15:
                    n_vec.normalize()
                    out_of_plane_dist = abs((p3 - p0).dot(n_vec))
                    avg_edge = sum(edges) / 4.0
                    warpage_ratio = out_of_plane_dist / max(avg_edge, 1e-15)
                    warpage_vals.append(warpage_ratio)

            else:
                continue

            n_corner = 3 if elem.element_type in (ElementType.TRI3, ElementType.TRI6) else 4
            jac = self._scaled_jacobian(coords[:n_corner])

            elem.quality = {
                'aspect_ratio': ar,
                'skewness': skew,
                'jacobian': jac,
                'min_angle': min(angles),
                'max_angle': max(angles),
                'warpage': warpage_ratio
            }

            aspect_ratios.append(ar)
            skewness_vals.append(skew)
            jacobians.append(jac)
            all_angles.extend(angles)
            if jac < 0.0:
                report.inverted_elements += 1

            if skew > skewness_threshold:
                report.poor_elements += 1

        if aspect_ratios:
            report.min_aspect_ratio = min(aspect_ratios)
            report.max_aspect_ratio = max(aspect_ratios)
            report.avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        if skewness_vals:
            report.min_skewness = min(skewness_vals)
            report.max_skewness = max(skewness_vals)
            report.avg_skewness = sum(skewness_vals) / len(skewness_vals)
        if warpage_vals:
            report.max_warpage = max(warpage_vals)
            report.avg_warpage = sum(warpage_vals) / len(warpage_vals)
        if jacobians:
            report.min_jacobian = min(jacobians)
            report.avg_jacobian = sum(jacobians) / len(jacobians)
        if all_angles:
            report.min_angle = min(all_angles)
            report.max_angle = max(all_angles)

        self.quality_report = report
        self._print_quality_report(report)
        return report

    @staticmethod
    def _scaled_jacobian(points: List[Vector]) -> float:
        """Scaled Jacobian of a triangle or quadrilateral, from its corners.

            1.0  ideal (square / equilateral triangle)
            0.0  degenerate: a corner has collapsed, the element has no area
            < 0  inverted: a corner folds back through the element

        This is the quality measure that most directly predicts failure inside
        the solver: _compute_element_stiffness inverts the Jacobian at every
        Gauss point, so an element at or below zero here is the one that raises
        and gets skipped -- and a skipped element is a hole in the structure.
        """
        n = len(points)
        if n < 3:
            return 0.0
        # Reference normal, for orientation only. For a quad the mean
        # plane (diagonal cross product) -- the plane the element is formed on.
        ref = (points[1] - points[0]).cross(points[2] - points[0])
        if n == 4:
            rd = (points[2] - points[0]).cross(points[3] - points[1])
            if rd.length > 1e-15:
                ref = rd
        if ref.length < 1e-15:
            return 0.0
        ref = ref.normalized()

        worst = float('inf')
        for i in range(n):
            e1 = points[(i + 1) % n] - points[i]
            e2 = points[(i - 1) % n] - points[i]
            l1, l2 = e1.length, e2.length
            if l1 < 1e-15 or l2 < 1e-15:
                return 0.0
            worst = min(worst, e1.cross(e2).dot(ref) / (l1 * l2))

        # An equilateral triangle peaks at sin(60 deg); normalise so that the
        # ideal element of either topology reads 1.0.
        if n == 3:
            worst *= 2.0 / math.sqrt(3.0)
        return max(-1.0, min(1.0, worst))

    @staticmethod
    def _triangle_angles(p0: Vector, p1: Vector, p2: Vector) -> List[float]:
        edges = [p1 - p0, p2 - p1, p0 - p2]
        angles = []
        for i in range(3):
            e1 = -edges[(i - 1) % 3]
            e2 = edges[i]
            cos_a = e1.dot(e2) / max(e1.length * e2.length, 1e-15)
            cos_a = max(-1.0, min(1.0, cos_a))
            angles.append(math.degrees(math.acos(cos_a)))
        return angles

    @staticmethod
    def _quad_angles(p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> List[float]:
        points = [p0, p1, p2, p3]
        angles = []
        for i in range(4):
            prev_pt = points[(i - 1) % 4]
            curr_pt = points[i]
            next_pt = points[(i + 1) % 4]
            e1 = prev_pt - curr_pt
            e2 = next_pt - curr_pt
            cos_a = e1.dot(e2) / max(e1.length * e2.length, 1e-15)
            cos_a = max(-1.0, min(1.0, cos_a))
            angles.append(math.degrees(math.acos(cos_a)))
        return angles

    def _print_quality_report(self, r: MeshQualityReport) -> None:
        print("\n" + "=" * 65)
        print("  PERM — ARTICULATED DATA QUALITY REPORT")
        print("=" * 65)
        print(f"  Total Nodes:      {r.total_nodes}")
        print(f"  Total Elements:   {r.total_elements}")
        print(f"    Triangles:      {r.tri_count}")
        print(f"    Quads:          {r.quad_count}")
        print("-" * 65)
        print(f"  Aspect Ratio   —  min: {r.min_aspect_ratio:.3f}  max: {r.max_aspect_ratio:.3f}  avg: {r.avg_aspect_ratio:.3f}")
        print(f"  Skewness       —  min: {r.min_skewness:.3f}  max: {r.max_skewness:.3f}  avg: {r.avg_skewness:.3f}")
        print(f"  Warpage (Quad) —  max: {r.max_warpage:.3f}  avg: {r.avg_warpage:.3f}")
        print(f"  Angle Range    —  min: {r.min_angle:.1f}°  max: {r.max_angle:.1f}°")
        print(f"  Scaled Jacobian — min: {r.min_jacobian:.3f}  avg: {r.avg_jacobian:.3f}"
              f"   (1.0 ideal, <=0 degenerate)")
        print("-" * 65)
        if r.inverted_elements:
            print(f"  ** {r.inverted_elements} INVERTED/FOLDED element(s) (Jacobian < 0). They are NOT")
            print("  ** skipped: the solver integrates |det J|, so they enter the model with")
            print("  ** corrupt stiffness and produce false stress peaks around them.")
            print("  ** Fix them first: PERM panel -> 'Select Problem Faces'.")
        print(f"  Poor Elements (skew > {r.quality_threshold}):  {r.poor_elements}  ({100*r.poor_elements/max(r.total_elements,1):.1f}%)")
        print("=" * 65 + "\n")

    # ── STEP 6: VISUALIZATION IN BLENDER ─────────────────────────────────

    def visualize_quality_on_mesh(self, metric=MeshQualityMetric.SKEWNESS, color_good: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.0), color_bad: Tuple[float, ...] = (1.0, 0.1, 0.0, 1.0)) -> bpy.types.Object:
        metric = _resolve_metric(metric)
        self._ensure_working_copy()
        obj_copy = self.working_object.copy()
        obj_copy.data = self.working_object.data.copy()
        obj_copy.name = f"{self.source_name}_FEA_{metric}"
        bpy.context.collection.objects.link(obj_copy)

        mesh = obj_copy.data
        color_attr_name = "FEA_Quality"
        if hasattr(mesh, 'color_attributes'):
            if color_attr_name not in mesh.color_attributes:
                mesh.color_attributes.new(name=color_attr_name, type='BYTE_COLOR', domain='CORNER')
            color_layer = mesh.color_attributes[color_attr_name]
        else:
            if color_attr_name not in mesh.vertex_colors:
                mesh.vertex_colors.new(name=color_attr_name)
            color_layer = mesh.vertex_colors[color_attr_name]

        values = [elem.quality[metric] for elem in self.elements.values() if metric in elem.quality]
        if not values: return obj_copy

        v_min, v_max = min(values), max(values)
        v_range = v_max - v_min if v_max > v_min else 1.0

        for poly in mesh.polygons:
            # Use the vertex-to-element map built in build_fe_model: once a face is
            # skipped (an n-gon), polygon order and element ids no longer match and
            # poly.index + 1 would shift every later face by one element. Faces with
            # no element are left uncoloured.
            elem_id = self._poly_to_elem.get(poly.index)
            if elem_id is not None and elem_id in self.elements:
                val = self.elements[elem_id].quality.get(metric, 0)
                t = (val - v_min) / v_range
                color = tuple(color_good[c] * (1 - t) + color_bad[c] * t for c in range(4))
                for loop_idx in poly.loop_indices:
                    color_layer.data[loop_idx].color = color

        try:
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'SOLID'
                            space.shading.color_type = 'VERTEX'
            if hasattr(mesh, 'color_attributes') and color_attr_name in mesh.color_attributes:
                mesh.color_attributes.active_color = mesh.color_attributes[color_attr_name]
                mesh.color_attributes.render_color_index = mesh.color_attributes.find(color_attr_name)
        except Exception:
            pass

        print(f"[PERM] Quality visualization created: '{obj_copy.name}'")
        return obj_copy

    # ── STEP 7: EXPORT TO FEA FORMATS ────────────────────────────────────

    def _nastran_float(self, val: float) -> str:
        """Safely formats floats to strictly 8 characters to prevent Nastran crash."""
        if val == 0.0: return "     0.0"
        s = f"{val:8.4g}"
        if len(s) > 8: s = f"{val:8.3g}"
        if len(s) > 8: s = f"{val:8.2e}"
        return s[:8].rjust(8)

    def export_abaqus_inp(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            f.write("** PERM - Physics-Encoded Reality Modeling\n")
            f.write(f"** Source: {self.source_name}\n")
            f.write(f"** Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**\n*HEADING\nPERM FE Model - {self.source_name}\n")
            f.write("**\n*NODE\n")
            for node in sorted(self.nodes.values(), key=lambda n: n.id):
                f.write(f"{node.id}, {node.x:.8e}, {node.y:.8e}, {node.z:.8e}\n")

            type_map = {
                ElementType.TRI3: "S3",
                ElementType.TRI6: "STRI65",
                ElementType.QUAD4: "S4R",
                ElementType.QUAD8: "S8R"
            }

            for etype in ElementType:
                elems = [e for e in self.elements.values() if e.element_type == etype]
                if not elems: continue
                abq_type = type_map[etype]
                f.write(f"**\n*ELEMENT, TYPE={abq_type}\n")
                for elem in sorted(elems, key=lambda e: e.id):
                    node_str = ", ".join(str(nid) for nid in elem.node_ids)
                    f.write(f"{elem.id}, {node_str}\n")

            for set_name, node_ids in self.node_sets.items():
                if not node_ids: continue
                f.write(f"**\n*NSET, NSET={set_name}\n")
                ids = sorted(node_ids)
                for i in range(0, len(ids), 16):
                    chunk = ids[i:i+16]
                    f.write(", ".join(str(n) for n in chunk) + "\n")

            for mat in self.materials.values():
                f.write(f"**\n*MATERIAL, NAME={mat.name}\n*ELASTIC\n")
                f.write(f"{mat.youngs_modulus:.6e}, {mat.poissons_ratio:.4f}\n*DENSITY\n{mat.density:.2f},\n")

            f.write("**\n** END OF PERM MODEL\n")
        print(f"[PERM] Exported Abaqus .inp: {filepath}")

    def export_nastran_bdf(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            f.write("$ PERM - Physics-Encoded Reality Modeling\n")
            f.write(f"$ Source: {self.source_name}\nBEGIN BULK\n")

            for node in sorted(self.nodes.values(), key=lambda n: n.id):
                nx = self._nastran_float(node.x)
                ny = self._nastran_float(node.y)
                nz = self._nastran_float(node.z)
                f.write(f"GRID    {node.id:8d}        {nx}{ny}{nz}\n")

            type_map = {
                ElementType.TRI3: "CTRIA3",
                ElementType.TRI6: "CTRIA6",
                ElementType.QUAD4: "CQUAD4",
                ElementType.QUAD8: "CQUAD8"
            }

            for elem in sorted(self.elements.values(), key=lambda e: e.id):
                card = type_map[elem.element_type]
                nids = "".join(f"{nid:8d}" for nid in elem.node_ids)
                f.write(f"{card:<8s}{elem.id:8d}{elem.material_id+1:8d}{nids}\n")

            for mat in self.materials.values():
                f.write(f"MAT1    {mat.id+1:8d}{mat.youngs_modulus:8.1e}{mat.poissons_ratio:8.4f}{mat.density:8.1f}\n")

            f.write("ENDDATA\n")
        print(f"[PERM] Exported Nastran .bdf: {filepath}")

    def export_json(self, filepath: str) -> None:
        data = {
            'metadata': {
                'generator': 'PERM - Physics-Encoded Reality Modeling',
                'source': self.source_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'scale_factor': self.scale_factor
            },
            'nodes': {str(n.id): {'x': n.x, 'y': n.y, 'z': n.z, 'boundary': n.boundary} for n in self.nodes.values()},
            'elements': {str(e.id): {'type': e.element_type.name, 'nodes': e.node_ids, 'material': e.material_id, 'thickness': e.thickness, 'quality': e.quality} for e in self.elements.values()},
            'node_sets': {name: sorted(list(ids)) for name, ids in self.node_sets.items()},
            'materials': {str(m.id): m.to_dict() for m in self.materials.values()},
            'quality_report': {
                'total_nodes': self.quality_report.total_nodes,
                'total_elements': self.quality_report.total_elements,
                'avg_aspect_ratio': self.quality_report.avg_aspect_ratio,
                'avg_skewness': self.quality_report.avg_skewness,
                'avg_warpage': self.quality_report.avg_warpage,
                'poor_elements': self.quality_report.poor_elements
            } if self.quality_report else None
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[PERM] Exported JSON: {filepath}")

    def get_model_summary(self) -> str:
        tri_count = sum(1 for e in self.elements.values() if e.element_type in (ElementType.TRI3, ElementType.TRI6))
        quad_count = sum(1 for e in self.elements.values() if e.element_type in (ElementType.QUAD4, ElementType.QUAD8))
        bbox_min = Vector((min(n.x for n in self.nodes.values()), min(n.y for n in self.nodes.values()), min(n.z for n in self.nodes.values())))
        bbox_max = Vector((max(n.x for n in self.nodes.values()), max(n.y for n in self.nodes.values()), max(n.z for n in self.nodes.values())))
        dims = bbox_max - bbox_min

        lines = [
            "╔══════════════════════════════════════════╗",
            "║     PERM ARTICULATED DATA SUMMARY        ║",
            "╠══════════════════════════════════════════╣",
            f"  Source:        {self.source_name}",
            f"  Nodes:         {len(self.nodes)}",
            f"  Elements:      {len(self.elements)}",
            f"    Triangles:   {tri_count}",
            f"    Quads:       {quad_count}",
            f"  Node Sets:     {len(self.node_sets)}",
            f"  Materials:     {len(self.materials)}",
            "  Bounding Box:",
            f"    X: {bbox_min.x:.4f} → {bbox_max.x:.4f}  (Δ={dims.x:.4f})",
            f"    Y: {bbox_min.y:.4f} → {bbox_max.y:.4f}  (Δ={dims.y:.4f})",
            f"    Z: {bbox_min.z:.4f} → {bbox_max.z:.4f}  (Δ={dims.z:.4f})",
            "╚══════════════════════════════════════════╝"
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: SOLVER DATA STRUCTURES  (BCs, loads, sections, results)
# ─────────────────────────────────────────────────────────────────────────────

class BCType(Enum):
    FIXED = auto()       # All 6 DOFs constrained (clamped edge): ux,uy,uz,θx,θy,θz
    PINNED = auto()      # ux, uy, uz constrained; rotations free (simply-supported)
    ROLLER_X = auto()    # uy, uz fixed; ux, rotations free
    ROLLER_Y = auto()    # ux, uz fixed; uy, rotations free
    ROLLER_Z = auto()    # ux, uy fixed; uz, rotations free
    CUSTOM = auto()


class LoadType(Enum):
    NODAL_FORCE = auto()
    GRAVITY = auto()
    PRESSURE = auto()
    DISTRIBUTED = auto()


@dataclass
class BoundaryCondition:
    name: str
    bc_type: BCType
    node_set: str
    prescribed_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoadCase:
    name: str
    load_type: LoadType
    node_set: str = ""
    element_set: str = ""
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    magnitude: float = 0.0
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class FEAResults:
    displacements: Optional[np.ndarray] = None      # shape (n_nodes, 6)
    # Support reactions R = K·u − F at constrained DOFs, GLOBAL frame,
    # shape (n_nodes, 6) = [Rx, Ry, Rz, Mx, My, Mz]; zero at free DOFs.
    reaction_forces: Optional[np.ndarray] = None
    reaction_nodes: Optional[Dict[int, List[int]]] = None   # node id -> constrained local DOFs
    reaction_total: Optional[np.ndarray] = None             # ΣR (force part), N
    applied_total: Optional[np.ndarray] = None              # ΣF applied (force part), N
    stresses: Optional[Dict[int, np.ndarray]] = None         # membrane σ at centroid (LOCAL frame)
    stresses_top: Optional[Dict[int, np.ndarray]] = None     # top fiber σ = [σxx, σyy, τxy] (LOCAL)
    stresses_bot: Optional[Dict[int, np.ndarray]] = None     # bottom fiber σ (LOCAL frame)
    moments: Optional[Dict[int, np.ndarray]] = None          # Mx, My, Mxy per element (LOCAL frame)
    # GLOBAL-frame copies of the Cartesian stresses & moments, stored as
    # [g_xx, g_yy, g_xy]. The per-element LOCAL frames differ across the mesh,
    # so local components cannot be averaged element-to-element; these global
    # copies can. Invariants (von Mises, principal) are frame-independent and
    # are still computed from the local tensor. Used by the heatmaps.
    stresses_global: Optional[Dict[int, np.ndarray]] = None
    stresses_top_global: Optional[Dict[int, np.ndarray]] = None
    stresses_bot_global: Optional[Dict[int, np.ndarray]] = None
    moments_global: Optional[Dict[int, np.ndarray]] = None
    strains: Optional[Dict[int, np.ndarray]] = None
    # Transverse shear (Reissner-Mindlin). shear_forces are the resultants
    # [Qx, Qy] per unit width in the element LOCAL frame; shear_forces_global is
    # the same vector in global coords (3 components — it is a VECTOR, not a
    # 2nd-order tensor, so it rotates with T, not T·(·)·Tᵀ). max_shear_stress is
    # the peak through-thickness value, 1.5·|Q|/t for the parabolic profile.
    shear_forces: Optional[Dict[int, np.ndarray]] = None
    shear_forces_global: Optional[Dict[int, np.ndarray]] = None
    max_shear_stress: Optional[Dict[int, float]] = None
    von_mises: Optional[Dict[int, float]] = None             # max of top/bot
    von_mises_top: Optional[Dict[int, float]] = None
    von_mises_bot: Optional[Dict[int, float]] = None
    # Principal stresses: stored as [σ1, σ2] where σ1 ≥ σ2 (algebraic ordering)
    # σ1 positive = tension, σ2 negative = compression (typically)
    principal_stresses: Optional[Dict[int, np.ndarray]] = None       # at the fibre with the larger von Mises stress
    principal_stresses_top: Optional[Dict[int, np.ndarray]] = None   # [σ1_top, σ2_top]
    principal_stresses_bot: Optional[Dict[int, np.ndarray]] = None   # [σ1_bot, σ2_bot]
    max_displacement: float = 0.0
    max_uz: float = 0.0
    max_von_mises: float = 0.0
    max_principal_tension: float = 0.0   # most tensile σ1 across all elements/fibers
    min_principal_compression: float = 0.0  # most compressive σ2 across all elements/fibers
    total_strain_energy: float = 0.0
    solve_time: float = 0.0
    # Truss-specific results
    truss_axial_forces: Optional[Dict[int, float]] = None    # eid → N (signed)
    truss_axial_stresses: Optional[Dict[int, float]] = None  # eid → σ (signed)
    beam_forces: Optional[Dict[int, dict]] = None          # eid -> internal forces at i / mid / j, stresses
    connection_forces: Optional[Dict[str, dict]] = None    # joint name -> force and moment carried


# ─────────────────────────────────────────────────────────────────────────────
# SHAPE FUNCTIONS & NUMERICAL INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class ShapeFunctions:

    @staticmethod
    def gauss_points_tri(order: int = 1) -> List[Tuple[Tuple[float, float], float]]:
        if order == 1:
            return [((1/3, 1/3), 0.5)]
        elif order == 3:
            return [((1/6, 1/6), 1/6), ((2/3, 1/6), 1/6), ((1/6, 2/3), 1/6)]
        elif order == 4:
            return [((1/3, 1/3), -27/96), ((1/5, 1/5), 25/96),
                    ((3/5, 1/5), 25/96), ((1/5, 3/5), 25/96)]
        else:
            return ShapeFunctions.gauss_points_tri(3)

    @staticmethod
    def gauss_points_quad(order: int = 2) -> List[Tuple[Tuple[float, float], float]]:
        if order == 1:
            return [((0.0, 0.0), 4.0)]
        elif order == 2:
            g = 1.0 / math.sqrt(3)
            pts = [(-g, -g), (g, -g), (g, g), (-g, g)]
            return [((xi, eta), 1.0) for xi, eta in pts]
        elif order == 3:
            g = math.sqrt(3/5)
            w = [5/9, 8/9, 5/9]
            coords = [-g, 0.0, g]
            return [((coords[j], coords[i]), w[j] * w[i])
                    for i in range(3) for j in range(3)]
        else:
            return ShapeFunctions.gauss_points_quad(2)

    @staticmethod
    def tri3_dN(xi: float, eta: float) -> np.ndarray:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    @staticmethod
    def tri6_dN(xi: float, eta: float) -> np.ndarray:
        zeta = 1 - xi - eta
        return np.array([
            [-(4*zeta - 1), 4*xi - 1, 0, 4*(zeta - xi), 4*eta, -4*eta],
            [-(4*zeta - 1), 0, 4*eta - 1, -4*xi, 4*xi, 4*(zeta - eta)]
        ])

    @staticmethod
    def quad4_N(xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array([
            (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)
        ])

    @staticmethod
    def quad4_dN(xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ])

    @staticmethod
    def quad8_dN(xi: float, eta: float) -> np.ndarray:
        dNdxi = np.array([
            0.25 * (1 - eta) * (2*xi + eta), 0.25 * (1 - eta) * (2*xi - eta),
            0.25 * (1 + eta) * (2*xi + eta), 0.25 * (1 + eta) * (2*xi - eta),
            -xi * (1 - eta), 0.5 * (1 - eta**2), -xi * (1 + eta), -0.5 * (1 - eta**2)
        ])
        dNdeta = np.array([
            0.25 * (1 - xi) * (xi + 2*eta), 0.25 * (1 + xi) * (-xi + 2*eta),
            0.25 * (1 + xi) * (xi + 2*eta), 0.25 * (1 - xi) * (-xi + 2*eta),
            -0.5 * (1 - xi**2), -(1 + xi) * eta, 0.5 * (1 - xi**2), -(1 - xi) * eta
        ])
        return np.array([dNdxi, dNdeta])


# ─────────────────────────────────────────────────────────────────────────────
# CORE: PERM FEA SOLVER
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 3D BEAM ELEMENT (BEAM_B2) — section properties and local axes
# ─────────────────────────────────────────────────────────────────────────────
# A beam is a mesh EDGE (or a pair of vertices) assigned an element type of kind BEAM_B2, exactly as a truss bar:
# it keeps its own material and cross-section. Local axes: x along the member (first to second node), z 'up'
# (the projection of global Z on the plane normal to x; global X for a vertical member), y = z × x, then (y, z)
# turned by the section's roll angle about x. Iy is the second moment about local y, i.e. vertical bending of a
# horizontal member (the strong axis of a standing rectangle b × h); Iz is about local z.
# Formulation: 2-node Timoshenko beam, exact for end loads, with shear flexibility Φ = 12 E I / (G A_s L²);
# A_s = 0 (shear factor 0 on a generic section) gives the Euler-Bernoulli beam. Self-weight is applied with
# consistent nodal loads (w L / 2 and w L² / 12 at each end), so element forces are exact for a uniform load.

def _perm_rect_torsion(b: float, h: float) -> float:
    """St Venant torsion constant of a solid rectangle b × h (Roark)."""
    a, c = max(b, h), min(b, h)
    if a <= 0.0 or c <= 0.0:
        return 0.0
    r = c / a
    return a * c ** 3 * (1.0 / 3.0 - 0.21 * r * (1.0 - r ** 4 / 12.0))


def _perm_beam_props(sec) -> dict:
    """Beam section properties from a scene Cross-Section (PERM_SectionPG) or a dict with the same field names.
    Shapes: RECT (b along local y, h along local z), RHS (b, h, wall t), CIRC (d), CHS (d, wall t);
    GENERIC uses the typed A, Iy, Iz, J, shear factor and extreme-fibre distances."""
    if isinstance(sec, dict):
        g = lambda k, d=0.0: float(sec.get(k, d) or 0.0)
        shape = sec.get('beam_shape', 'GENERIC') or 'GENERIC'
    else:
        g = lambda k, d=0.0: float(getattr(sec, k, d) or 0.0)
        shape = getattr(sec, 'beam_shape', 'GENERIC') or 'GENERIC'
    b, h, t, d = g('dim_b'), g('dim_h'), g('dim_t'), g('dim_d')
    circular = False
    if shape == 'RECT' and b > 0 and h > 0:
        A = b * h; Iy = b * h ** 3 / 12.0; Iz = h * b ** 3 / 12.0; J = _perm_rect_torsion(b, h)
        Asy = Asz = A * 5.0 / 6.0; cy, cz = 0.5 * b, 0.5 * h
    elif shape == 'RHS' and b > 0 and h > 0 and 0.0 < 2.0 * t < min(b, h):
        bi, hi = b - 2.0 * t, h - 2.0 * t
        A = b * h - bi * hi
        Iy = (b * h ** 3 - bi * hi ** 3) / 12.0; Iz = (h * b ** 3 - hi * bi ** 3) / 12.0
        bm, hm = b - t, h - t
        J = 2.0 * t * bm ** 2 * hm ** 2 / (bm + hm)                  # Bredt, thin-walled closed section
        Asz = 2.0 * t * h; Asy = 2.0 * t * b                         # the walls parallel to each shear force
        cy, cz = 0.5 * b, 0.5 * h
    elif shape == 'CIRC' and d > 0:
        A = math.pi * d ** 2 / 4.0; Iy = Iz = math.pi * d ** 4 / 64.0; J = 2.0 * Iy
        Asy = Asz = 0.9 * A; cy = cz = 0.5 * d; circular = True
    elif shape == 'CHS' and d > 0 and 0.0 < 2.0 * t < d:
        ro, ri = 0.5 * d, 0.5 * d - t
        A = math.pi * (ro ** 2 - ri ** 2); Iy = Iz = math.pi * (ro ** 4 - ri ** 4) / 4.0; J = 2.0 * Iy
        Asy = Asz = 0.5 * A; cy = cz = ro; circular = True
    else:
        shape = 'GENERIC'
        A, Iy, Iz, J = g('area'), g('Iy'), g('Iz'), g('J')
        k = g('shear_factor', 5.0 / 6.0)
        Asy = Asz = k * A
        cy, cz = g('fibre_y'), g('fibre_z')
    return dict(shape=shape, A=A, Iy=Iy, Iz=Iz, J=J, Asy=Asy, Asz=Asz, cy=cy, cz=cz, circular=circular,
                roll=math.radians(g('roll_deg')))


def _perm_beam_axes(ex, roll: float = 0.0) -> np.ndarray:
    """Rows: local x (member axis), y and z in global coordinates (see the block comment above)."""
    ex = np.asarray(ex, float); ex = ex / np.linalg.norm(ex)
    ref = np.array([1.0, 0.0, 0.0]) if abs(ex[2]) > 0.999 else np.array([0.0, 0.0, 1.0])
    ez = ref - (ref @ ex) * ex; ez = ez / np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        ey, ez = c * ey + s * ez, -s * ey + c * ez
    return np.vstack([ex, ey, ez])


def _perm_beam_k_local(E: float, G: float, bp: dict, L: float) -> np.ndarray:
    """12 × 12 local stiffness of the 3D Timoshenko beam, DOFs per node (u, v, w, θx, θy, θz)."""
    A, Iy, Iz, J = bp['A'], bp['Iy'], bp['Iz'], bp['J']
    phy = 12.0 * E * Iz / (G * bp['Asy'] * L * L) if bp['Asy'] > 0 else 0.0    # bending in x-y (v, θz)
    phz = 12.0 * E * Iy / (G * bp['Asz'] * L * L) if bp['Asz'] > 0 else 0.0    # bending in x-z (w, θy)
    k = np.zeros((12, 12))
    ea, gj = E * A / L, G * J / L
    k[0, 0] = k[6, 6] = ea; k[0, 6] = k[6, 0] = -ea
    k[3, 3] = k[9, 9] = gj; k[3, 9] = k[9, 3] = -gj
    a = E * Iz / (L ** 3 * (1.0 + phy))
    for (i, j), v in {(1, 1): 12 * a, (7, 7): 12 * a, (1, 7): -12 * a, (1, 5): 6 * a * L, (1, 11): 6 * a * L,
                      (5, 7): -6 * a * L, (7, 11): -6 * a * L, (5, 5): (4 + phy) * a * L * L,
                      (11, 11): (4 + phy) * a * L * L, (5, 11): (2 - phy) * a * L * L}.items():
        k[i, j] = k[j, i] = v
    b = E * Iy / (L ** 3 * (1.0 + phz))
    for (i, j), v in {(2, 2): 12 * b, (8, 8): 12 * b, (2, 8): -12 * b, (2, 4): -6 * b * L, (2, 10): -6 * b * L,
                      (4, 8): 6 * b * L, (8, 10): 6 * b * L, (4, 4): (4 + phz) * b * L * L,
                      (10, 10): (4 + phz) * b * L * L, (4, 10): (2 - phz) * b * L * L}.items():
        k[i, j] = k[j, i] = v
    return k


def _perm_beam_sigma(S, bp) -> Tuple[float, float]:
    """Largest and smallest normal stress at a station with internal forces S = (N, Vy, Vz, T, My, Mz).
    Rectangular-type sections: the corners (|My| cz / Iy + |Mz| cy / Iz); circular: the resultant moment."""
    N, My, Mz = float(S[0]), float(S[4]), float(S[5])
    s0 = N / bp['A'] if bp['A'] > 0 else 0.0
    if bp['circular']:
        m = math.hypot(My, Mz) * bp['cz'] / bp['Iy'] if bp['cz'] > 0 and bp['Iy'] > 0 else 0.0
    else:
        m = ((abs(My) * bp['cz'] / bp['Iy']) if bp['cz'] > 0 and bp['Iy'] > 0 else 0.0) + \
            ((abs(Mz) * bp['cy'] / bp['Iz']) if bp['cy'] > 0 and bp['Iz'] > 0 else 0.0)
    return s0 + m, s0 - m


class PERMSolver:
    """
    Global DOF layout per node: [ux, uy, uz, θx, θy, θz]   (6 DOFs)

    θz is the drilling rotation. It is constrained to the membrane's own
    in-plane rotation ω = ½(∂v/∂x - ∂u/∂y) by a Hughes-Brezzi penalty rather
    than propped up with a fictitious spring, which keeps rigid-body motion
    exact at folds (see _drilling_K_hb).

    Element stiffness is built in the element-local frame at 5 DOF/node
    [ux, uy, uz, θx, θy] — membrane (ux,uy) plus Mindlin bending (uz,θx,θy) —
    then expanded to 6/node, given the drilling term, and rotated to global.

    IMPORTANT convention: θx, θy are components of the rotation VECTOR, not the
    Mindlin section rotations β. They are related by βx = θy, βy = -θx. MITC4 is
    formulated in β, so the element B-matrices apply that mapping explicitly.
    Skipping it is invisible on flat meshes but destroys rigid-body motion of
    any folded assembly.
    """

    DOFS_PER_NODE = 6   # ux,uy,uz,θx,θy,θz  (θz = drilling; full 6-DOF shell)

    # Hughes-Brezzi drilling penalty, as a multiple of G·t. Ties θz to the
    # membrane rotation ω instead of adding a bare diagonal spring, so it is
    # rigid-body consistent AND leaves no null mode.
    #
    # For QUAD4 the penalty uses the in-plane rotation of the full QM6 field
    # (incompatible modes included) and is condensed with it
    # (_membrane_drilling_K_qm6), so it does not stiffen in-plane bending. γ is
    # then set by the other job of the drilling DOF: carrying bending moments
    # across nodes where facets meet at an angle. With γ = 0.1 the ratios to the
    # reference solutions are 0.993 for MacNeal's 5:1 cantilever and 0.999 (z) /
    # 1.004 (y) for the twisted beam, t = 0.32, 12x2 mesh. A much smaller γ
    # leaves the drilling rotation nearly free and makes thick shells with
    # non-coplanar facets too flexible. TRI3 (CST, no incompatible modes) and the
    # plate-only element use the separate penalty (_drilling_K_hb) with the same γ.
    #
    # 1-point integration of this term is NOT a valid shortcut: one constraint
    # per element leaves hourglass drilling modes (13 null modes on a
    # simply-supported plate, NaN results). The 2x2 rule is required.
    DRILL_GAMMA = 0.1

    # Bare-diagonal fallback for element types with no Hughes-Brezzi path
    # (TRI6/QUAD8, membrane-only). See _stabilize_drilling_diag.
    DRILL_FALLBACK = 1e-3

    # Local-frame DOF indices (element-local still has NO drilling: 5 local DOFs)
    _IDX_UX, _IDX_UY, _IDX_UZ, _IDX_RX, _IDX_RY = 0, 1, 2, 3, 4

    def __init__(self, converter: 'PERMMeshConverter'):
        if not converter.nodes or not converter.elements:
            raise ValueError("FE model not built. Run converter.build_fe_model() first.")
        if not HAS_SCIPY:
            raise ImportError("SciPy is required. Install in Blender's Python.")

        self.converter = converter
        # Ensure converter has a sections dict (older module 1 versions may not)
        if not hasattr(converter, 'sections') or converter.sections is None:
            converter.sections = {}

        self.n_nodes = len(converter.nodes)
        self.n_elements = len(converter.elements)
        self.n_dofs = self.n_nodes * self.DOFS_PER_NODE

        sorted_node_ids = sorted(converter.nodes.keys())
        self.node_id_to_index: Dict[int, int] = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
        self.index_to_node_id: Dict[int, int] = {idx: nid for nid, idx in self.node_id_to_index.items()}

        self.node_coords = np.zeros((self.n_nodes, 3))
        for nid, node in converter.nodes.items():
            idx = self.node_id_to_index[nid]
            self.node_coords[idx] = [node.x, node.y, node.z]

        self.boundary_conditions: List[BoundaryCondition] = []
        self.load_cases: List[LoadCase] = []

        self._K_global: Optional[sparse.csr_matrix] = None
        self._F_global: Optional[np.ndarray] = None
        self._constrained_dofs: Set[int] = set()
        self._prescribed_values: Dict[int, float] = {}
        self.results = FEAResults()

        print(f"[PERM Solver] Initialized: {self.n_nodes} nodes, "
              f"{self.n_elements} elements, {self.n_dofs} DOFs (6 DOF/node)")

    # ── BC / LOAD API (unchanged surface) ───────────────────────────────

    def add_boundary_condition(self, name: str, bc_type: BCType, node_set: str,
                                prescribed_values: Dict[str, float] = None) -> None:
        if node_set not in self.converter.node_sets:
            available = list(self.converter.node_sets.keys())
            raise KeyError(f"Node set '{node_set}' not found. Available: {available}")
        bc = BoundaryCondition(name=name, bc_type=bc_type, node_set=node_set,
                               prescribed_values=prescribed_values or {})
        self.boundary_conditions.append(bc)
        n = len(self.converter.node_sets[node_set])
        print(f"[PERM Solver] BC '{name}': {bc_type.name} on '{node_set}' ({n} nodes)")

    def add_nodal_force(self, node_set: str, fx: float = 0.0, fy: float = 0.0,
                        fz: float = 0.0, name: str = "") -> None:
        if not name: name = f"Force_{node_set}"
        magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
        if magnitude < 1e-15: return
        direction = (fx / magnitude, fy / magnitude, fz / magnitude)
        load = LoadCase(name=name, load_type=LoadType.NODAL_FORCE, node_set=node_set,
                        direction=direction, magnitude=magnitude)
        self.load_cases.append(load)
        n = len(self.converter.node_sets.get(node_set, set()))
        print(f"[PERM Solver] Load '{name}': F=({fx:.1f}, {fy:.1f}, {fz:.1f}) N on '{node_set}' ({n} nodes)")

    def add_nodal_force_per_node(self, node_set: str,
                                  per_node_forces: dict, name: str = "") -> None:
        """Per-node force application: each node in node_set gets its own force.
        
        per_node_forces: {node_id (int): (fx, fy, fz)}
        Nodes in node_set not present in this dict get zero force.
        
        Implementation: stores the dict on the LoadCase via load.per_node_forces;
        _apply_nodal_force uses it if present, else falls back to uniform.
        """
        if not name: name = f"PerNodeForce_{node_set}"
        load = LoadCase(name=name, load_type=LoadType.NODAL_FORCE, node_set=node_set,
                        direction=(0, 0, 1), magnitude=1.0)  # placeholders
        # Attach per-node dict as a dynamic attribute
        load.per_node_forces = per_node_forces
        self.load_cases.append(load)
        total_fx = sum(f[0] for f in per_node_forces.values())
        total_fy = sum(f[1] for f in per_node_forces.values())
        total_fz = sum(f[2] for f in per_node_forces.values())
        n = len(self.converter.node_sets.get(node_set, set()))
        print(f"[PERM Solver] Load '{name}' (per-node): "
              f"total=({total_fx:.1f}, {total_fy:.1f}, {total_fz:.1f}) N "
              f"distributed over {len(per_node_forces)} explicit nodes "
              f"(group has {n} nodes)")

    def add_gravity(self, gx: float = 0.0, gy: float = 0.0, gz: float = -9.81,
                    name: str = "Gravity") -> None:
        load = LoadCase(name=name, load_type=LoadType.GRAVITY, gravity_vector=(gx, gy, gz))
        self.load_cases.append(load)
        print(f"[PERM Solver] Load '{name}': g=({gx:.2f}, {gy:.2f}, {gz:.2f}) m/s²")

    # ── ELEMENT GEOMETRY HELPERS ────────────────────────────────────────

    def _get_element_coords(self, elem: FEElement) -> np.ndarray:
        indices = [self.node_id_to_index[nid] for nid in elem.node_ids]
        return self.node_coords[indices]

    def _compute_local_system(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build right-handed local frame (e_x, e_y, e_z) with e_z = element normal.

        Returns:
            local_coords: (n,3) coordinates in element-local frame (z ≈ 0)
            T: (3,3) rotation s.t.  v_global = T @ v_local
        """
        p0, p1 = coords[0], coords[1]
        ex = p1 - p0
        ex_len = np.linalg.norm(ex)
        if ex_len < 1e-15:
            ex = np.array([1.0, 0.0, 0.0])
        else:
            ex = ex / ex_len

        # For quads use p3 to define plane, for tris use p2
        ref = coords[2] if len(coords) >= 3 else coords[1]
        v_temp = ref - p0
        ez = np.cross(ex, v_temp)
        ez_len = np.linalg.norm(ez)
        if ez_len < 1e-15:
            ez = np.array([0.0, 0.0, 1.0])
        else:
            ez = ez / ez_len

        if len(coords) >= 4:
            # Quads use the MEAN plane of the four corners (normal = the
            # diagonal cross product). The 0-1-2 plane alone folds a strongly
            # warped quad over in projection (negative Jacobian -> false peak).
            nd = np.cross(coords[2] - coords[0], coords[3] - coords[1])
            nd_len = np.linalg.norm(nd)
            if nd_len > 1e-15:
                ez = nd / nd_len
                ex = ex - (ex @ ez) * ez
                ex_n = np.linalg.norm(ex)
                ex = ex / ex_n if ex_n > 1e-15 else np.cross(ez, [1.0, 0.0, 0.0] if abs(ez[0]) < 0.9 else [0.0, 1.0, 0.0])
                ex = ex / np.linalg.norm(ex)

        ey = np.cross(ez, ex)
        ey_len = np.linalg.norm(ey)
        if ey_len > 1e-15:
            ey = ey / ey_len

        T = np.array([ex, ey, ez]).T  # columns are local basis vectors in global coords
        centered = coords - p0
        local_coords = (T.T @ centered.T).T

        return local_coords, T

    @staticmethod
    def _voigt2d_local_to_global(voigt2d: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Rotate an in-plane (plane-stress) 2-D tensor from the element-LOCAL
        frame to the GLOBAL frame; return [g_xx, g_yy, g_xy].

        voigt2d = [s_xx, s_yy, s_xy] in the local frame (local z = element normal).
        T maps local→global (its columns are the local basis in global coords),
        so the global tensor is  σ_g = T · σ_l · Tᵀ  with the local tensor
        embedded in 3-D as plane stress (σ_zz = σ_xz = σ_yz = 0). We return the
        in-plane global components [g_xx, g_yy, g_xy], in the same order as the
        sx/sy/txy (index 0/1/2) lookup. Being in one frame, they are safe to
        average across elements. Works for stress tensors and
        moment-resultant tensors alike (both are symmetric 2nd-order tensors).
        """
        sxx, syy, sxy = float(voigt2d[0]), float(voigt2d[1]), float(voigt2d[2])
        s_local = np.array([[sxx, sxy, 0.0],
                            [sxy, syy, 0.0],
                            [0.0, 0.0, 0.0]])
        s_global = T @ s_local @ T.T
        return np.array([s_global[0, 0], s_global[1, 1], s_global[0, 1]])

    # ── CONSTITUTIVE MATRICES ───────────────────────────────────────────

    @staticmethod
    def _rebar_factors(elem: FEElement, E_concrete: float,
                       thickness: float) -> Tuple[float, float]:
        """Linear transformed-section stiffness multipliers for smeared rebar.

        Returns (f_membrane, f_bending), each ≥ 1.0, that scale the plain-concrete
        Dm and Db of a shell element. Both are 1.0 (no change) unless the element
        carries reinforcement (rebar_ratio > 0 and steel_modulus > 0).

        Derivation (uncracked transformed section, symmetric top+bottom mats,
        per unit width b=1, modular ratio n = E_s / E_c, ρ = A_s/(b·t),
        lever z = t/2 − cover):
            axial :  EA_eff = E_c·t·(1 + (n−1)·ρ)          → f_m = 1 + (n−1)·ρ
            bending: EI_eff = E_c·t³/12 · (1 + 12(n−1)ρ(z/t)²)
                     = E_c·t³/12 · (1 + 3(n−1)ρ(1−2c/t)²)  → f_b as below
        Rebar near the faces boosts bending more than membrane — that is exactly
        why one scalar "equivalent E" cannot capture RC, and both factors differ.
        """
        rho = float(getattr(elem, 'rebar_ratio', 0.0) or 0.0)
        E_s = float(getattr(elem, 'steel_modulus', 0.0) or 0.0)
        if rho <= 0.0 or E_s <= 0.0 or E_concrete <= 0.0 or thickness <= 0.0:
            return 1.0, 1.0
        n = E_s / E_concrete
        cover = float(getattr(elem, 'rebar_cover', 0.0) or 0.0)
        # Clamp cover so the bars stay inside the section (centroid can't cross mid-plane).
        cover = min(max(cover, 0.0), 0.5 * thickness)
        f_m = 1.0 + (n - 1.0) * rho
        lever_ratio = 1.0 - 2.0 * cover / thickness   # = 2z/t ∈ [0,1]
        f_b = 1.0 + 3.0 * (n - 1.0) * rho * lever_ratio**2
        return f_m, f_b

    def _compute_Dm(self, material: MaterialProperty, thickness: float,
                    rebar_factor: float = 1.0) -> np.ndarray:
        """Membrane constitutive matrix: integrated through thickness.
        σ_membrane = (Dm/t) · ε   →   force/length = Dm · ε
        We multiply by t inside Dm so the membrane B-matrix integrates a 2D area
        in the local frame (the t·area gives volume).

        Actually: standard convention is K_m = ∫_A Bᵀ (E_plane_stress · t) B dA
        So we return E_plane_stress · t.

        rebar_factor (default 1.0) scales the axial rigidity for a transformed
        reinforced-concrete section — see _rebar_factors.
        """
        E, nu, t = material.youngs_modulus, material.poissons_ratio, thickness
        factor = (E * t) / (1 - nu**2) * rebar_factor
        return factor * np.array([
            [1.0,  nu,  0.0],
            [nu,   1.0, 0.0],
            [0.0,  0.0, (1 - nu) / 2.0]
        ])

    def _compute_Db(self, material: MaterialProperty, thickness: float,
                    rebar_factor: float = 1.0) -> np.ndarray:
        """Bending constitutive (plate flexural rigidity matrix).
        M = Db · κ   where   κ = [κxx, κyy, 2κxy]
        Db = (E t³ / 12(1-ν²)) · [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]

        rebar_factor (default 1.0) scales the flexural rigidity for a transformed
        reinforced-concrete section — see _rebar_factors.
        """
        E, nu, t = material.youngs_modulus, material.poissons_ratio, thickness
        D_flex = E * t**3 / (12.0 * (1 - nu**2)) * rebar_factor
        return D_flex * np.array([
            [1.0,  nu,  0.0],
            [nu,   1.0, 0.0],
            [0.0,  0.0, (1 - nu) / 2.0]
        ])

    # ── MEMBRANE B-MATRIX ───────────────────────────────────────────────

    def _build_Bm(self, dN_phys: np.ndarray, n_nodes: int) -> np.ndarray:
        """Membrane B-matrix: ε = [εxx, εyy, γxy] = Bm · [ux,uy]_local
        Shape: (3, 2*n_nodes)
        """
        Bm = np.zeros((3, 2 * n_nodes))
        for i in range(n_nodes):
            dNx = dN_phys[0, i]
            dNy = dN_phys[1, i]
            Bm[0, 2*i]     = dNx
            Bm[1, 2*i + 1] = dNy
            Bm[2, 2*i]     = dNy
            Bm[2, 2*i + 1] = dNx
        return Bm

    # ── MITC4 PLATE BENDING (Bathe-Dvorkin 1985) ────────────────────────

    def _mitc4_Kb(self, coords_local: np.ndarray, material: MaterialProperty,
                  thickness: float, rebar_factor_b: float = 1.0) -> np.ndarray:
        """MITC4 bending + transverse shear stiffness. Returns 12x12 in DOF order
        [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3, w4,θx4,θy4].
        
        Reference: Dvorkin & Bathe (1984) "A continuum mechanics based four-node
        shell element for general non-linear analysis", Eng. Comput. 1, 77-88.
        The transverse shear strains are evaluated at four tying points and
        interpolated, eliminating shear locking.
        """
        x = coords_local[:, 0]
        y = coords_local[:, 1]
        E, nu, t = material.youngs_modulus, material.poissons_ratio, thickness

        Db = self._compute_Db(material, t, rebar_factor=rebar_factor_b)
        # Shear modulus times thickness, with shear correction factor 5/6
        G = E / (2.0 * (1.0 + nu))
        Ds = (5.0 / 6.0) * G * t * np.eye(2)

        K = np.zeros((12, 12))
        tie = self._mitc4_tying_rows(x, y)

        # Gauss integration (2x2 for bending, 2x2 for shear — same here)
        gauss = ShapeFunctions.gauss_points_quad(2)

        for (xi, eta), w in gauss:
            dN_nat = ShapeFunctions.quad4_dN(xi, eta)
            J = dN_nat @ np.column_stack([x, y])
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-20:
                continue
            J_inv = np.linalg.inv(J)
            dN_xy = J_inv @ dN_nat

            # Bending B (curvatures from the section rotations βx, βy)
            # κ = [βx,x ; βy,y ; βx,y + βy,x],  with  βx = θy,  βy = -θx
            # (see the derivation in _mitc4_tying_rows). DOF slots per node are
            # [w_i, θx_i, θy_i] → 3i+0, 3i+1, 3i+2.
            Bb = np.zeros((3, 12))
            for i in range(4):
                Bb[0, 3*i + 2] = +dN_xy[0, i]   # κxx =  ∂θy/∂x
                Bb[1, 3*i + 1] = -dN_xy[1, i]   # κyy = -∂θx/∂y
                Bb[2, 3*i + 2] = +dN_xy[1, i]   # 2κxy = ∂θy/∂y - ∂θx/∂x
                Bb[2, 3*i + 1] = -dN_xy[0, i]

            K += Bb.T @ Db @ Bb * abs(det_J) * w

            Bs_cart = self._mitc4_Bs_at(tie, J_inv, xi, eta)
            K += Bs_cart.T @ Ds @ Bs_cart * abs(det_J) * w

        return K

    def _mitc4_tying_rows(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Transverse-shear B rows at the four MITC tying points.

        Returns (Bs_A, Bs_C, Bs_B, Bs_D) — γ_ξz at A(0,-1) and C(0,1),
        γ_ηz at B(1,0) and D(-1,0). Each is a length-12 row over the DOF
        ordering [w_i, θx_i, θy_i].

        Evaluating the transverse shear only at these tying points and
        interpolating is what makes MITC4 locking-free.
        """
        def _shear_b_at(xi_tp, eta_tp, comp):
            """Bs row at a tying point; comp='xi' for γ_xi-z, 'eta' for γ_eta-z."""
            # Bilinear shape function derivatives
            dN_natural = 0.25 * np.array([
                [-(1 - eta_tp), (1 - eta_tp), (1 + eta_tp), -(1 + eta_tp)],
                [-(1 - xi_tp), -(1 + xi_tp), (1 + xi_tp), (1 - xi_tp)]
            ])
            N_vals = 0.25 * np.array([
                (1 - xi_tp) * (1 - eta_tp), (1 + xi_tp) * (1 - eta_tp),
                (1 + xi_tp) * (1 + eta_tp), (1 - xi_tp) * (1 + eta_tp)
            ])
            # Jacobian at this point
            J = dN_natural @ np.column_stack([x, y])
            # Covariant base vectors g_xi, g_eta in (x,y) plane
            g_xi = J[0, :]   # ∂(x,y)/∂ξ
            g_eta = J[1, :]  # ∂(x,y)/∂η

            # γ_xi-z = ∂w/∂ξ + (g_xi · β) where β = (βx, βy) are the Mindlin
            # SECTION rotations (u = z·βx, v = z·βy).
            #
            # The nodal DOFs are the components of the rotation VECTOR
            # Θ = (θx, θy, θz), not β. For Θ × r with r = (0,0,z):
            #     u = +z·θy ,  v = -z·θx      ⇒   βx = θy ,  βy = -θx
            # so (g · β) = g[0]·θy - g[1]·θx.
            #
            # Omitting this 90° mapping makes the element frame-inconsistent at
            # folds (each element's β would be rotated about its OWN normal),
            # which destroys rigid-body motion of any non-planar assembly. It
            # cancels exactly on flat meshes, which is why it stayed hidden.
            # In terms of nodal DOFs [w_i, θx_i, θy_i]:
            B_row = np.zeros(12)
            if comp == 'xi':
                g = g_xi
                for i in range(4):
                    B_row[3*i] = dN_natural[0, i]      # ∂w/∂ξ
            else:  # eta
                g = g_eta
                for i in range(4):
                    B_row[3*i] = dN_natural[1, i]      # ∂w/∂η
            for i in range(4):
                B_row[3*i + 1] = -N_vals[i] * g[1]     # θx  (βy = -θx)
                B_row[3*i + 2] = +N_vals[i] * g[0]     # θy  (βx = +θy)
            return B_row, J

        # Bs rows at the four tying points
        Bs_A, _ = _shear_b_at(0.0, -1.0, 'xi')   # γ_ξz at A
        Bs_C, _ = _shear_b_at(0.0,  1.0, 'xi')   # γ_ξz at C
        Bs_B, _ = _shear_b_at(1.0,  0.0, 'eta')  # γ_ηz at B
        Bs_D, _ = _shear_b_at(-1.0, 0.0, 'eta')  # γ_ηz at D
        return (Bs_A, Bs_C, Bs_B, Bs_D)

    @staticmethod
    def _mitc4_Bs_at(tying_rows: tuple, J_inv: np.ndarray,
                     xi: float, eta: float) -> np.ndarray:
        """Interpolate the tying-point shear rows to (ξ,η) and rotate to
        Cartesian. Returns (2,12): [γ_xz; γ_yz] = Bs · [w,θx,θy]×4.

            γ_ξz(ξ,η) = ½(1-η)·γ_ξz|_A + ½(1+η)·γ_ξz|_C
            γ_ηz(ξ,η) = ½(1-ξ)·γ_ηz|_D + ½(1+ξ)·γ_ηz|_B

        The covariant components are e_ξ = g_ξ·γ and e_η = g_η·γ, i.e. e = J·γ
        with the rows of J = ∂(x,y)/∂(ξ,η) being g_ξ and g_η. Hence γ = J⁻¹·e,
        the same map as for the gradient of a shape function.

        J⁻ᵀ is not a substitute: it equals J⁻¹ only when J is symmetric
        (rectangles aligned with the local axes). On skewed or tapered quads it
        rotates and scales the transverse shear: w = x on a parallelogram gives
        γ = (0.5, 1.0) instead of (1, 0).
        """
        Bs_A, Bs_C, Bs_B, Bs_D = tying_rows
        Bs_xi  = 0.5 * (1 - eta) * Bs_A + 0.5 * (1 + eta) * Bs_C
        Bs_eta = 0.5 * (1 - xi)  * Bs_D + 0.5 * (1 + xi)  * Bs_B
        return J_inv @ np.vstack([Bs_xi, Bs_eta])

    # ── MITC3 TRIANGLE (Lee & Bathe 2004) ─────────────────────────

    @staticmethod
    def _mitc3_geometry(coords_local: np.ndarray):
        """Local x, y of the three corners and J = ∂(x,y)/∂(r,s) (rows g_r, g_s)."""
        x = np.asarray(coords_local, float)[:3, 0]
        y = np.asarray(coords_local, float)[:3, 1]
        J = np.array([[x[1] - x[0], y[1] - y[0]],
                      [x[2] - x[0], y[2] - y[0]]])
        return x, y, J

    @staticmethod
    def _mitc3_tying_rows(J: np.ndarray) -> tuple:
        """Covariant transverse-shear rows of MITC3, DOF order [w_i, θx_i, θy_i]×3.

        e_r = ∂w/∂r + g_r·β and e_s = ∂w/∂s + g_s·β with the section rotations
        β = (θy, −θx), as in MITC4. Tying points (Lee & Bathe 2004):
            (1) r = 1/2, s = 0  : e_r      (2) r = 0, s = 1/2 : e_s
            (3) r = s = 1/2     : e_r and e_s
        Assumed field:  ẽ_r = e_r(1) + c·s,  ẽ_s = e_s(2) − c·r,
                        c = e_r(3) − e_r(1) − e_s(3) + e_s(2)
        It gives a constant tangential shear strain along each edge and makes the
        element independent of the node numbering. Returns (e1, e2, c).
        """
        dN = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

        def row(r, s, k):
            N = (1.0 - r - s, r, s)
            g = J[k]
            b = np.zeros(9)
            for i in range(3):
                b[3*i] = dN[k, i]                 # ∂w/∂r or ∂w/∂s
                b[3*i + 1] = -N[i] * g[1]         # θx  (βy = −θx)
                b[3*i + 2] = +N[i] * g[0]         # θy  (βx = +θy)
            return b

        e1 = row(0.5, 0.0, 0)
        e2 = row(0.0, 0.5, 1)
        c = row(0.5, 0.5, 0) - e1 - row(0.5, 0.5, 1) + e2
        return e1, e2, c

    @staticmethod
    def _mitc3_Bs_at(tying: tuple, J_inv: np.ndarray, r: float, s: float) -> np.ndarray:
        """(2, 9): [γ_xz; γ_yz] at (r, s) from the MITC3 assumed covariant field."""
        e1, e2, c = tying
        return J_inv @ np.vstack([e1 + c * s, e2 - c * r])

    @staticmethod
    def _plate_Bb(dN_xy: np.ndarray, n: int) -> np.ndarray:
        """(3, 3n) bending B: κ = [βx,x ; βy,y ; βx,y + βy,x], βx = θy, βy = −θx."""
        Bb = np.zeros((3, 3 * n))
        for i in range(n):
            Bb[0, 3*i + 2] = +dN_xy[0, i]
            Bb[1, 3*i + 1] = -dN_xy[1, i]
            Bb[2, 3*i + 2] = +dN_xy[1, i]
            Bb[2, 3*i + 1] = -dN_xy[0, i]
        return Bb

    def _mitc3_Kb(self, coords_local: np.ndarray, material: MaterialProperty,
                  thickness: float, rebar_factor_b: float = 1.0) -> np.ndarray:
        """MITC3 bending + transverse shear stiffness, 9×9, DOF order
        [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3] — the triangular partner of MITC4.

        Linear w and rotations; constant curvature; transverse shear from the
        MITC3 tying field (_mitc3_tying_rows), integrated with 3 points (exact for
        the quadratic integrand). Reference: Lee, P.S. & Bathe, K.J. (2004),
        'Development of MITC isotropic triangular shell finite elements',
        Computers & Structures 82, 945-962.
        """
        x, y, J = self._mitc3_geometry(coords_local)
        det_J = float(np.linalg.det(J))
        K = np.zeros((9, 9))
        if abs(det_J) < 1e-20:
            return K
        J_inv = np.linalg.inv(J)
        E, nu, t = material.youngs_modulus, material.poissons_ratio, thickness
        Db = self._compute_Db(material, t, rebar_factor=rebar_factor_b)
        Ds = (5.0 / 6.0) * E / (2.0 * (1.0 + nu)) * t * np.eye(2)
        dN_xy = J_inv @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        Bb = self._plate_Bb(dN_xy, 3)
        K += Bb.T @ Db @ Bb * (0.5 * abs(det_J))
        tying = self._mitc3_tying_rows(J)
        for (r, s), w in ShapeFunctions.gauss_points_tri(3):
            Bs = self._mitc3_Bs_at(tying, J_inv, r, s)
            K += Bs.T @ Ds @ Bs * abs(det_J) * w
        return K

    @staticmethod
    def _consistent_bending_stabilizer(coords_local: np.ndarray, n_nodes: int,
                                       k: float) -> np.ndarray:
        """(3n × 3n) out-of-plane stabiliser for elements WITHOUT bending
        stiffness (membrane-only), in the [w_i, θx_i, θy_i] slots.

        Springs to ground on w, θx and θy would not do: a spring on w resists a
        rigid translation along the element normal, so part of the load would go
        into the springs instead of the supports. This stabiliser penalises only
        what a rigid motion cannot produce:
            w   : the deviation of the nodal w from their best-fit plane
            θx  : θx_i − ∂w/∂y ,  θy : θy_i + ∂w/∂x  (rigid rotation: θx = w,y,
                  θy = −w,x, with the gradient of the best-fit plane)
        so it stores no energy under any rigid motion and keeps K non-singular.
        """
        xy = np.asarray(coords_local, float)[:n_nodes, :2]
        C = np.column_stack([np.ones(n_nodes), xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()])
        H = np.linalg.pinv(C)                    # (3, n): plane coefficients from w
        Pw = C @ H                               # projector onto planar w fields
        Bw = np.zeros((n_nodes, 3 * n_nodes))
        Br = np.zeros((2 * n_nodes, 3 * n_nodes))
        for i in range(n_nodes):
            Br[2*i, 3*i + 1] = 1.0
            Br[2*i + 1, 3*i + 2] = 1.0
            for j in range(n_nodes):
                Bw[i, 3*j] = (1.0 if i == j else 0.0) - Pw[i, j]
                Br[2*i, 3*j] -= H[2, j]          # − ∂w/∂y
                Br[2*i + 1, 3*j] += H[1, j]      # + ∂w/∂x
        return k * (Bw.T @ Bw + Br.T @ Br)

    # ── ELEMENT STIFFNESS DISPATCH ──────────────────────────────────────

    def _compute_truss_stiffness(self, elem: 'FEElement') -> Tuple[np.ndarray, float, np.ndarray]:
        """2-node truss element stiffness in 3D global frame.
        
        Local axial stiffness:  K_axial = (EA/L) * [[1,-1],[-1,1]]
        Global 6×6 K in (ux1,uy1,uz1, ux2,uy2,uz2) ordering via direction cosines.
        
        Returns:
            K (6, 6) global stiffness in translation DOFs
            L (float) length of truss
            axis (3,) unit vector along truss
        """
        if (getattr(elem.element_type, 'name', '') != 'TRUSS_T2'):
            raise ValueError(f"Element {elem.id} is not a TRUSS_T2")
        if len(elem.node_ids) != 2:
            raise ValueError(f"TRUSS_T2 requires 2 nodes, got {len(elem.node_ids)}")
        
        # Get section area
        section = self.converter.sections.get(elem.section_id, None) if elem.section_id else None
        if section is None:
            raise ValueError(f"Truss {elem.id} has no section data (section_id={elem.section_id})")
        area = float(section.get('area', 0.0))
        if area <= 0:
            raise ValueError(f"Truss {elem.id} has invalid area {area}")
        
        # Get material (use first material as default)
        material = self.converter.materials.get(elem.material_id, self.converter.materials[0])
        E = material.youngs_modulus
        
        # Geometry
        coords = self._get_element_coords(elem)  # (2, 3)
        vec = coords[1] - coords[0]
        L = float(np.linalg.norm(vec))
        if L < 1e-15:
            raise ValueError(f"Truss {elem.id} has zero length")
        cx, cy, cz = vec[0]/L, vec[1]/L, vec[2]/L
        
        # Transformation: T (2×6) maps global translation DOFs → local axial DOFs
        T = np.array([
            [cx, cy, cz, 0,  0,  0],
            [0,  0,  0,  cx, cy, cz]
        ])
        k_axial = (E * area / L) * np.array([[1, -1], [-1, 1]])
        K_global_trans = T.T @ k_axial @ T
        return K_global_trans, L, np.array([cx, cy, cz])

    # ── BEAM ELEMENT ───────────────────────────────────────────────

    def _compute_beam_stiffness(self, elem: 'FEElement'):
        """BEAM_B2: global 12 × 12 stiffness (6 DOF per node) and the pieces stress recovery needs:
        (K_global, L, R = local axes as rows, k_local, T)."""
        sec = self.converter.sections.get(elem.section_id) if elem.section_id else None
        if sec is None or not sec.get('beam'):
            raise ValueError(f"Beam {elem.id} has no beam section data (section_id={elem.section_id})")
        bp = sec['beam']
        if min(bp['A'], bp['Iy'], bp['Iz'], bp['J']) <= 0.0:
            raise ValueError(f"Beam {elem.id}: section '{sec.get('name')}' needs a shape, or A, Iy, Iz and J > 0")
        m = self.converter.materials.get(elem.material_id, self.converter.materials[0])
        E = float(m.youngs_modulus); G = E / (2.0 * (1.0 + float(m.poissons_ratio)))
        X = self._get_element_coords(elem); v = X[1] - X[0]; L = float(np.linalg.norm(v))
        if L < 1e-12:
            raise ValueError(f"Beam {elem.id} has zero length")
        R = _perm_beam_axes(v / L, bp['roll'])
        k = _perm_beam_k_local(E, G, bp, L)
        T = np.zeros((12, 12))
        for b_ in range(4):
            T[3 * b_:3 * b_ + 3, 3 * b_:3 * b_ + 3] = R
        return T.T @ k @ T, L, R, k, T

    def _beam_results(self, eid, elem) -> Optional[dict]:
        """Internal forces (local axes, positive-face convention: N > 0 tension) at both ends and at mid-length,
        and the extreme normal stresses of the three stations."""
        _, L, R, k, T = self._compute_beam_stiffness(elem)
        bp = self.converter.sections[elem.section_id]['beam']
        ue = np.concatenate([self.results.displacements[self.node_id_to_index[n]] for n in elem.node_ids])
        feq = (getattr(self, '_beam_feq', None) or {}).get(eid, np.zeros(12))
        f = k @ (T @ ue) - T @ feq                     # forces of the two nodes on the element (local)
        q = R @ (2.0 * feq[0:3] / L)                   # uniform load per length on the element (local)
        Si, Sj = -f[0:6], f[6:12]
        rx = np.array([-0.5 * L, 0.0, 0.0]); rq = np.array([-0.25 * L, 0.0, 0.0])
        Sm = np.r_[-f[0:3] - 0.5 * L * q, -f[3:6] - np.cross(rx, f[0:3]) - np.cross(rq, 0.5 * L * q)]
        sig = [_perm_beam_sigma(S, bp) for S in (Si, Sm, Sj)]
        return dict(i=Si, mid=Sm, j=Sj, L=L, axes=R, smax=max(s[0] for s in sig), smin=min(s[1] for s in sig),
                    section=self.converter.sections[elem.section_id].get('name', ''))

    # ── CONNECTIONS (fixed / pinned / hinge) ───────────────────────

    def _joint_penalty(self, blocks):
        """Penalty stiffness of links (m, s, W): g = [u_s - u_m - θ_m × r ; θ_s - θ_m], energy ½ gᵀ W g (W 6 × 6, global)."""
        if not blocks:
            return None
        P = self.node_coords; D = self.DOFS_PER_NODE
        rows, cols, vals = [], [], []
        for m, sl, W in blocks:
            B = self._link_B(P[sl] - P[m])
            Kp = B.T @ W @ B
            dofs = np.r_[m * D + np.arange(6), sl * D + np.arange(6)]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            nz = np.abs(Kp) > 0
            rows.append(rr[nz]); cols.append(cc[nz]); vals.append(Kp[nz])
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.n_dofs, self.n_dofs)).tocsr()

    @staticmethod
    def _link_B(r):
        Rx = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        B = np.zeros((6, 12))
        B[0:3, 0:3] = -np.eye(3); B[0:3, 3:6] = Rx; B[0:3, 6:9] = np.eye(3)
        B[3:6, 3:6] = -np.eye(3); B[3:6, 9:12] = np.eye(3)
        return B

    def _connection_ties(self):
        """Add the connection links to K (before the supports, so they are part of the structure)."""
        conns = getattr(self.converter, '_connections', None) or []
        self._joint_iface = {}
        if not conns:
            return
        D = self.DOFS_PER_NODE; P = self.node_coords
        dg = getattr(self, '_diag_structure', None)
        if dg is None:
            dg = np.abs(self._K_global.diagonal()).reshape(-1, D)
        kt = PERM_CONN_PENALTY * float(dg[:, :3].max()); kr = PERM_CONN_PENALTY * float(dg[:, 3:].max())
        I3 = np.eye(3)

        def W(rot):
            w = np.zeros((6, 6)); w[:3, :3] = kt * I3; w[3:, 3:] = rot
            return w
        blocks = []
        from scipy.spatial import cKDTree
        for c in conns:
            A = [self.node_id_to_index[n] for n in c['A'] if n in self.node_id_to_index]
            B = [self.node_id_to_index[n] for n in c['B'] if n in self.node_id_to_index]
            if c['ctype'] == 'HINGE':
                a = np.asarray(c['axis'], float); Wl = W(kr * (I3 - np.outer(a, a)))
            elif c['ctype'] == 'PINNED':
                Wl = W(np.zeros((3, 3)))
            else:
                Wl = W(kr * I3)
            iface = []
            if c['method'] == 'pairs':
                if A and B:
                    aset = set(A); tree = cKDTree(P[A])
                    for s in B:
                        if s in aset:
                            continue                       # merged with side A: continuous already
                        _, kk = tree.query(P[s]); m = A[int(kk)]
                        blocks.append((m, s, Wl)); iface.append((m, s, Wl))
            else:
                ma, mb = self.node_id_to_index[c['MA']], self.node_id_to_index[c['MB']]
                Wr = W(kr * I3)
                blocks += [(ma, s, Wr) for s in A] + [(mb, s, Wr) for s in B]
                blocks.append((ma, mb, Wl)); iface.append((ma, mb, Wl))
            if c['method'] == 'pairs' and not iface:
                c['method'] = 'merged'                     # every side-B node is a side-A node: continuous already
            self._joint_iface[c['name']] = iface
        Kp = self._joint_penalty(blocks)
        if Kp is not None:
            self._K_global = (self._K_global + Kp).tocsr()
        print(f"[PERM] connections: {len(conns)} joint(s), {len(blocks)} link(s) "
              f"({sum(len(v) for v in self._joint_iface.values())} across the joints)")

    def _connection_forces(self) -> None:
        """Force and moment that side A transmits to side B at each joint (global axes, about the joint centre)."""
        conns = getattr(self.converter, '_connections', None) or []
        out = {}
        if not conns or self.results.displacements is None:
            self.results.connection_forces = out
            return
        U = self.results.displacements; P = self.node_coords
        for c in conns:
            F = np.zeros(3); M = np.zeros(3); ctr = np.asarray(c['centre'], float)
            iface = (getattr(self, '_joint_iface', None) or {}).get(c['name'], [])
            for m, sl, Wl in iface:
                g = self._link_B(P[sl] - P[m]) @ np.r_[U[m], U[sl]]
                f = -(Wl @ g)                                 # link force on the side-B node
                F += f[:3]; M += f[3:] + np.cross(P[sl] - ctr, f[:3])
            ax = c.get('axis')
            # relative rotation of side B against side A across the joint (the released rotation, if any)
            rot = 0.0
            if c['method'] == 'point':
                ma, mb = self.node_id_to_index[c['MA']], self.node_id_to_index[c['MB']]
                rot = float(np.linalg.norm(U[mb, 3:6] - U[ma, 3:6]))
            elif iface:
                rot = max(float(np.linalg.norm(U[sl, 3:6] - U[m, 3:6])) for m, sl, _ in iface)
            out[c['name']] = dict(F=F, M=M, M_axis=(float(M @ np.asarray(ax, float)) if ax is not None else None),
                                  links=len(iface), ctype=c['ctype'], method=c['method'], centre=ctr,
                                  axis=(None if ax is None else np.asarray(ax, float)), nA=len(c['A']), nB=len(c['B']),
                                  split=c.get('split', 0), rot=rot)
            s = (f"[PERM] joint '{c['name']}' ({c['ctype']}, {c['method']}): F = ({F[0]/1e3:+.3f}, {F[1]/1e3:+.3f}, "
                 f"{F[2]/1e3:+.3f}) kN, M = ({M[0]/1e3:+.3f}, {M[1]/1e3:+.3f}, {M[2]/1e3:+.3f}) kN·m")
            if ax is not None:
                s += f", about the hinge axis {out[c['name']]['M_axis']/1e3:+.4f} kN·m"
            if c['ctype'] != 'FIXED':
                s += f", turns {np.degrees(rot):.3f}°"
            print(s)
        big = [(n_, v['rot']) for n_, v in out.items() if v['ctype'] != 'FIXED' and v['rot'] > PERM_CONN_ROT_WARN]
        if big:
            print(f"[PERM] ⚠ {len(big)} joint(s) turn more than {np.degrees(PERM_CONN_ROT_WARN):.2f}° "
                  f"(largest {np.degrees(max(r for _, r in big)):.2f}° at '{max(big, key=lambda t: t[1])[0]}'): a released "
                  f"joint may leave a part free to turn - a mechanism. Pinned joints that all lie on one line let the part "
                  f"turn about that line (posts in a row topple sideways); use HINGE about an axis the rest of the structure "
                  f"holds, or FIXED. Linear results are not valid for a part that is free to move.")
        self.results.connection_forces = out

    def _stabilize_truss_only_nodes(self, rows_list, cols_list, vals_list):
        """For nodes connected only to trusses (no shells), add tiny stiffness to:
          - Rotational DOFs (θx, θy): trusses don't contribute to these
          - Translation DOFs perpendicular to all incident truss axes
        
        Stabilization scale: E × 1e-9 — small enough not to affect real results,
        large enough to keep K invertible.
        
        Without this, singularity in K causes NaN solution.
        """
        shell_nodes = set()
        truss_nodes = set()
        node_truss_axes = {}  # nid → list of axis unit vectors
        for eid, elem in self.converter.elements.items():
            if getattr(elem, 'viz_only', False):
                continue
            if (getattr(elem.element_type, 'name', '') == 'TRUSS_T2'):
                truss_nodes.update(elem.node_ids)
                try:
                    _, _, axis = self._compute_truss_stiffness(elem)
                    for nid in elem.node_ids:
                        node_truss_axes.setdefault(nid, []).append(axis)
                except Exception:
                    pass
            else:  # shell types
                shell_nodes.update(elem.node_ids)
        
        truss_only_nodes = truss_nodes - shell_nodes
        if not truss_only_nodes:
            return
        
        # Use the first material's E as the reference scale
        material = self.converter.materials[0]
        stab = material.youngs_modulus * 1e-9
        
        for nid in truss_only_nodes:
            if nid not in self.node_id_to_index:
                continue
            idx = self.node_id_to_index[nid]
            base = idx * self.DOFS_PER_NODE
            # Stabilize rotational DOFs (θx, θy, θz) — trusses never contribute
            for d in (3, 4, 5):
                rows_list.append(base + d)
                cols_list.append(base + d)
                vals_list.append(stab)
            # Stabilize translation DOFs that have no truss axis along them
            axes = node_truss_axes.get(nid, [])
            for d in range(3):
                has_stiffness = False
                for axis in axes:
                    if abs(axis[d]) > 0.01:
                        has_stiffness = True
                        break
                if not has_stiffness:
                    rows_list.append(base + d)
                    cols_list.append(base + d)
                    vals_list.append(stab)
        print(f"[PERM Solver] Stabilized {len(truss_only_nodes)} truss-only nodes")

    def _compute_element_stiffness(self, elem: FEElement) -> Tuple[np.ndarray, np.ndarray]:
        """Compute element K and return (K_global, T_block_3x3).

        K_global:    stiffness in the GLOBAL frame at DOFS_PER_NODE = 6 per node,
                     ordered [ux,uy,uz,θx,θy,θz] -- 24x24 for QUAD4. The element
                     is formed on 5 local DOFs and expanded to 6 by
                     _transform_K_to_global, which adds the Hughes-Brezzi
                     drilling term for θz.
        T_block_3x3: rotation matrix from local to global (for stress recovery).
        """
        coords = self._get_element_coords(elem)
        material = self.converter.materials.get(elem.material_id,
                                                  self.converter.materials[0])
        etype = elem.element_type
        etype_name = etype.name if hasattr(etype, 'name') else str(etype)
        t = elem.thickness

        n_nodes_elem = {'TRI3': 3, 'TRI6': 6, 'QUAD4': 4, 'QUAD8': 8}.get(etype_name)
        if n_nodes_elem is None:
            raise ValueError(f"Unsupported element type: {etype}")

        # Build the local frame from the CORNER nodes (3 tri / 4 quad), then map
        # EVERY node into it — TRI6/QUAD8 membrane integration needs all 6/8
        # rows, and slicing only the corners silently produced a shape mismatch
        # that got swallowed as an element-skipped error.
        n_corner = 3 if etype_name.startswith('TRI') else 4
        _, T = self._compute_local_system(coords[:n_corner])
        local_coords = (T.T @ (coords - coords[0]).T).T

        # Transformed-section multipliers for smeared reinforcement (1.0 = plain)
        f_m, f_b = self._rebar_factors(elem, material.youngs_modulus, t)

        # Membrane stiffness. QUAD4: QM6 incompatible modes (in-plane bending
        # without locking — critical for webs carrying beam bending), condensed
        # together with the drilling penalty (_membrane_drilling_K_qm6).
        Kmd = None
        if etype_name == 'QUAD4':
            Kmd = self._membrane_drilling_K_qm6(local_coords, material, t, rebar_factor=f_m)
            Km = np.zeros((8, 8))
        else:
            Km = self._compute_membrane_K(local_coords, material, t, etype_name,
                                          n_nodes_elem, rebar_factor=f_m)

        # Bending + transverse shear: MITC4 for QUAD4, MITC3 for TRI3.
        # TRI6/QUAD8 have no bending formulation: raise, so the assembly takes
        # the membrane-only path and counts and reports them.
        if etype_name == 'QUAD4':
            Kb = self._mitc4_Kb(local_coords[:4], material, t, rebar_factor_b=f_b)
        elif etype_name == 'TRI3':
            Kb = self._mitc3_Kb(local_coords[:3], material, t, rebar_factor_b=f_b)
        else:
            raise NotImplementedError(
                f"{etype_name} has no bending formulation: membrane only")

        # Assemble into 5-DOF/node local stiffness, then expand to 6/node and
        # add the membrane-drilling block (QUAD4) or the drilling constraint.
        K_local = self._assemble_membrane_bending(Km, Kb, n_nodes_elem)
        K6 = self._expand_5_to_6(K_local, n_nodes_elem)
        if Kmd is not None:
            K6 += self._scatter_membrane_drilling(Kmd, n_nodes_elem)
        elif etype_name in ('QUAD4', 'TRI3'):
            K6 += self._drilling_K_hb(local_coords, material, t, n_nodes_elem)
        else:
            K6 = self._stabilize_drilling_diag(K_local, K6, n_nodes_elem)

        # Transform to global 6-DOF/node frame, then link the flat element to
        # the true node positions (shell offset and warping correction)
        K_global = self._transform_K_to_global(K6, T, n_nodes_elem)
        return self._with_offset(elem, K_global, T, local_coords), T

    def _compute_element_stiffness_bending_only(self, elem: FEElement) -> Tuple[np.ndarray, np.ndarray]:
        """QUAD4 PLATE: MITC4 bending + transverse shear + drilling, no membrane.

        The exact complement of _compute_element_stiffness_membrane_only. That
        one keeps QM6 and drops MITC4; this one keeps MITC4 and drops QM6, with
        nothing put in its place -- the membrane block is identically zero.

        WHAT THIS ELEMENT IS. A Reissner-Mindlin PLATE, not a shell. It resists
        being bent out of its plane and it resists transverse shear. It does
        not resist being stretched, sheared or bent WITHIN its plane at all.

        WHAT IT COSTS, measured on a free-free unit element:

            QM6 + MITC4 + drilling :  6 near-zero eigenvalues  (the 6 rigid-body
                                      modes -- correct)
            MITC4 + drilling only  : 11 near-zero eigenvalues  (6 rigid-body +
                                      5 spurious)

        The five extra are the membrane deformation modes. Drilling ties
        theta_z to omega = (dv/dx - du/dy)/2, so it does touch u and v -- but
        only their skew-symmetric part. The symmetric part (stretch and
        in-plane shear) keeps zero energy, and no amount of drilling penalty
        recovers it.

        WHEN IT IS VALID. A flat plate carrying transverse load, where the
        in-plane DOFs are simply unloaded. Verified: a 8x2 cantilever plate
        gives uz identical to the full shell to 2.8e-11 relative -- QM6
        contributes nothing to out-of-plane bending, so removing it changes
        that answer not at all.

        WHEN IT IS NOT. Anything folded. A box girder or a bridge deck with
        webs carries global bending as membrane action of the flanges; this
        element deletes that load path. Any in-plane load component -- gravity
        on a vertical web included -- meets zero stiffness. assemble_global_
        stiffness counts these elements and says so on every run.
        """
        coords = self._get_element_coords(elem)
        material = self.converter.materials.get(elem.material_id,
                                                  self.converter.materials[0])
        etype_name = getattr(elem.element_type, 'name', str(elem.element_type))
        if etype_name != 'QUAD4':
            raise NotImplementedError(
                f"Plate (MITC4-only) is a QUAD4 formulation; {etype_name} has no "
                f"validated bending element. Re-mesh those faces as quads.")
        t = elem.thickness
        _, T = self._compute_local_system(coords[:4])
        local_coords = (T.T @ (coords - coords[0]).T).T
        _, f_b = self._rebar_factors(elem, material.youngs_modulus, t)

        # Membrane block identically zero -- this is the whole point.
        Km = np.zeros((8, 8))
        Kb = self._mitc4_Kb(local_coords[:4], material, t, rebar_factor_b=f_b)

        K_local = self._assemble_membrane_bending(Km, Kb, 4)
        K6 = self._expand_5_to_6(K_local, 4)
        # Drilling still required: without a theta_z term every node carries a
        # zero on the diagonal and the assembled matrix is singular.
        K6 += self._drilling_K_hb(local_coords, material, t, 4)
        K_global = self._transform_K_to_global(K6, T, 4)
        return self._with_offset(elem, K_global, T, local_coords), T

    def _compute_element_stiffness_membrane_only(self, elem: FEElement) -> Tuple[np.ndarray, np.ndarray]:
        """Membrane-only element stiffness: faces assigned 'Membrane shell', and
        TRI6/QUAD8, which have no bending formulation. The out-of-plane DOFs get
        a tiny rigid-body-consistent stabiliser (see
        _consistent_bending_stabilizer) so K stays non-singular.
        """
        coords = self._get_element_coords(elem)
        material = self.converter.materials.get(elem.material_id,
                                                  self.converter.materials[0])
        etype = elem.element_type
        etype_name = etype.name if hasattr(etype, 'name') else str(etype)
        t = elem.thickness
        n_nodes_elem = {'TRI3': 3, 'TRI6': 6, 'QUAD4': 4, 'QUAD8': 8}.get(etype_name)
        n_corner = 3 if etype_name.startswith('TRI') else 4
        _, T = self._compute_local_system(coords[:n_corner])
        local_coords = (T.T @ (coords - coords[0]).T).T
        f_m, _ = self._rebar_factors(elem, material.youngs_modulus, t)
        Kmd = None
        if etype_name == 'QUAD4':
            Kmd = self._membrane_drilling_K_qm6(local_coords, material, t, rebar_factor=f_m)
            Km = np.zeros((8, 8))
            diag_m = max(float(np.max(np.abs(np.diag(Kmd)[[0, 1, 3, 4, 6, 7, 9, 10]]))), 1.0)
        else:
            Km = self._compute_membrane_K(local_coords, material, t, etype_name,
                                          n_nodes_elem, rebar_factor=f_m)
            diag_m = max(float(np.max(np.abs(np.diag(Km)))), 1.0)
        Kb_stab = self._consistent_bending_stabilizer(local_coords, n_nodes_elem, diag_m * 1e-8)
        K_local = self._assemble_membrane_bending(Km, Kb_stab, n_nodes_elem)
        K6 = self._expand_5_to_6(K_local, n_nodes_elem)
        if Kmd is not None:
            K6 += self._scatter_membrane_drilling(Kmd, n_nodes_elem)
        elif etype_name in ('QUAD4', 'TRI3'):
            K6 += self._drilling_K_hb(local_coords, material, t, n_nodes_elem)
        else:
            K6 = self._stabilize_drilling_diag(K_local, K6, n_nodes_elem)
        K_global = self._transform_K_to_global(K6, T, n_nodes_elem)
        return self._with_offset(elem, K_global, T, local_coords), T

    def _compute_membrane_K(self, coords_local: np.ndarray, material: MaterialProperty,
                            thickness: float, etype_name: str, n_nodes: int,
                            rebar_factor: float = 1.0) -> np.ndarray:
        """Membrane K in local frame, size (2n × 2n) for DOFs [ux1,uy1, ux2,uy2,...]"""
        Dm = self._compute_Dm(material, thickness, rebar_factor=rebar_factor)

        if etype_name == 'TRI3':
            dN_func = ShapeFunctions.tri3_dN
            gauss = ShapeFunctions.gauss_points_tri(1)
        elif etype_name == 'TRI6':
            dN_func = ShapeFunctions.tri6_dN
            gauss = ShapeFunctions.gauss_points_tri(3)
        elif etype_name == 'QUAD4':
            dN_func = ShapeFunctions.quad4_dN
            gauss = ShapeFunctions.gauss_points_quad(2)
        elif etype_name == 'QUAD8':
            dN_func = ShapeFunctions.quad8_dN
            gauss = ShapeFunctions.gauss_points_quad(3)
        else:
            raise ValueError(f"Unsupported: {etype_name}")

        Km = np.zeros((2 * n_nodes, 2 * n_nodes))
        for (xi, eta), w in gauss:
            dN = dN_func(xi, eta)
            J = dN @ coords_local[:n_nodes, :2]
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-20:
                continue
            J_inv = np.linalg.inv(J)
            dN_phys = J_inv @ dN
            Bm = self._build_Bm(dN_phys, n_nodes)
            Km += Bm.T @ Dm @ Bm * abs(det_J) * w
        return Km

    def _membrane_drilling_K_qm6(self, coords_local: np.ndarray, material: MaterialProperty,
                                 thickness: float, rebar_factor: float = 1.0) -> np.ndarray:
        """QUAD4 membrane (QM6) and Hughes-Brezzi drilling condensed together.
        Returns 12×12 in the local DOF order [u1, v1, θz1, ..., u4, v4, θz4].

        The drilling penalty γ ∫ (θz − ω)² dA compares θz with the in-plane
        rotation ω = ½(∂v/∂x − ∂u/∂y) of the FULL QM6 field, incompatible modes
        included, and the modes are condensed out of both terms at once.

        Why the full field: with ω from the bilinear part only, the bilinear v
        misses the x² term that the incompatible mode supplies in pure in-plane
        bending, so ω_bilinear = κx/2 while the true rotation (and θz) is κx. The
        penalty then fights the bending mode unless γ is very small (MacNeal 5:1
        cantilever: γ = 0.1 → 0.801, γ = 1 → 0.293), and a very small γ leaves
        the drilling rotation nearly free: where facets meet at an angle
        (curved, twisted or rough scanned surfaces) bending moments are not
        carried across properly and thick shells come out too flexible (MacNeal
        twisted beam, t = 0.32: 1.32 × the reference with γ = 1e-3). With ω from
        the enriched field, pure bending of a rectangle stores no drilling
        energy at any γ, so γ = 0.1 is used (see DRILL_GAMMA).
        """
        Dm = self._compute_Dm(material, thickness, rebar_factor=rebar_factor)
        E, nu = material.youngs_modulus, material.poissons_ratio
        gamma = self.DRILL_GAMMA * E / (2.0 * (1.0 + nu)) * thickness
        xy = np.asarray(coords_local, float)[:4, :2]
        J0 = ShapeFunctions.quad4_dN(0.0, 0.0) @ xy
        det_J0 = float(np.linalg.det(J0))
        use_inc = abs(det_J0) > 1e-20           # degenerate element: plain Q4 + drilling
        J0_inv = np.linalg.inv(J0) if use_inc else None
        Kqq = np.zeros((12, 12)); Kqa = np.zeros((12, 4)); Kaa = np.zeros((4, 4))
        for (xi, eta), w in ShapeFunctions.gauss_points_quad(2):
            dN = ShapeFunctions.quad4_dN(xi, eta)
            J = dN @ xy
            det_J = float(np.linalg.det(J))
            if abs(det_J) < 1e-20:
                continue
            dNp = np.linalg.inv(J) @ dN
            N = ShapeFunctions.quad4_N(xi, eta)
            Bu = np.zeros((3, 12)); bq = np.zeros(12)
            for i in range(4):
                Bu[0, 3*i] = dNp[0, i]
                Bu[1, 3*i + 1] = dNp[1, i]
                Bu[2, 3*i] = dNp[1, i]
                Bu[2, 3*i + 1] = dNp[0, i]
                bq[3*i] = +0.5 * dNp[1, i]       # θz − ω : −∂ω/∂u_i
                bq[3*i + 1] = -0.5 * dNp[0, i]   #          −∂ω/∂v_i
                bq[3*i + 2] = N[i]               #          ∂θz/∂θz_i
            jw = abs(det_J) * w
            Kqq += (Bu.T @ Dm @ Bu + gamma * np.outer(bq, bq)) * jw
            if use_inc:
                # incompatible modes N5 = 1−ξ², N6 = 1−η², Taylor-corrected gradients
                dNi = (det_J0 / det_J) * (J0_inv @ np.array([[-2.0 * xi, 0.0],
                                                             [0.0, -2.0 * eta]]).T)
                Ba = np.zeros((3, 4)); ba = np.zeros(4)
                for m in range(2):
                    Ba[0, 2*m] = dNi[0, m]
                    Ba[1, 2*m + 1] = dNi[1, m]
                    Ba[2, 2*m] = dNi[1, m]
                    Ba[2, 2*m + 1] = dNi[0, m]
                    ba[2*m] = +0.5 * dNi[1, m]
                    ba[2*m + 1] = -0.5 * dNi[0, m]
                Kqa += (Bu.T @ Dm @ Ba + gamma * np.outer(bq, ba)) * jw
                Kaa += (Ba.T @ Dm @ Ba + gamma * np.outer(ba, ba)) * jw
        if use_inc:
            try:
                return Kqq - Kqa @ np.linalg.solve(Kaa, Kqa.T)
            except np.linalg.LinAlgError:
                pass
        return Kqq

    @staticmethod
    def _scatter_membrane_drilling(Kmd: np.ndarray, n_nodes: int) -> np.ndarray:
        """Place the [u, v, θz] membrane-drilling block into the 6-DOF/node local K."""
        K = np.zeros((6 * n_nodes, 6 * n_nodes))
        idx = np.array([[6*i, 6*i + 1, 6*i + 5] for i in range(n_nodes)]).ravel()
        K[np.ix_(idx, idx)] = Kmd
        return K

    def _drilling_K_hb(self, coords_local: np.ndarray,
                       material: MaterialProperty, thickness: float,
                       n_nodes: int) -> np.ndarray:
        """Hughes-Brezzi drilling constraint, returned as (6n x 6n) LOCAL.

        Penalises the mismatch between the drilling DOF θz and the true in-plane
        rotation of the membrane field,  ω = ½(∂v/∂x - ∂u/∂y):

            K_drill = γ ∫ (θz - ω)ᵀ (θz - ω) dA ,   γ = DRILL_GAMMA · G · t

        Unlike a bare diagonal penalty k·θz², this is RIGID-BODY CONSISTENT: a
        rigid rotation about the element normal gives θz = ω exactly, so it
        stores no energy. A bare diagonal term instead resists genuine rigid
        rotation wherever an element normal has a component along the rotation
        axis — harmless on a flat mesh (there the drilling DOF is decoupled and
        simply stays zero) but a real spurious stiffness at every fold.

        It also ties θz to the membrane, so — unlike penalising only the
        deviation from the element-mean drilling — it leaves NO null mode on a
        flat region whose rotational DOFs are unconstrained (e.g. a simply
        supported slab).

        Reference: Hughes & Brezzi (1989), "On drilling degrees of freedom",
        CMAME 72, 105-121.
        """
        K = np.zeros((6 * n_nodes, 6 * n_nodes))
        E, nu = material.youngs_modulus, material.poissons_ratio
        G = E / (2.0 * (1.0 + nu))
        gamma = self.DRILL_GAMMA * G * thickness

        n_geom = 3 if n_nodes == 3 else 4
        if n_geom == 3:
            # Linear triangle: constant gradients, 3-point interior rule.
            pts = [((1/6, 1/6), 1/3), ((2/3, 1/6), 1/3), ((1/6, 2/3), 1/3)]
            N_of = lambda xi, eta: np.array([1 - xi - eta, xi, eta])
            dN_nat = ShapeFunctions.tri3_dN(0.0, 0.0)
        else:
            pts = ShapeFunctions.gauss_points_quad(2)
            N_of = ShapeFunctions.quad4_N
            dN_nat = None

        xy = coords_local[:n_geom, :2]

        for (xi, eta), w in pts:
            dN = dN_nat if dN_nat is not None else ShapeFunctions.quad4_dN(xi, eta)
            N = N_of(xi, eta)
            J = dN @ xy
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-20:
                continue
            dN_phys = np.linalg.inv(J) @ dN
            # Triangle rule weights are fractions of the element AREA
            # (= |detJ|/2), the quad rule integrates over |detJ| directly.
            jw = abs(det_J) * w * (0.5 if n_geom == 3 else 1.0)
            B = np.zeros(6 * n_nodes)
            for i in range(n_geom):
                B[6*i + 0] = +0.5 * dN_phys[1, i]   # -∂ω/∂u_i
                B[6*i + 1] = -0.5 * dN_phys[0, i]   # -∂ω/∂v_i
                B[6*i + 5] = N[i]                   #  ∂θz_h/∂θz_i
            K += gamma * np.outer(B, B) * jw
        return K

    def _assemble_membrane_bending(self, Km: np.ndarray, Kb: Optional[np.ndarray],
                                    n_nodes: int) -> np.ndarray:
        """Combine membrane (2n×2n, DOFs ux,uy) and bending (3n×3n, DOFs w,θx,θy)
        into local-frame element stiffness of size (5n × 5n) with DOF ordering
        [ux,uy,uz,θx,θy] per node.
        
        If Kb is None (element type not yet bending-capable), bending block is zero
        plus a small numerical stabilizer to keep K nonsingular.
        """
        n_dof = 5 * n_nodes
        K = np.zeros((n_dof, n_dof))

        # Membrane: insert Km rows/cols into positions (5i+0, 5i+1) per node
        for I in range(n_nodes):
            for J in range(n_nodes):
                K[5*I  , 5*J  ] = Km[2*I  , 2*J  ]
                K[5*I  , 5*J+1] = Km[2*I  , 2*J+1]
                K[5*I+1, 5*J  ] = Km[2*I+1, 2*J  ]
                K[5*I+1, 5*J+1] = Km[2*I+1, 2*J+1]

        # Bending: insert Kb rows/cols into positions (5i+2, 5i+3, 5i+4) per node
        if Kb is not None:
            for I in range(n_nodes):
                for J in range(n_nodes):
                    for a in range(3):
                        for b in range(3):
                            K[5*I + 2 + a, 5*J + 2 + b] = Kb[3*I + a, 3*J + b]
        else:
            # Small stabilizer on bending DOFs only — for elements without bending
            # formulation. Keeps K nonsingular but doesn't contribute to membrane
            # response. Magnitude tied to membrane diagonal to be conditioning-safe.
            diag_m = max(np.max(np.abs(np.diag(Km))), 1.0)
            stab = diag_m * 1e-8
            for I in range(n_nodes):
                K[5*I + 2, 5*I + 2] = stab
                K[5*I + 3, 5*I + 3] = stab
                K[5*I + 4, 5*I + 4] = stab

        return K

    @staticmethod
    def _expand_5_to_6(K_local: np.ndarray, n_nodes: int) -> np.ndarray:
        """Expand a 5-DOF/node local K [ux,uy,uz,θx,θy] to 6-DOF/node by
        inserting an empty θz (drilling) row/column per node."""
        R = np.zeros((6 * n_nodes, 5 * n_nodes))
        for i in range(n_nodes):
            R[6*i:6*i+3, 5*i:5*i+3] = np.eye(3)   # translations
            R[6*i+3, 5*i+3] = 1.0                 # θx_l
            R[6*i+4, 5*i+4] = 1.0                 # θy_l
            # 6*i+5 (θz_l drilling) intentionally left zero — filled by
            # _drilling_K_hb (QUAD4) or _stabilize_drilling_diag (fallback).
        return R @ K_local @ R.T

    def _stabilize_drilling_diag(self, K5: np.ndarray, K6: np.ndarray,
                                 n_nodes: int) -> np.ndarray:
        """Fallback drilling stabiliser for element types without a proper
        Hughes-Brezzi constraint (TRI6/QUAD8, membrane-only). Bare diagonal
        penalty: NOT rigid-body consistent, so it adds a small spurious stiffness
        at folds. Acceptable only because these elements carry no bending here;
        QUAD4 and TRI3, the elements for shells in bending, take the correct
        path (_membrane_drilling_K_qm6, _drilling_K_hb).
        """
        _rot = [abs(K5[5*i+3, 5*i+3]) for i in range(n_nodes)]
        _rot += [abs(K5[5*i+4, 5*i+4]) for i in range(n_nodes)]
        scale = max(_rot) if _rot else 0.0
        if scale <= 0.0:
            scale = max(float(np.max(np.abs(np.diag(K5)))), 1.0)
        k_drill = scale * self.DRILL_FALLBACK
        for i in range(n_nodes):
            K6[6*i+5, 6*i+5] += k_drill
        return K6

    def _transform_K_to_global(self, K6_local: np.ndarray, T: np.ndarray,
                                n_nodes: int) -> np.ndarray:
        """Rotate a 6-DOF/node LOCAL element K into the global frame.

        Translations and rotations are both rotated by T, so all three global
        rotations are kept — nothing is dropped or projected. A fold node
        therefore carries a genuine 3-component rotation vector: the plate that
        needs a given rotation supplies real stiffness, and its neighbour
        supplies the drilling constraint about its own normal.
        """
        T6 = np.zeros((6 * n_nodes, 6 * n_nodes))
        for i in range(n_nodes):
            b = 6 * i
            T6[b:b+3, b:b+3] = T
            T6[b+3:b+6, b+3:b+6] = T
        return T6 @ K6_local @ T6.T

    # ── SHELL OFFSET ──────────────────────────────────────────────

    @staticmethod
    def _offset_matrix(elem, T: np.ndarray, n_nodes: int, local_coords=None):
        """Rigid links from the element's NODES to its flat MID-SURFACE.

        The element is formed FLAT, on the projection of its corners onto their
        mean plane (normal n = T[:, 2]). Two things put a node off that surface:
          * the shell offset e: the mid-surface lies at e·n from the nodes;
          * warping: a warped quad's corners sit at ±h from the mean plane.
        Node i is linked to the point x_i + d_i of the flat mid-surface,
            d_i = (e − (z_i − z̄))·n ,  z_i = local z of node i, z̄ = corner mean.
        Rigid-body kinematics of each link:
            u_mid = u_node + θ × d_i = u_node − [d_i]× θ ,    θ_mid = θ_node
        so  q_mid = A q_node  with, per node,  A_i = [[I, −[d_i]×], [0, I]],
        and K_node = Aᵀ K_mid A.

        Why the warping links: without them the corner offsets ±h are dropped,
        and a rigid rotation of a warped quad strains the flat element (about
        200 N per mrad at 1 % warp for a 2.4 cm, 10 cm concrete quad). Internal
        forces then do not balance in moment (the reactions miss moment
        equilibrium by up to ~1 kN·m on scan meshes) and twisting is slightly
        over-stiff. With the links every rigid motion of the true (warped)
        geometry maps to a rigid motion of the flat element, so it is
        stress-free (MacNeal 1978; Taylor 1987). Flat elements without offset
        get A = I (None).
        """
        e = float(getattr(elem, 'offset', 0.0) or 0.0)
        dz = np.full(n_nodes, e)
        if local_coords is not None and n_nodes >= 3:
            z = np.asarray(local_coords, float)[:n_nodes, 2]
            n_corner = 3 if n_nodes in (3, 6) else 4
            dz = e - (z - z[:n_corner].mean())
        if float(np.max(np.abs(dz))) < 1e-12:
            return None
        nrm = T[:, 2]
        A = np.eye(6 * n_nodes)
        for i in range(n_nodes):
            d = dz[i] * nrm
            S = np.array([[0.0, -d[2], d[1]],
                          [d[2], 0.0, -d[0]],
                          [-d[1], d[0], 0.0]])      # [d]×  :  S @ θ = d × θ
            A[6*i:6*i+3, 6*i+3:6*i+6] = -S
        return A

    def _with_offset(self, elem, K_global: np.ndarray, T: np.ndarray,
                     local_coords=None) -> np.ndarray:
        """Element K (global frame, flat mid-surface DOFs) -> K on the node DOFs."""
        A = self._offset_matrix(elem, T, len(elem.node_ids), local_coords)
        return K_global if A is None else A.T @ K_global @ A

    # ── ASSEMBLY ────────────────────────────────────────────────────────

    def assemble_global_stiffness(self) -> None:
        print("[PERM Solver] Assembling global stiffness matrix...")
        t_start = time.time()
        # typed arrays: 16 bytes per stiffness entry instead of ~100 for Python lists of numbers (same entries, same order)
        from array import array as _typed_array
        rows_list, cols_list, vals_list = _typed_array('i'), _typed_array('i'), _typed_array('d')

        total = self.n_elements
        milestone = max(total // 10, 1)
        self._element_T_cache: Dict[int, np.ndarray] = {}

        skipped_count = 0
        skipped_by_type = defaultdict(int)
        n_truss = 0
        n_beam = 0
        n_shell = 0
        n_viz_only = 0
        n_membrane_only = 0        # forced: element type has no bending formulation
        n_membrane_by_choice = 0   # requested: user assigned a membrane-only type
        n_bending_by_choice = 0    # requested: user assigned a plate (MITC4-only) type
        for count, (eid, elem) in enumerate(self.converter.elements.items()):
            if count % milestone == 0:
                pct = 100 * count / total
                print(f"[PERM Solver]    Element {count}/{total} ({pct:.0f}%)")

            # Skip viz-only elements (e.g. shell faces replaced by trusses)
            if getattr(elem, 'viz_only', False):
                n_viz_only += 1
                continue
            
            # beam elements: 12 x 12 on all six DOFs of both nodes
            if getattr(elem.element_type, 'name', '') == 'BEAM_B2':
                try:
                    Ke_b = self._compute_beam_stiffness(elem)[0]
                except Exception as e:
                    print(f"[PERM Solver] Beam {eid} skipped: {e}")
                    skipped_by_type['BEAM_B2 (error)'] += 1
                    skipped_count += 1
                    continue
                dof_indices = []
                for nid in elem.node_ids:
                    base = self.node_id_to_index[nid] * self.DOFS_PER_NODE
                    dof_indices.extend(range(base, base + self.DOFS_PER_NODE))
                for i in range(12):
                    for j in range(12):
                        v = Ke_b[i, j]
                        if abs(v) > 1e-30:
                            rows_list.append(dof_indices[i])
                            cols_list.append(dof_indices[j])
                            vals_list.append(v)
                n_beam += 1
                continue

            # Truss elements: special handling
            if (getattr(elem.element_type, 'name', '') == 'TRUSS_T2'):
                try:
                    Ke_truss, _, _ = self._compute_truss_stiffness(elem)
                except Exception as e:
                    print(f"[PERM Solver] Truss {eid} skipped: {e}")
                    skipped_by_type['TRUSS_T2 (error)'] += 1
                    skipped_count += 1
                    continue
                # Map 6 truss DOFs (ux1,uy1,uz1, ux2,uy2,uz2) into the global
                # 6-DOF/node layout (DOFS_PER_NODE = 6); the truss contributes
                # nothing to the three rotational DOFs at either node.
                dof_indices = []
                for nid in elem.node_ids:
                    node_idx = self.node_id_to_index[nid]
                    base = node_idx * self.DOFS_PER_NODE
                    dof_indices.extend([base + 0, base + 1, base + 2])
                for i in range(6):
                    for j in range(6):
                        v = Ke_truss[i, j]
                        if abs(v) > 1e-30:
                            rows_list.append(dof_indices[i])
                            cols_list.append(dof_indices[j])
                            vals_list.append(v)
                n_truss += 1
                continue
            
            # Shell elements
            #
            # A face the user deliberately assigned a membrane-only element type
            # takes that path directly — no exception, and it is NOT counted in
            # the "no bending stiffness" warning below, which exists to flag
            # UNINTENDED hinges.
            # Same for a face deliberately assigned the PLATE type: MITC4 and
            # drilling, no membrane. Also a deliberate choice, also excluded
            # from the unintended-hinge warning, and reported on its own terms
            # below because what it removes is the opposite half of the shell.
            if getattr(elem, 'bending_only', False):
                try:
                    Ke, T_elem = self._compute_element_stiffness_bending_only(elem)
                    n_bending_by_choice += 1
                except Exception as e:
                    print(f"[PERM Solver] Element {eid} skipped: {e}")
                    skipped_count += 1
                    skipped_by_type[elem.element_type.name + ' (error)'] += 1
                    continue
                self._element_T_cache[eid] = T_elem
                n_shell += 1
                dof_indices = []
                for nid in elem.node_ids:
                    node_idx = self.node_id_to_index[nid]
                    for d in range(self.DOFS_PER_NODE):
                        dof_indices.append(node_idx * self.DOFS_PER_NODE + d)
                for i in range(len(dof_indices)):
                    for j in range(len(dof_indices)):
                        if abs(Ke[i, j]) > 1e-30:
                            rows_list.append(dof_indices[i])
                            cols_list.append(dof_indices[j])
                            vals_list.append(Ke[i, j])
                continue

            if getattr(elem, 'membrane_only', False):
                try:
                    Ke, T_elem = self._compute_element_stiffness_membrane_only(elem)
                    n_membrane_by_choice += 1
                except Exception as e:
                    print(f"[PERM Solver] Element {eid} skipped: {e}")
                    skipped_count += 1
                    skipped_by_type[elem.element_type.name + ' (error)'] += 1
                    continue
                self._element_T_cache[eid] = T_elem
                n_shell += 1
                dof_indices = []
                for nid in elem.node_ids:
                    node_idx = self.node_id_to_index[nid]
                    for d in range(self.DOFS_PER_NODE):
                        dof_indices.append(node_idx * self.DOFS_PER_NODE + d)
                for i in range(len(dof_indices)):
                    for j in range(len(dof_indices)):
                        if abs(Ke[i, j]) > 1e-30:
                            rows_list.append(dof_indices[i])
                            cols_list.append(dof_indices[j])
                            vals_list.append(Ke[i, j])
                continue

            try:
                Ke, T_elem = self._compute_element_stiffness(elem)
            except NotImplementedError:
                # No validated bending for this element type — degrade to
                # membrane-only. This is NOT silent: see the warning after the
                # loop. Such an element has no out-of-plane stiffness at all.
                try:
                    Ke, T_elem = self._compute_element_stiffness_membrane_only(elem)
                    skipped_by_type[elem.element_type.name + ' (membrane-only)'] += 1
                    n_membrane_only += 1
                except Exception as e2:
                    print(f"[PERM Solver] Element {eid} skipped: {e2}")
                    skipped_count += 1
                    skipped_by_type[elem.element_type.name + ' (skipped)'] += 1
                    continue
            except Exception as e:
                print(f"[PERM Solver] Element {eid} skipped: {e}")
                skipped_count += 1
                skipped_by_type[elem.element_type.name + ' (error)'] += 1
                continue
            self._element_T_cache[eid] = T_elem
            n_shell += 1

            dof_indices = []
            for nid in elem.node_ids:
                node_idx = self.node_id_to_index[nid]
                for d in range(self.DOFS_PER_NODE):
                    dof_indices.append(node_idx * self.DOFS_PER_NODE + d)

            n_elem_dofs = len(dof_indices)
            for i in range(n_elem_dofs):
                for j in range(n_elem_dofs):
                    if abs(Ke[i, j]) > 1e-30:
                        rows_list.append(dof_indices[i])
                        cols_list.append(dof_indices[j])
                        vals_list.append(Ke[i, j])

        if n_bending_by_choice > 0:
            print(f"[PERM Solver] {n_bending_by_choice} elements are PLATE (MITC4-only) "
                  f"BY ASSIGNMENT (Element Types -> Plate, MITC4 only).")
            print("[PERM Solver]   They carry out-of-plane load only. They have NO "
                  "in-plane stiffness")
            print("[PERM Solver]   (5 spurious zero-energy modes per element) — valid "
                  "for a flat plate under")
            print("[PERM Solver]   transverse load, NOT for folded structures, whose "
                  "bending is carried")
            print("[PERM Solver]   by membrane action of the flanges.")

        if n_membrane_by_choice > 0:
            print(f"[PERM Solver] {n_membrane_by_choice} elements are membrane-only "
                  f"BY ASSIGNMENT (Element Types -> Membrane shell).")
            print("[PERM Solver]   They carry in-plane load only and act as hinges "
                  "out of plane — as requested.")

        if n_membrane_only > 0:
            pct = 100.0 * n_membrane_only / max(total, 1)
            print("[PERM Solver] " + "!" * 62)
            print(f"[PERM Solver] ⚠ {n_membrane_only} elements ({pct:.1f}%) have "
                  f"NO BENDING STIFFNESS.")
            print("[PERM Solver]   They carry in-plane (membrane) load only — out of "
                  "plane they act as")
            print("[PERM Solver]   HINGES. QUAD4 (MITC4) and TRI3 (MITC3) carry bending; "
                  "TRI6/QUAD8 do not.")
            print("[PERM Solver]   Deflections will be TOO LARGE and moments missing "
                  "in those regions.")
            print("[PERM Solver]   Fix: re-mesh to QUAD4 (QuadRemesher / Instant "
                  "Meshes).")
            print("[PERM Solver] " + "!" * 62)

        # Stabilize truss-only nodes (perpendicular translations + rotational DOFs)
        if n_truss > 0:
            self._stabilize_truss_only_nodes(rows_list, cols_list, vals_list)

        rows = np.frombuffer(rows_list, dtype=np.int32)
        cols = np.frombuffer(cols_list, dtype=np.int32)
        vals = np.frombuffer(vals_list, dtype=np.float64)

        K_coo = sparse.coo_matrix((vals, (rows, cols)),
                                   shape=(self.n_dofs, self.n_dofs))
        self._K_global = K_coo.tocsr()
        del K_coo, rows, cols, vals, rows_list, cols_list, vals_list

        # Stabilize any zero-diagonal DOFs (nodes from skipped N-gons / degenerate
        # elements). Without this, a zero-diagonal row makes K singular → NaN.
        # The stabilizer magnitude is tiny (1e-12 × max diagonal) — it shifts
        # free, unconstrained DOFs to near-zero displacement without affecting
        # real structural response.
        diag_vals = self._K_global.diagonal()
        max_diag = float(np.max(np.abs(diag_vals)))
        # Detection and replacement use different values:
        #   bad_threshold — flags only genuinely DEAD DOFs (≈zero / negative
        #     diagonal): nodes whose every incident face was a skipped N-gon.
        #     Kept tiny so valid (small) rotational DOFs are NOT isolated —
        #     isolating those strips real coupling and re-introduces the NaN.
        #   stab_value — replacement diagonal, set LARGE (= max structural
        #     diagonal) so a dead DOF's residual gravity force gives F/K ≈ 0
        #     displacement instead of the runaway drift a tiny stab produced.
        # Valid rotational DOFs get real stiffness from MITC4 bending plus the
        # Hughes-Brezzi drilling constraint, so there is nothing singular to
        # patch — only genuinely DEAD DOFs (nodes whose every incident face was
        # skipped) are anchored here.
        bad_threshold = max_diag * 1e-12 if max_diag > 0 else 1.0
        stab_value    = max_diag         if max_diag > 0 else 1.0
        bad_mask = diag_vals < bad_threshold
        for _nid in (getattr(self.converter, '_joint_masters', None) or ()):     # joint-centre nodes get their
            if _nid in self.node_id_to_index:                                   # stiffness from the links, later
                _b = self.node_id_to_index[_nid] * self.DOFS_PER_NODE
                bad_mask[_b:_b + self.DOFS_PER_NODE] = False
        n_stabilized = int(np.sum(bad_mask))
        if n_stabilized > 0:
            bad_indices = np.where(bad_mask)[0]
            self._K_global = self._K_global.tolil()
            for idx in bad_indices:
                self._K_global[idx, idx] = stab_value   # SET, not +=
            self._K_global = self._K_global.tocsr()
            self._stab_value = stab_value   # stored for visualize_displacement
            n_neg  = int(np.sum(diag_vals[bad_mask] < 0))
            n_zero = n_stabilized - n_neg
            print(f'[PERM Solver] ⚠ Stabilized {n_stabilized} dead DOFs '
                  f'({n_neg} negative, {n_zero} zero/near-zero) — anchored, F/K≈0.')

        # ── Diagnostic: rotational vs translational diagonal health ─────────
        # Per-node DOF order: 0,1,2 = ux,uy,uz (translation); 3,4,5 = θx,θy,θz.
        # θz is held by the Hughes-Brezzi drilling constraint, which is
        # deliberately small (DRILL_GAMMA), so rotational diagonals are expected
        # to sit well below translational ones — that is not a fault. A count of
        # genuinely 'weak' DOFs near zero indicates disconnected geometry.
        _idx    = np.arange(diag_vals.size)
        _is_rot = (_idx % self.DOFS_PER_NODE) >= 3
        _rotd   = diag_vals[_is_rot]
        _trd    = diag_vals[~_is_rot]
        _weak   = max_diag * 1e-8
        print(f'[PERM Diag] translational diag: min={_trd.min():.3e}  '
              f'weak(<{_weak:.2e})={int(np.sum(_trd < _weak))}')
        print(f'[PERM Diag] rotational    diag: min={_rotd.min():.3e}  '
              f'weak(<{_weak:.2e})={int(np.sum(_rotd < _weak))}')

        t_elapsed = time.time() - t_start
        nnz = self._K_global.nnz
        print(f"[PERM Solver] Assembly complete: {t_elapsed:.1f}s, {nnz:,} non-zeros")
        print(f"[PERM Solver]   Shells: {n_shell}, Trusses: {n_truss}, Beams: {n_beam}, Viz-only: {n_viz_only}")
        if skipped_count > 0 or skipped_by_type:
            print("[PERM Solver] Element handling summary:")
            for kind, n in skipped_by_type.items():
                print(f"               {kind}: {n}")
            if skipped_count > 0:
                print(f"[PERM Solver] ⚠ {skipped_count} elements were SKIPPED entirely "
                      f"— their nodes have no stiffness contribution. "
                      f"This will cause singular matrix / NaN results.")

    def assemble_force_vector(self) -> None:
        self._F_global = np.zeros(self.n_dofs)
        self._beam_feq = {}            # consistent nodal loads of each beam (global, 12), for its end forces
        for load in self.load_cases:
            if load.load_type == LoadType.NODAL_FORCE:
                self._apply_nodal_force(load)
            elif load.load_type == LoadType.GRAVITY:
                self._apply_gravity(load)
            elif load.load_type == LoadType.PRESSURE:
                self._apply_pressure(load)
            elif load.load_type == LoadType.DISTRIBUTED:
                self._apply_surface_load(load)
        # Compute physically meaningful totals.
        # The L2 norm |F| is mathematically valid but doesn't directly tell you
        # whether your gravity got applied correctly — for that, the sum of force
        # components per axis is what matters. A negative net Fz with magnitude
        # close to ρgtA confirms gravity assembled correctly.
        total_Fx = float(np.sum(self._F_global[0::self.DOFS_PER_NODE]))
        total_Fy = float(np.sum(self._F_global[1::self.DOFS_PER_NODE]))
        total_Fz = float(np.sum(self._F_global[2::self.DOFS_PER_NODE]))
        L2_norm = float(np.linalg.norm(self._F_global))
        print("[PERM Solver] Force vector assembled:")
        print(f"               Total Fx = {total_Fx:+.4e} N")
        print(f"               Total Fy = {total_Fy:+.4e} N")
        print(f"               Total Fz = {total_Fz:+.4e} N  "
              f"({total_Fz/9.81:+.2f} kg equivalent)")
        print(f"               L2 norm  = {L2_norm:.4e} N  (not a total — diagnostic only)")

    def _apply_nodal_force(self, load: LoadCase) -> None:
        node_ids = self.converter.node_sets.get(load.node_set, set())
        if not node_ids: return
        # with patch loads on, a point load (1-3 nodes, total or per-node) is spread over a loading patch
        if PERM_PATCH_LOADS and len(node_ids) <= 3:
            pw = self._patch_weights([n for n in node_ids if n in self.node_id_to_index])
            if pw is not None:
                _pn = getattr(load, 'per_node_forces', None)
                if _pn:                                   # the group's total force = sum of its per-node values
                    _v2n = (getattr(self.converter, '_src_vert_to_node', None) or getattr(self.converter, '_vert_to_node', {}))
                    F = np.zeros(3)
                    for vidx, f in _pn.items():
                        nid = _v2n.get(int(vidx))
                        if nid is not None and nid in self.node_id_to_index:
                            F += np.asarray(f, float)[:3]
                else:
                    F = load.magnitude * np.asarray(load.direction, float)
                idxs, wts, _ = pw
                for k, w in zip(idxs, wts):
                    base = int(k) * self.DOFS_PER_NODE
                    for d in range(3):
                        self._F_global[base + d] += F[d] * w
                self.__dict__.setdefault('_patched_loads', {})[load.name] = len(idxs)
                return
            print(f"[PERM] load '{load.name}': no loadable faces within the {PERM_PATCH_SIZE[0]:.2f} x "
                  f"{PERM_PATCH_SIZE[1]:.2f} m patch - left on its node(s)")
        # Per-node mode: use the stored dict, fall back to zero for unlisted nodes
        per_node = getattr(load, 'per_node_forces', None)
        if per_node:
            # Map vertex-index keys to node-id keys.
            # The per_node dict from the operator uses Blender vertex indices
            # (which correspond to converter._vert_to_node[vidx] = node_id).
            vert_to_node = (getattr(self.converter, '_src_vert_to_node', None)
                            or getattr(self.converter, '_vert_to_node', {}))
            applied_count = 0
            for vidx, force in per_node.items():
                nid = vert_to_node.get(int(vidx))
                if nid is None or nid not in self.node_id_to_index:
                    continue
                idx = self.node_id_to_index[nid]
                base = idx * self.DOFS_PER_NODE
                fx, fy, fz = force
                self._F_global[base + 0] += fx
                self._F_global[base + 1] += fy
                self._F_global[base + 2] += fz
                applied_count += 1
            print(f"[PERM Solver]   Applied per-node forces to {applied_count} nodes")
            return
        # Uniform mode: distribute total force equally
        force_per_node = load.magnitude / len(node_ids)
        for nid in node_ids:
            idx = self.node_id_to_index[nid]
            base = idx * self.DOFS_PER_NODE
            for d in range(3):  # only translations
                self._F_global[base + d] += force_per_node * load.direction[d]

    def _apply_gravity(self, load: LoadCase) -> None:
        gx, gy, gz = load.gravity_vector
        for eid, elem in self.converter.elements.items():
            if getattr(elem, 'viz_only', False):
                continue
            material = self.converter.materials.get(elem.material_id,
                                                      self.converter.materials[0])
            # beam gravity, consistent loads: w L / 2 and ± w L² / 12 (end moments) at each end
            if getattr(elem.element_type, 'name', '') == 'BEAM_B2':
                sec = self.converter.sections.get(elem.section_id, None) if elem.section_id else None
                if sec is None or not sec.get('beam'):
                    continue
                coords = self._get_element_coords(elem)
                v = coords[1] - coords[0]
                L = float(np.linalg.norm(v))
                if L <= 0:
                    continue
                w = material.density * sec['beam']['A'] * np.array([gx, gy, gz])     # N per metre
                mi = (L * L / 12.0) * np.cross(v / L, w)
                fe = np.r_[0.5 * w * L, mi, 0.5 * w * L, -mi]
                self._beam_feq[eid] = self._beam_feq.get(eid, np.zeros(12)) + fe
                for k_, nid in enumerate(elem.node_ids):
                    base = self.node_id_to_index[nid] * self.DOFS_PER_NODE
                    self._F_global[base:base + 6] += fe[6 * k_:6 * k_ + 6]
                continue
            # Truss gravity: mass = ρ × A × L
            if (getattr(elem.element_type, 'name', '') == 'TRUSS_T2'):
                section = self.converter.sections.get(elem.section_id, None) if elem.section_id else None
                if section is None: continue
                area = float(section.get('area', 0.0))
                if area <= 0: continue
                coords = self._get_element_coords(elem)
                L = float(np.linalg.norm(coords[1] - coords[0]))
                mass = material.density * area * L
                fg = mass * np.array([gx, gy, gz])
                force_per_node = fg / 2.0
                for nid in elem.node_ids:
                    idx = self.node_id_to_index[nid]
                    base = idx * self.DOFS_PER_NODE
                    for d in range(3):
                        self._F_global[base + d] += force_per_node[d]
                continue
            # Shell gravity: mass = ρ × t × A
            coords = self._get_element_coords(elem)
            area = self._compute_element_area(coords, elem.element_type)
            mass = material.density * elem.thickness * area
            fg = mass * np.array([gx, gy, gz])
            # consistent nodal shares ∫N_i dA / A (equal only for
            # parallelograms and triangles); the total is unchanged
            wts = self._nodal_load_weights(elem, coords)
            for k, nid in enumerate(elem.node_ids):
                idx = self.node_id_to_index[nid]
                base = idx * self.DOFS_PER_NODE
                for d in range(3):
                    self._F_global[base + d] += fg[d] * wts[k]

    def _apply_pressure(self, load: LoadCase) -> None:
        if load.element_set == 'all':
            elem_ids = set(self.converter.elements.keys())
        else:
            elem_ids = self.converter.element_sets.get(load.element_set, set())
        for eid in elem_ids:
            elem = self.converter.elements[eid]
            coords = self._get_element_coords(elem)
            area = self._compute_element_area(coords, elem.element_type)
            normal = self._compute_element_normal(coords)
            total_force = load.magnitude * area * normal
            wts = self._nodal_load_weights(elem, coords)        # consistent shares
            for k, nid in enumerate(elem.node_ids):
                idx = self.node_id_to_index[nid]
                base = idx * self.DOFS_PER_NODE
                for d in range(3):
                    self._F_global[base + d] += total_force[d] * wts[k]

    def add_surface_load(self, element_loads: Dict[int, np.ndarray], name: str = "Surface load") -> None:
        """Force per unit area on shell elements. element_loads: element id -> traction (qx, qy, qz) in
        N/m2, GLOBAL axes, per unit of the element's own (true) area."""
        load = LoadCase(name=name, load_type=LoadType.DISTRIBUTED)
        load.element_loads = {int(k): np.asarray(v, float)[:3] for k, v in element_loads.items()}
        self.load_cases.append(load)

    def _apply_surface_load(self, load: LoadCase) -> None:
        for eid, q in (getattr(load, 'element_loads', None) or {}).items():
            elem = self.converter.elements.get(eid)
            if elem is None or getattr(elem, 'viz_only', False) or _perm_is_line(elem):
                continue
            coords = self._get_element_coords(elem)
            area = self._compute_element_area(coords, elem.element_type)
            wts = self._nodal_load_weights(elem, coords)          # consistent shares, as self-weight
            for k, nid in enumerate(elem.node_ids):
                base = self.node_id_to_index[nid] * self.DOFS_PER_NODE
                for d in range(3):
                    self._F_global[base + d] += q[d] * area * wts[k]

    def _nodal_load_weights(self, elem, coords: np.ndarray) -> np.ndarray:
        """Share of a uniformly distributed element load taken by each node.

        QUAD4: consistent shares ∫N_i dA / ∫dA on the flat (mean-plane) element,
        2×2 Gauss (exact for a bilinear map). They are 1/4 each only for a
        parallelogram; on a tapered quad the nodes at the wide end carry more.
        An equal split would put each element's load at the average of its
        corners instead of its centroid. Triangles (exact) and the
        quadratic types keep equal shares. Falls back to equal shares for a
        folded or degenerate quad (a Jacobian that is not positive).
        """
        n = len(elem.node_ids)
        if getattr(elem.element_type, 'name', '') != 'QUAD4' or n != 4:
            return np.full(n, 1.0 / n)
        try:
            _, T = self._compute_local_system(coords[:4])
            xy = (T.T @ (coords[:4] - coords[0]).T).T[:, :2]
            w = np.zeros(4); tot = 0.0
            for (xi, eta), gw in ShapeFunctions.gauss_points_quad(2):
                det_J = float(np.linalg.det(ShapeFunctions.quad4_dN(xi, eta) @ xy))
                if det_J <= 0.0:
                    return np.full(4, 0.25)
                w += ShapeFunctions.quad4_N(xi, eta) * det_J * gw
                tot += det_J * gw
            return w / tot if tot > 0.0 else np.full(4, 0.25)
        except Exception:
            return np.full(4, 0.25)

    def _compute_element_area(self, coords: np.ndarray, etype) -> float:
        etype_name = etype.name if hasattr(etype, 'name') else str(etype)
        if etype_name in ('TRI3', 'TRI6'):
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
        elif etype_name in ('QUAD4', 'QUAD8'):
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]
            a1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
            a2 = 0.5 * np.linalg.norm(np.cross(v2, v3))
            return a1 + a2
        return 0.0

    def _compute_element_normal(self, coords: np.ndarray) -> np.ndarray:
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        n = np.cross(v1, v2)
        if len(coords) >= 4:                       # mean plane, as the element frame
            nd = np.cross(coords[2] - coords[0], coords[3] - coords[1])
            if np.linalg.norm(nd) > 1e-15:
                n = nd
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        return n / norm


    # ── CONNECTION DETAILS ─────────────────────────────────────────

    def _span_axes(self):
        """Span axis (horizontal unit vector from the first to the last support group) and the across axis."""
        if getattr(self, '_span_ax', None) is not None:
            return self._span_ax, self._span_across
        cs = []
        for bc in self.boundary_conditions:
            ids = [self.node_id_to_index[n] for n in self.converter.node_sets.get(bc.node_set, set()) if n in self.node_id_to_index]
            if ids:
                cs.append(self.node_coords[ids].mean(0))
        ax = np.array([1.0, 0.0, 0.0])
        if len(cs) >= 2:
            best = (0.0, 0, 1)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    d = (cs[j] - cs[i]) * np.array([1.0, 1.0, 0.0])
                    if np.linalg.norm(d) > best[0]:
                        best = (float(np.linalg.norm(d)), i, j)
            d = (cs[best[2]] - cs[best[1]]) * np.array([1.0, 1.0, 0.0]); ax = d / np.linalg.norm(d)
        self._span_ax = ax; self._span_across = np.cross([0.0, 0.0, 1.0], ax)
        return self._span_ax, self._span_across

    def _element_table(self):
        """Per shell element: node indices, centre, unit normal, area, class (1 concrete, 2 steel, 3 massless), thickness,
        offset and whether its nodes lie on the TOP of its concrete (offset pointing down) or the BOTTOM (offset pointing up)."""
        if getattr(self, '_el_table', None) is not None:
            return self._el_table
        P = self.node_coords; rows = []
        for eid, e in self.converter.elements.items():
            if getattr(e, 'viz_only', False) or _perm_is_line(e):
                continue
            idx = [self.node_id_to_index[n] for n in e.node_ids if n in self.node_id_to_index]
            if len(idx) < 3:
                continue
            Q = P[idx]
            if len(idx) >= 4:
                nv = np.cross(Q[2] - Q[0], Q[3] - Q[1]); area = 0.5 * np.linalg.norm(nv)
            else:
                nv = np.cross(Q[1] - Q[0], Q[2] - Q[0]); area = 0.5 * np.linalg.norm(nv)
            if area <= 0:
                continue
            n = nv / np.linalg.norm(nv)
            m = self.converter.materials.get(e.material_id, self.converter.materials[0])
            E, rho = float(m.youngs_modulus), float(m.density)
            cls = 2 if E >= 1.0e11 else (3 if rho <= 0.0 else 1)
            off = float(getattr(e, 'offset', 0.0))
            rows.append((idx, Q.mean(0), n, area, cls, float(e.thickness), off))
        self._el_table = rows
        return rows

    def _patch_weights(self, node_ids):
        """Loading patch for a point load: nodes and force shares (sum 1), or None if no patch surface is found."""
        key = tuple(sorted(node_ids))
        cache = self.__dict__.setdefault('_patch_cache', {})
        if key in cache:
            return cache[key]
        P = self.node_coords; ax, ac = self._span_axes()
        ii = [self.node_id_to_index[n] for n in node_ids]
        c = P[ii].mean(0); hx, hy = 0.5 * PERM_PATCH_SIZE[0], 0.5 * PERM_PATCH_SIZE[1]
        share = defaultdict(float)
        if not hasattr(self, '_no_patch_faces'):           # the deck fill under a tube cannot be loaded
            self._no_patch_faces = {tuple(sorted(self.node_id_to_index[n] for n in self.converter.elements[e].node_ids[:3]))
                                 for e in (getattr(self.converter, '_plate_fill', None) or ())}
        for idx, cen, n, area, cls, t, off in self._element_table():
            if cls != 1 or abs(n[2]) < 0.7 or abs(cen[2] - c[2]) > 0.03:
                continue
            if self._no_patch_faces and tuple(sorted(idx)) in self._no_patch_faces:
                continue
            d = cen - c
            if abs(d @ ax) <= hx and abs(d @ ac) <= hy:
                for k in idx:
                    share[k] += area / len(idx)
        if not share:
            cache[key] = None
            return None
        tot = sum(share.values())
        out = (np.array(list(share.keys())), np.array(list(share.values())) / tot, tot)
        cache[key] = out
        print(f"[PERM] point load at ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) spread over a "
              f"{PERM_PATCH_SIZE[0]:.2f} x {PERM_PATCH_SIZE[1]:.2f} m patch: {len(out[0])} nodes, {tot:.4f} m2 of loaded faces")
        return out

    def _bearing_pad(self, node_index, other_centre):
        """Bearing pad of one support node: soffit nodes from the node PERM_PAD_LENGTH inward, within the web width.
        Returns (node indices, weights summing to 1) by tributary length along the span."""
        P = self.node_coords; ax, ac = self._span_axes(); p = P[node_index]
        inward = 1.0 if (other_centre - p) @ ax >= 0 else -1.0
        d = P - p
        s = inward * (d @ ax); t = d @ ac; z = d[:, 2]
        cand = np.nonzero((s >= -0.005) & (s <= PERM_PAD_LENGTH + 0.005) & (np.abs(t) <= 0.5 * PERM_PAD_WIDTH + 0.01)
                          & (np.abs(z) <= 0.008))[0]          # the soffit row only (the next row up is >= 1 cm higher)
        if node_index not in set(cand.tolist()):
            cand = np.r_[cand, node_index]
        ss = np.clip(s[cand], 0.0, PERM_PAD_LENGTH); o = np.argsort(ss, kind="stable"); cand, ss = cand[o], ss[o]
        mids = 0.5 * (ss[1:] + ss[:-1])
        lo = np.r_[0.0, mids]; hi = np.r_[mids, PERM_PAD_LENGTH]
        w = np.clip(hi - lo, 0.0, None)
        if w.sum() <= 0:
            w = np.ones(len(cand))
        return cand, w / w.sum()

    def _rigid_penalty(self, pairs, kt, kr):
        """Penalty stiffness of rigid links (master m, slave s): u_s = u_m + theta_m x r, theta_s = theta_m."""
        if not pairs:
            return None
        P = self.node_coords; D = self.DOFS_PER_NODE
        rows, cols, vals = [], [], []
        W = np.diag([kt, kt, kt, kr, kr, kr])
        for m, sl in pairs:
            r = P[sl] - P[m]
            Rx = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
            B = np.zeros((6, 12))
            B[0:3, 0:3] = -np.eye(3); B[0:3, 3:6] = Rx; B[0:3, 6:9] = np.eye(3)
            B[3:6, 3:6] = -np.eye(3); B[3:6, 9:12] = np.eye(3)
            Kp = B.T @ W @ B
            dofs = np.r_[m * D + np.arange(6), sl * D + np.arange(6)]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            nz = np.abs(Kp) > 0
            rows.append(rr[nz]); cols.append(cc[nz]); vals.append(Kp[nz])
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.n_dofs, self.n_dofs)).tocsr()

    def _internal_ties(self):
        """Rigid base plates under the railing posts and through-thickness ties of the overhang layers (added to K)."""
        self._detail_info = info = dict(plates=0, plate_nodes=0, ties=0, tie_lower_nodes=0)
        if not (PERM_POST_PLATES or PERM_TIE_OVERHANG or getattr(self.converter, '_plate_ties', None)):
            return
        P = self.node_coords; D = self.DOFS_PER_NODE; ax, ac = self._span_axes()
        dg = np.abs(self._K_global.diagonal()).reshape(-1, D)
        kt = PERM_RIGID_PENALTY * float(dg[:, :3].max()); kr = PERM_RIGID_PENALTY * float(dg[:, 3:].max())
        el = self._element_table()
        pairs = []
        for m_, s_ in (getattr(self.converter, '_plate_ties', None) or []):     # tube nodes within 4 mm of their plate node
            if m_ in self.node_id_to_index and s_ in self.node_id_to_index:
                pairs.append((self.node_id_to_index[m_], self.node_id_to_index[s_]))
        if PERM_POST_PLATES and str(PERM_PLATE_MODEL).lower() == "rigid":
            steel_n, conc_n, top_n = set(), set(), set()
            for idx, cen, n, area, cls, t, off in el:
                if cls == 2:
                    steel_n.update(idx)
                elif cls == 1:
                    conc_n.update(idx)
                    if abs(n[2]) >= 0.7 and off * n[2] <= 0.0:         # nodes on the TOP of this concrete
                        top_n.update(idx)
            joint = np.array(sorted(steel_n & conc_n), dtype=int)
            if len(joint):
                # posts = clusters of joint nodes (single linkage, 0.10 m)
                lab = -np.ones(len(joint), dtype=int); k = 0
                for i0 in range(len(joint)):
                    if lab[i0] >= 0:
                        continue
                    stack = [i0]; lab[i0] = k
                    while stack:
                        i = stack.pop()
                        near = np.nonzero((lab < 0) & (np.linalg.norm(P[joint] - P[joint[i]], axis=1) < 0.10))[0]
                        lab[near] = k; stack.extend(near.tolist())
                    k += 1
                topa = np.array(sorted(top_n), dtype=int); h = 0.5 * PERM_PLATE_SIZE
                self._rigid_plates = []
                conn_node_set = getattr(self.converter, '_conn_nodes', None) or set()
                for g in range(k):
                    ring = joint[lab == g]; c = P[ring].mean(0)
                    if conn_node_set and any(self.index_to_node_id[int(i_)] in conn_node_set for i_ in ring):
                        continue                       # an explicit connection is defined at this post
                    d = P[topa] - c
                    foot = topa[(np.abs(d @ ax) <= h) & (np.abs(d @ ac) <= h) & (np.abs(d[:, 2]) <= 0.04)]
                    foot = np.unique(np.r_[foot, ring])
                    if len(foot) < 2:
                        continue
                    m = int(foot[np.argmin(np.linalg.norm(P[foot] - c, axis=1))])
                    pairs += [(m, int(sl)) for sl in foot if sl != m]
                    self._rigid_plates.append(dict(centre=c, nodes=foot, master=m))
                    info['plates'] += 1; info['plate_nodes'] += len(foot)
        if PERM_TIE_OVERHANG:
            upper, lower = set(), set()
            for idx, cen, n, area, cls, t, off in el:
                if cls != 1 or abs(off) <= 0.0 or abs(n[2]) < 0.5:
                    continue
                (upper if off * n[2] < 0.0 else lower).update(idx)     # offset down: nodes on the top surface
            only_lower = np.array(sorted(lower - upper), dtype=int); up = np.array(sorted(upper), dtype=int)
            if len(only_lower) and len(up):
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(P[up][:, :2])
                    cand = tree.query_ball_point(P[only_lower][:, :2], r=PERM_TIE_MAX_PLAN)
                except Exception:
                    cand = [list(np.nonzero(np.linalg.norm(P[up][:, :2] - q[:2], axis=1) <= PERM_TIE_MAX_PLAN)[0]) for q in P[only_lower]]
                for sl, cc in zip(only_lower, cand):
                    if not cc:
                        continue
                    cc = up[np.array(cc, dtype=int)]
                    rise = P[cc, 2] - P[sl, 2]
                    ok = (rise > 0.02) & (rise <= PERM_TIE_MAX_RISE)
                    if not ok.any():
                        continue
                    cc = cc[ok]
                    m = int(cc[np.argmin(np.linalg.norm(P[cc, :2] - P[sl, :2], axis=1))])
                    pairs.append((m, int(sl))); info['ties'] += 1
                info['tie_lower_nodes'] = int(len(only_lower))
        Kp = self._rigid_penalty(pairs, kt, kr)
        if Kp is not None:
            self._K_global = (self._K_global + Kp).tocsr()
        print(f"[PERM] rigid base plates: {info['plates']} posts, {info['plate_nodes']} deck/post nodes "
              f"({PERM_PLATE_SIZE:.2f} x {PERM_PLATE_SIZE:.2f} m); overhang ties: {info['ties']} of "
              f"{info['tie_lower_nodes']} lower-layer nodes tied to the layer above")

    # ── BOUNDARY CONDITIONS ─────────────────────────────────────────────

    def _apply_boundary_conditions(self) -> None:
        self._constrained_dofs.clear()
        self._prescribed_values.clear()
        self._max_diag_structure = float(np.max(np.abs(self._K_global.diagonal())))   # before the rigid ties: penalties scale on the structure
        self._diag_structure = np.abs(self._K_global.diagonal()).reshape(-1, self.DOFS_PER_NODE)   # structure only
        self._internal_ties()            # rigid base plates + overhang ties are part of the structure (in K_free)
        self._connection_ties()          # joints between mesh parts, also part of the structure
        pad_blocks = []                      # (dof list, weights) of the bearing-pad constraints
        pad_dofs = set()
        self._bearing_pads = []
        _centres = {}
        for bc in self.boundary_conditions:
            ids = [self.node_id_to_index[n] for n in self.converter.node_sets.get(bc.node_set, set()) if n in self.node_id_to_index]
            if ids:
                _centres[bc.name] = self.node_coords[ids].mean(0)

        for bc in self.boundary_conditions:
            node_ids = self.converter.node_sets.get(bc.node_set, set())
            constrained = self._get_constrained_local_dofs(bc.bc_type,
                                                           bc.prescribed_values)
            if (PERM_BEARING_PADS and constrained and bc.bc_type in (BCType.PINNED, BCType.ROLLER_X, BCType.ROLLER_Y, BCType.ROLLER_Z)
                    and 0 < len(node_ids) <= 8 and len(_centres) >= 2
                    and not any(abs(v) > 0 for v in (bc.prescribed_values or {}).values())):
                other = [c for nm, c in _centres.items() if nm != bc.name]
                oc = max(other, key=lambda c: float(np.linalg.norm((c - _centres[bc.name])[:2])))
                new_set = set()
                for nid in node_ids:
                    if nid not in self.node_id_to_index:
                        continue
                    pad, w = self._bearing_pad(self.node_id_to_index[nid], oc)
                    for d in constrained:
                        pad_blocks.append((pad * self.DOFS_PER_NODE + d, w))
                        pad_dofs.update((pad * self.DOFS_PER_NODE + d).tolist())
                    new_set.update(self.index_to_node_id[int(k)] for k in pad)
                    self._bearing_pads.append(dict(group=bc.node_set, node=nid, nodes=pad, weights=w))
                # the support group now lists its pad nodes (reactions and the report follow them)
                self.converter.node_sets[bc.node_set] = new_set
                continue
            if not constrained:
                print(f"[PERM Solver] ⚠ BC '{bc.name}' ({bc.bc_type.name}) "
                      f"constrains NO degrees of freedom — it has no effect. "
                      f"For CUSTOM, name the DOFs to fix as prescribed_values "
                      f"keys, e.g. {{'uy': 0.0, 'rx': 0.0}}.")
                continue
            for nid in node_ids:
                idx = self.node_id_to_index[nid]
                base = idx * self.DOFS_PER_NODE
                for local_dof in constrained:
                    global_dof = base + local_dof
                    self._constrained_dofs.add(global_dof)
                    dof_name = self.DOF_NAMES[local_dof]
                    prescribed = bc.prescribed_values.get(dof_name, 0.0)
                    self._prescribed_values[global_dof] = prescribed

        # Keep the unconstrained K and the applied F: reactions are recovered
        # from them after the solve as R = K·u − F (exact, independent of the
        # penalty value -- unlike penalty·(u0 − u), which divides two tiny
        # numbers and loses accuracy).
        self._K_free = self._K_global.copy()
        self._F_applied = self._F_global.copy()

        diag = self._K_global.diagonal()
        max_diag = np.max(np.abs(diag)) if len(diag) > 0 else 1.0
        max_diag = getattr(self, '_max_diag_structure', None) or max_diag            # the structure's, not the rigid ties'
        penalty = max_diag * 1e7
        if pad_blocks:                       # bearing pads: k * w w^T on each restrained direction of each pad
            kp = PERM_PAD_PENALTY * max_diag
            rr, cc, vv = [], [], []
            for dofs, w in pad_blocks:
                a, b = np.meshgrid(dofs, dofs, indexing="ij")
                rr.append(a.ravel()); cc.append(b.ravel()); vv.append(kp * np.outer(w, w).ravel())
            self._K_global = (self._K_global + sparse.coo_matrix(
                (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=self._K_global.shape)).tocsr()
            print(f"[PERM] bearing pads: {len(self._bearing_pads)} pads, {PERM_PAD_LENGTH:.2f} m along the span, "
                  f"{sum(len(p['nodes']) for p in self._bearing_pads)} soffit nodes")

        # Track for diagnostics: how much physical force are we discarding/preserving?
        F_before = self._F_global.copy() if self._F_global is not None else None

        penalty_vec = np.zeros(self.n_dofs)
        for dof in self._constrained_dofs:
            penalty_vec[dof] = penalty
            # Penalty method: ADD penalty*prescribed to the existing F. A load
            # already assembled at a constrained DOF stays in F and becomes part
            # of the reaction; it must NOT be discarded here.
            self._F_global[dof] += penalty * self._prescribed_values.get(dof, 0.0)
        self._K_global += sparse.diags(penalty_vec)
        self._reaction_dofs = set(self._constrained_dofs) | pad_dofs

        # Diagnostic: how much applied force lives at constrained DOFs?
        # (Useful for understanding reaction-force behavior.)
        if F_before is not None:
            applied_at_constrained = sum(
                abs(F_before[d]) for d in self._constrained_dofs
            )
            total_applied = float(np.sum(np.abs(F_before)))
            if total_applied > 0:
                frac = applied_at_constrained / total_applied
                if frac > 0.01:  # only print if meaningful
                    print(f"[PERM Solver]   Note: {frac*100:.1f}% of applied force "
                          f"lives at constrained DOFs (becomes reaction force).")

        print(f"[PERM Solver] Applied BCs: {len(self._constrained_dofs)} "
              f"constrained DOFs ({len(self.boundary_conditions)} conditions)")

    DOF_NAMES = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']

    @classmethod
    def _get_constrained_local_dofs(cls, bc_type: BCType,
                                    prescribed: Optional[Dict[str, float]] = None
                                    ) -> List[int]:
        """Return list of local DOF indices to constrain.
        DOF ordering: 0=ux, 1=uy, 2=uz, 3=θx, 4=θy, 5=θz

        CUSTOM constrains exactly the DOFs named as keys in `prescribed`, e.g.
        {'uy': 0.0, 'rx': 0.0, 'rz': 0.0} for a symmetry plane whose normal is
        the global y axis. (A symmetry plane fixes the out-of-plane translation
        and the two rotations that lie IN the plane.)
        """
        if bc_type == BCType.FIXED:    return [0, 1, 2, 3, 4, 5]  # clamped (all 6)
        if bc_type == BCType.PINNED:   return [0, 1, 2]           # simply-supported
        if bc_type == BCType.ROLLER_X: return [1, 2]
        if bc_type == BCType.ROLLER_Y: return [0, 2]
        if bc_type == BCType.ROLLER_Z: return [0, 1]
        if bc_type == BCType.CUSTOM:
            if not prescribed:
                return []
            return sorted(cls.DOF_NAMES.index(k) for k in prescribed
                          if k in cls.DOF_NAMES)
        return []

    # ── SOLVE ───────────────────────────────────────────────────────────

    def solve(self) -> FEAResults:
        t_total_start = time.time()
        self._preflight_check()

        self.assemble_global_stiffness()
        self.assemble_force_vector()
        self._apply_boundary_conditions()

        # Enforce symmetry (cleans up accumulated numerical asymmetry)
        self._K_global = 0.5 * (self._K_global + self._K_global.T)

        print(f"[PERM Solver] Solving {self.n_dofs:,} DOF system...")
        t_solve_start = time.time()

        try:
            u, engine, resid = _perm_linear_solve(self._K_global, self._F_global)
        except Exception as e:
            print(f"[PERM Solver] Direct solve failed: {e}")
            print("[PERM Solver] Trying iterative solver (CG)...")
            u, info = spla.cg(self._K_global, self._F_global, rtol=1e-8, maxiter=10000)
            engine, resid = "SciPy CG (iterative)", float("nan")
            if info != 0:
                print(f"[PERM Solver] WARNING: CG did not converge (info={info})")
        self._solve_engine = engine

        t_solve = time.time() - t_solve_start
        print(f"[PERM Solver] Solve complete: {t_solve:.2f}s  [{engine}; residual {resid:.1e}]")

        self.results.displacements = u.reshape((self.n_nodes, self.DOFS_PER_NODE))
        self.results.solve_time = t_solve

        # Translation magnitudes
        u_trans = self.results.displacements[:, :3]
        disp_mag = np.linalg.norm(u_trans, axis=1)
        self.results.max_displacement = float(np.max(disp_mag))
        self.results.max_uz = float(np.max(np.abs(u_trans[:, 2])))

        self._compute_reactions(u)
        self._connection_forces()
        self._solve_id = f"{time.time():.6f}"

        print("[PERM Solver] Computing stresses and strains...")
        self._recover_stresses()
        self._print_results_summary(time.time() - t_total_start)
        return self.results

    def _preflight_check(self) -> None:
        errors = []
        if not self.boundary_conditions:
            errors.append("No boundary conditions defined. Model will be singular.")
        if not self.load_cases:
            errors.append("No loads defined. Displacements will be zero.")

        n_constrained = 0
        for bc in self.boundary_conditions:
            n_nodes = len(self.converter.node_sets.get(bc.node_set, set()))
            n_dofs_per = len(self._get_constrained_local_dofs(
                bc.bc_type, bc.prescribed_values))
            n_constrained += n_nodes * n_dofs_per

        if n_constrained < 6:
            errors.append(f"Only {n_constrained} DOFs constrained. "
                          f"Need at least 6 to prevent rigid body motion.")

        # Expected gravity force sanity check
        for load in self.load_cases:
            if load.load_type == LoadType.GRAVITY:
                total_mass = 0.0
                for elem in self.converter.elements.values():
                    mat = self.converter.materials.get(elem.material_id,
                                                         self.converter.materials[0])
                    coords = self._get_element_coords(elem)
                    if _perm_is_line(elem):                                   # bars: rho A L
                        sec = self.converter.sections.get(elem.section_id, None) if elem.section_id else None
                        if sec:
                            a_ = sec['beam']['A'] if sec.get('beam') else float(sec.get('area', 0.0))
                            total_mass += mat.density * a_ * float(np.linalg.norm(coords[1] - coords[0]))
                        continue
                    A = self._compute_element_area(coords, elem.element_type)
                    total_mass += mat.density * elem.thickness * A
                expected_F = abs(total_mass * load.gravity_vector[2])
                print(f"[PERM Solver] Total model mass: {total_mass:.2f} kg "
                      f"→ expected |Fz| ≈ {expected_F:.2f} N")
                if expected_F < 1.0:
                    errors.append(f"Total gravity force is only {expected_F:.4f} N. "
                                  f"Check mesh area / scale.")

        if errors:
            print("\n[PERM Solver] ⚠ PRE-FLIGHT WARNINGS:")
            for err in errors:
                print(f"  • {err}")
            print()

    # ── STRESS RECOVERY ─────────────────────────────────────────────────

    def _recover_stresses(self) -> None:
        self.results.stresses = {}
        self.results.stresses_top = {}
        self.results.stresses_bot = {}
        self.results.moments = {}
        self.results.stresses_global = {}
        self.results.stresses_top_global = {}
        self.results.stresses_bot_global = {}
        self.results.moments_global = {}
        self.results.strains = {}
        self.results.shear_forces = {}
        self.results.shear_forces_global = {}
        self.results.max_shear_stress = {}
        self.results.von_mises = {}
        self.results.von_mises_top = {}
        self.results.von_mises_bot = {}
        self.results.principal_stresses = {}
        self.results.principal_stresses_top = {}
        self.results.principal_stresses_bot = {}
        # Truss results
        self.results.truss_axial_forces = {}
        self.results.truss_axial_stresses = {}
        self.results.beam_forces = {}

        for eid, elem in self.converter.elements.items():
            if getattr(elem, 'viz_only', False):
                continue

            # beams: internal forces at both ends and mid-length, extreme normal stresses
            if getattr(elem.element_type, 'name', '') == 'BEAM_B2':
                try:
                    self.results.beam_forces[eid] = self._beam_results(eid, elem)
                except Exception as e:
                    print(f"[PERM Solver] Beam {eid} result recovery failed: {e}")
                continue
            
            # Truss stress recovery: axial force only
            if (getattr(elem.element_type, 'name', '') == 'TRUSS_T2'):
                try:
                    section = self.converter.sections.get(elem.section_id, None) if elem.section_id else None
                    if section is None: continue
                    area = float(section.get('area', 0.0))
                    if area <= 0: continue
                    material = self.converter.materials.get(elem.material_id,
                                                              self.converter.materials[0])
                    E = material.youngs_modulus
                    coords = self._get_element_coords(elem)
                    vec = coords[1] - coords[0]
                    L = float(np.linalg.norm(vec))
                    if L < 1e-15: continue
                    axis = vec / L
                    # Get translation DOFs at both end nodes
                    idx1 = self.node_id_to_index[elem.node_ids[0]]
                    idx2 = self.node_id_to_index[elem.node_ids[1]]
                    u1 = self.results.displacements[idx1, 0:3]
                    u2 = self.results.displacements[idx2, 0:3]
                    elongation = float(np.dot(u2 - u1, axis))
                    strain = elongation / L
                    stress = E * strain
                    force = stress * area
                    self.results.truss_axial_forces[eid] = force
                    self.results.truss_axial_stresses[eid] = stress
                except Exception as e:
                    print(f"[PERM Solver] Truss {eid} stress recovery failed: {e}")
                continue
            
            etype_name = elem.element_type.name
            if etype_name not in ('QUAD4', 'TRI3', 'TRI6', 'QUAD8'):
                continue
            # Membrane stress is recovered for every shell type. Bending,
            # transverse shear and the top/bottom fibre split need a bending
            # formulation: QUAD4 (MITC4) and TRI3 (MITC3) — not for
            # faces assigned membrane-only.
            has_bending = (etype_name in ('QUAD4', 'TRI3')
                           and not getattr(elem, 'membrane_only', False))

            coords = self._get_element_coords(elem)
            material = self.converter.materials.get(elem.material_id,
                                                      self.converter.materials[0])
            # Local frame from the CORNER nodes, then map every node into it
            # (TRI6/QUAD8 have mid-side nodes the frame subset does not cover).
            n_corner = 3 if etype_name.startswith('TRI') else 4
            _, T = self._compute_local_system(coords[:n_corner])
            local_coords = (T.T @ (coords - coords[0]).T).T

            # Get element displacements in global, transform to local
            u_global = self._get_element_displacements_global(elem)
            _A_off = self._offset_matrix(elem, T, len(elem.node_ids), local_coords)
            if _A_off is not None:
                u_global = _A_off @ u_global       # node DOFs -> flat mid-surface (offset + warping)
            u_local = self._transform_disp_global_to_local(u_global, T, len(elem.node_ids))

            # Membrane strain at centroid
            Dm = self._compute_Dm(material, elem.thickness)
            Db = self._compute_Db(material, elem.thickness)

            if etype_name == 'QUAD4':
                xi_c, eta_c = 0.0, 0.0
                dN = ShapeFunctions.quad4_dN(xi_c, eta_c)
            elif etype_name == 'TRI3':
                xi_c, eta_c = 1.0 / 3.0, 1.0 / 3.0
                dN = ShapeFunctions.tri3_dN(xi_c, eta_c)
            elif etype_name == 'TRI6':
                xi_c, eta_c = 1.0 / 3.0, 1.0 / 3.0
                dN = ShapeFunctions.tri6_dN(xi_c, eta_c)
            elif etype_name == 'QUAD8':
                xi_c, eta_c = 0.0, 0.0
                dN = ShapeFunctions.quad8_dN(xi_c, eta_c)
            else:
                continue
            # …but the membrane B-matrix spans ALL nodes of the element.
            n_geom = len(elem.node_ids)

            J = dN @ local_coords[:n_geom, :2]
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-20:
                continue
            J_inv = np.linalg.inv(J)
            dN_phys = J_inv @ dN

            # Membrane part
            Bm = self._build_Bm(dN_phys, n_geom)
            u_mem = np.zeros(2 * n_geom)
            for i in range(n_geom):
                u_mem[2*i]     = u_local[5*i + 0]
                u_mem[2*i + 1] = u_local[5*i + 1]
            eps_mem = Bm @ u_mem
            N_membrane = Dm @ eps_mem  # force per unit length
            if getattr(elem, 'bending_only', False):
                N_membrane = np.zeros(3)   # a plate element carries no membrane force
            sigma_mem = N_membrane / elem.thickness  # convert to stress
            self.results.strains[eid] = eps_mem
            self.results.stresses[eid] = sigma_mem
            self.results.stresses_global[eid] = self._voigt2d_local_to_global(sigma_mem, T)

            if not has_bending:
                # Membrane-only element: no curvature, no fibre split. Report
                # the membrane state on both fibres so downstream consumers
                # (von Mises, principal, heatmaps) still get sane values.
                zero3 = np.zeros(3)
                self.results.moments[eid] = zero3
                self.results.moments_global[eid] = zero3
                self.results.shear_forces[eid] = np.zeros(2)
                self.results.shear_forces_global[eid] = np.zeros(3)
                self.results.max_shear_stress[eid] = 0.0
                self.results.stresses_top[eid] = sigma_mem
                self.results.stresses_bot[eid] = sigma_mem
                g_mem = self.results.stresses_global[eid]
                self.results.stresses_top_global[eid] = g_mem
                self.results.stresses_bot_global[eid] = g_mem
                sx, sy, txy = sigma_mem[0], sigma_mem[1], sigma_mem[2]
                vm = math.sqrt(max(sx**2 + sy**2 - sx*sy + 3*txy**2, 0.0))
                self.results.von_mises_top[eid] = vm
                self.results.von_mises_bot[eid] = vm
                self.results.von_mises[eid] = vm
                s_avg = (sx + sy) / 2
                R = math.sqrt(max(((sx - sy) / 2)**2 + txy**2, 0.0))
                pr = np.array([s_avg + R, s_avg - R])
                self.results.principal_stresses_top[eid] = pr
                self.results.principal_stresses_bot[eid] = pr
                self.results.principal_stresses[eid] = np.array(
                    [s_avg + R, s_avg - R, 0.0])
                continue

            # Bending part — curvatures at the centroid (MITC4 / MITC3)
            n_b = 4 if etype_name == 'QUAD4' else 3
            u_bend = np.zeros(3 * n_b)
            for i in range(n_b):
                u_bend[3*i + 0] = u_local[5*i + 2]
                u_bend[3*i + 1] = u_local[5*i + 3]
                u_bend[3*i + 2] = u_local[5*i + 4]
            if etype_name == 'QUAD4':
                x = local_coords[:4, 0]
                y = local_coords[:4, 1]
                dN_nat = ShapeFunctions.quad4_dN(xi_c, eta_c)
                J4 = dN_nat @ np.column_stack([x, y])
                J4_inv = np.linalg.inv(J4)
                dN_xy = J4_inv @ dN_nat
                # Same rotation convention as _mitc4_Kb: βx = θy, βy = -θx.
                Bb = self._plate_Bb(dN_xy, 4)
                # Transverse shear (Reissner-Mindlin) from the SAME MITC tying
                # interpolation used to build the stiffness.
                Bs = self._mitc4_Bs_at(self._mitc4_tying_rows(x, y), J4_inv, xi_c, eta_c)
            else:
                _x3, _y3, J3 = self._mitc3_geometry(local_coords)
                J3_inv = np.linalg.inv(J3)
                Bb = self._plate_Bb(J3_inv @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]), 3)
                Bs = self._mitc3_Bs_at(self._mitc3_tying_rows(J3), J3_inv, 1.0 / 3.0, 1.0 / 3.0)

            kappa = Bb @ u_bend
            M = Db @ kappa  # moment per unit length [Mxx, Myy, Mxy]
            self.results.moments[eid] = M
            self.results.moments_global[eid] = self._voigt2d_local_to_global(M, T)
            gamma_s = Bs @ u_bend                       # [γ_xz, γ_yz]
            G_mat = material.youngs_modulus / (2.0 * (1.0 + material.poissons_ratio))
            Q = (5.0 / 6.0) * G_mat * elem.thickness * gamma_s   # [Qx, Qy] per width
            self.results.shear_forces[eid] = Q
            # Q is a VECTOR: rotate with T (first two local axes), not T·(·)·Tᵀ.
            self.results.shear_forces_global[eid] = T[:, :2] @ Q
            # Peak through-thickness shear stress for a parabolic profile.
            self.results.max_shear_stress[eid] = float(
                1.5 * np.linalg.norm(Q) / elem.thickness)

            # Top fiber stress (z = +t/2):  σ_top = σ_mem + 6M/t²  (sign per convention)
            # Bottom fiber stress (z = -t/2): σ_bot = σ_mem - 6M/t²
            sigma_bend_top = 6.0 * M / elem.thickness**2
            sigma_top = sigma_mem + sigma_bend_top
            sigma_bot = sigma_mem - sigma_bend_top
            self.results.stresses_top[eid] = sigma_top
            self.results.stresses_bot[eid] = sigma_bot
            self.results.stresses_top_global[eid] = self._voigt2d_local_to_global(sigma_top, T)
            self.results.stresses_bot_global[eid] = self._voigt2d_local_to_global(sigma_bot, T)

            # Von Mises at top and bottom fibers
            def _vm(s):
                sx, sy, txy = s[0], s[1], s[2]
                return math.sqrt(max(sx**2 + sy**2 - sx*sy + 3*txy**2, 0.0))
            vm_top = _vm(sigma_top)
            vm_bot = _vm(sigma_bot)
            self.results.von_mises_top[eid] = vm_top
            self.results.von_mises_bot[eid] = vm_bot
            self.results.von_mises[eid] = max(vm_top, vm_bot)

            # Principal stresses computed on EACH fiber separately.
            # σ1 = (σx+σy)/2 + R, σ2 = (σx+σy)/2 - R   (algebraic ordering, σ1 ≥ σ2)
            # For a slab bottom fiber under gravity, σ1 > 0 (tension) is expected.
            def _principal(s):
                sx, sy, txy = s[0], s[1], s[2]
                s_avg = (sx + sy) / 2
                R = math.sqrt(max(((sx - sy) / 2)**2 + txy**2, 0.0))
                return np.array([s_avg + R, s_avg - R])  # [σ1, σ2]

            p_top = _principal(sigma_top)
            p_bot = _principal(sigma_bot)
            self.results.principal_stresses_top[eid] = p_top
            self.results.principal_stresses_bot[eid] = p_bot

            # Principal stresses at the fibre with the larger von Mises stress
            s = sigma_top if vm_top >= vm_bot else sigma_bot
            sx, sy, txy = s[0], s[1], s[2]
            s_avg = (sx + sy) / 2
            R = math.sqrt(max(((sx - sy) / 2)**2 + txy**2, 0.0))
            self.results.principal_stresses[eid] = np.array([s_avg + R, s_avg - R, 0.0])

        if self.results.von_mises:
            self.results.max_von_mises = max(self.results.von_mises.values())

        # Track extreme principal stresses across all elements/fibers.
        # max_principal_tension = most tensile σ1 found anywhere
        # min_principal_compression = most compressive σ2 (most negative) found anywhere
        if self.results.principal_stresses_top:
            all_sigma1 = []
            all_sigma2 = []
            for eid in self.results.principal_stresses_top:
                all_sigma1.append(self.results.principal_stresses_top[eid][0])
                all_sigma1.append(self.results.principal_stresses_bot[eid][0])
                all_sigma2.append(self.results.principal_stresses_top[eid][1])
                all_sigma2.append(self.results.principal_stresses_bot[eid][1])
            self.results.max_principal_tension = float(max(all_sigma1))
            self.results.min_principal_compression = float(min(all_sigma2))

        if self.results.displacements is not None and self._F_global is not None:
            u_flat = self.results.displacements.flatten()
            self.results.total_strain_energy = 0.5 * float(u_flat @ self._F_global)

        print(f"[PERM Solver] Stress recovery complete: max VM = {self.results.max_von_mises:.4e} Pa")

    def _get_element_displacements_global(self, elem: FEElement) -> np.ndarray:
        """Return 6*n_nodes vector of element nodal displacements in global frame
        (ordering ux,uy,uz,θx,θy,θz per node)."""
        u = []
        for nid in elem.node_ids:
            idx = self.node_id_to_index[nid]
            for d in range(self.DOFS_PER_NODE):
                u.append(self.results.displacements[idx, d])
        return np.array(u)

    def _transform_disp_global_to_local(self, u_global: np.ndarray, T: np.ndarray,
                                         n_nodes: int) -> np.ndarray:
        """Take the solved global 6-DOF/node displacements [ux,uy,uz,θx,θy,θz]
        back to element-local 5-DOF/node [ux_l,uy_l,uz_l,θx_l,θy_l] for stress
        recovery.

            translations: u_local = Tᵀ · u_global
            rotations:    (θx_l,θy_l) = T[:, :2]ᵀ · (θx,θy,θz)_global
                          — project the node's global rotation vector onto the
                          element's local x,y axes (the drilling θz_l is unused).
        """
        u_local = np.zeros(5 * n_nodes)
        T_T = T.T
        T2t = T[:, :2].T                                   # 2×3
        for i in range(n_nodes):
            gb = 6 * i
            lb = 5 * i
            u_local[lb:lb+3] = T_T @ u_global[gb:gb+3]     # translations
            u_local[lb+3:lb+5] = T2t @ u_global[gb+3:gb+6] # local θx,θy
        return u_local

    def _print_results_summary(self, total_time: float) -> None:
        r = self.results
        print("\n" + "=" * 65)
        print(f"  PERM {PERM_VERSION} — FEA RESULTS SUMMARY")
        print("=" * 65)
        print(f"  Model:  {self.n_nodes:,} nodes, {self.n_elements:,} elements")
        print(f"  DOFs:   {self.n_dofs:,}  (6 DOF/node: ux,uy,uz,θx,θy,θz)")
        print(f"  BCs:    {len(self.boundary_conditions)} conditions, "
              f"{len(self._constrained_dofs):,} constrained DOFs")
        print(f"  Loads:  {len(self.load_cases)} load cases")
        print("-" * 65)
        print(f"  Max displacement (mag):   {r.max_displacement:.6e} m  "
              f"({r.max_displacement*1000:.3f} mm)")
        print(f"  Max |uz|:                 {r.max_uz:.6e} m  "
              f"({r.max_uz*1000:.3f} mm)")
        print(f"  Max Von Mises:            {r.max_von_mises:.6e} Pa  "
              f"({r.max_von_mises/1e6:.3f} MPa)")
        print(f"  Max σ1 tension (any fib): {r.max_principal_tension:+.4e} Pa  "
              f"({r.max_principal_tension/1e6:+.3f} MPa)")
        print(f"  Max |σ2| compression:     {r.min_principal_compression:+.4e} Pa  "
              f"({r.min_principal_compression/1e6:+.3f} MPa)")
        print(f"  Strain energy:            {r.total_strain_energy:.6e} J")
        if r.truss_axial_forces:                                     # line elements and joints
            tf = list(r.truss_axial_forces.values())
            print(f"  Trusses: {len(tf)} bars, N {min(tf)/1e3:+.3f} to {max(tf)/1e3:+.3f} kN")
        if r.beam_forces:
            bf = list(r.beam_forces.values())
            nmax = max(abs(b[s][0]) for b in bf for s in ('i', 'mid', 'j'))
            mmax = max(math.hypot(b[s][4], b[s][5]) for b in bf for s in ('i', 'mid', 'j'))
            print(f"  Beams: {len(bf)} elements, max |N| {nmax/1e3:.3f} kN, max |M| {mmax/1e3:.3f} kN·m, "
                  f"sigma {min(b['smin'] for b in bf)/1e6:+.2f} to {max(b['smax'] for b in bf)/1e6:+.2f} MPa")
        if r.connection_forces:
            print(f"  Connections: {len(r.connection_forces)} joint(s) (forces in the console lines above and the report)")
        print("-" * 65)
        print(f"  Solve time:               {r.solve_time:.2f} s")
        print(f"  Total time:               {total_time:.2f} s")
        print("=" * 65 + "\n")

    # ── VISUALIZATION ───────────────────────────────────────────────────

    def visualize_displacement(self, component: str = 'magnitude',
                                scale_factor: float = 1.0,
                                deformed: bool = True) -> bpy.types.Object:
        """Create a deformed-mesh visualization in Blender.
        
        Fixes from v1:
          1. Copies converter.working_object (the cleaned mesh whose vertex indices
             match _vert_to_node), not source_object.
          2. Converts world-space displacement back to the copy's LOCAL frame before
             adding to vert.co — accounts for non-identity object transforms.
        """
        if self.results.displacements is None:
            raise RuntimeError("No results. Run solve() first.")

        # Replace any previous deformation-viz object for this component so we
        # don't stack duplicates. The live scale slider then drives the new one.
        for old in [o for o in bpy.data.objects
                    if o.get("perm_disp_viz") and o.get("perm_disp_component") == component]:
            old_mesh = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if old_mesh and old_mesh.users == 0 and isinstance(old_mesh, bpy.types.Mesh):
                bpy.data.meshes.remove(old_mesh)

        # Use working_object so vertex indices align with _vert_to_node
        src = self.converter.working_object or self.converter.source_object
        obj_copy = src.copy()
        obj_copy.data = src.data.copy()
        obj_copy.name = self.FIELD_DISPLAY_NAMES.get(
            component, f"Displacement {component}")
        if "perm_working_copy" in obj_copy:
            del obj_copy["perm_working_copy"]
        obj_copy["perm_source_name"] = self.converter.source_name
        bpy.context.collection.objects.link(obj_copy)
        obj_copy.select_set(False)

        mesh = obj_copy.data
        n_verts = len(mesh.vertices)
        disp_values = np.zeros(n_verts)
        disp_local = np.zeros((n_verts, 3))   # per-vertex displacement, object-local frame

        # World→local rotation for the copy
        world_mat = obj_copy.matrix_world
        world_rot_3x3 = world_mat.to_3x3()  # mathutils.Matrix
        try:
            local_rot_3x3 = world_rot_3x3.inverted()
        except Exception:
            local_rot_3x3 = world_rot_3x3.transposed()  # fallback if non-invertible

        mapped_count = 0
        for vert in mesh.vertices:
            if vert.index not in self.converter._vert_to_node:
                continue
            nid = self.converter._vert_to_node[vert.index]
            idx = self.node_id_to_index[nid]
            u = self.results.displacements[idx]  # 6-vec: ux,uy,uz,θx,θy,θz in global

            u_world = Vector((u[0], u[1], u[2]))

            if component == 'magnitude':
                disp_values[vert.index] = u_world.length
            elif component == 'ux':
                disp_values[vert.index] = u[0]
            elif component == 'uy':
                disp_values[vert.index] = u[1]
            elif component == 'uz':
                disp_values[vert.index] = u[2]

            # Local-frame displacement (UNSCALED) — stored so the scale slider can
            # re-apply it live (co = rest + scale·disp) without recomputing anything.
            u_local = local_rot_3x3 @ u_world
            disp_local[vert.index] = (u_local.x, u_local.y, u_local.z)
            mapped_count += 1

        # ── Targeted outlier smoothing (v7) ────────────────────────────────
        # Filter 1 — magnitude >1.5×p99: rogue nodes and wrong-magnitude quads
        # Filter 2 — wrong-direction weak nodes (K-diag < stab×1000 AND lateral
        #             dominant displacement on a node with near-zero stiffness)
        # Correct-direction weak nodes are left untouched.
        _mapped = disp_values[disp_values > 0]
        if len(_mapped) > 10:
            _p99    = float(np.percentile(_mapped, 99.0))
            _thresh = _p99 * 1.5
            _maxval = float(disp_values.max())
            _bad_set = set(int(_i) for _i in np.where(disp_values > _thresh)[0])
            _stab_ref = getattr(self, '_stab_value', None)
            if _stab_ref is not None and hasattr(self, '_K_global'):
                _Kd = self._K_global.diagonal()
                _wk = _stab_ref * 1000.0
                for _vi in range(n_verts):
                    if _vi in _bad_set:
                        continue
                    if _vi not in self.converter._vert_to_node:
                        continue
                    _ni = self.node_id_to_index.get(self.converter._vert_to_node[_vi])
                    if _ni is None:
                        continue
                    _b = self.DOFS_PER_NODE * _ni
                    if min(_Kd[_b], _Kd[_b+1], _Kd[_b+2]) >= _wk:
                        continue
                    _ux2, _uy2, _uz2 = disp_local[_vi]
                    if abs(_ux2) > abs(_uz2) or abs(_uy2) > abs(_uz2):
                        _bad_set.add(_vi)
            if _bad_set and _maxval > _thresh:
                _nbr = {v.index: [] for v in mesh.vertices}
                for _e in mesh.edges:
                    _ea, _eb = _e.vertices[0], _e.vertices[1]
                    _nbr[_ea].append(_eb)
                    _nbr[_eb].append(_ea)
                _n_fixed = 0
                for _vi in sorted(_bad_set):
                    _good = [_n for _n in _nbr[_vi] if _n not in _bad_set]
                    if _good:
                        _avg = np.mean([disp_local[_n] for _n in _good], axis=0)
                        disp_local[_vi]  = _avg
                        disp_values[_vi] = float(np.linalg.norm(_avg))
                    else:
                        disp_local[_vi]  = np.zeros(3)
                        disp_values[_vi] = 0.0
                    _n_fixed += 1
                print(f'[PERM Solver] ⚠ {_n_fixed} bad nodes smoothed '
                      f'(p99={_p99*1000:.3f} mm, max was {_maxval*1000:.1f} mm).')
        # Capture the rest (undeformed) local coordinates before moving anything,
        # then persist rest + displacement on the mesh for live rescaling.
        rest_co = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", rest_co)
        _store_disp_attributes(mesh, rest_co.reshape(n_verts, 3), disp_local)
        obj_copy["perm_disp_viz"] = True
        obj_copy["perm_disp_component"] = component

        if deformed:
            _apply_disp_scale_to_object(obj_copy, scale_factor)

        print(f"[PERM Solver] Displacement viz '{obj_copy.name}': "
              f"{mapped_count}/{n_verts} vertices mapped, scale=×{scale_factor:g} "
              f"(drag the Deformation Scale slider to rescale live)")

        layer = f"DISP_{component}"
        self._apply_vertex_colors_jet(mesh, disp_values, layer)
        _perm_store_legend(obj_copy, getattr(self, '_last_scale', None),
                           {'magnitude': 'Displacement |u|', 'ux': 'Displacement ux',
                            'uy': 'Displacement uy', 'uz': 'Displacement uz'}.get(component, 'Displacement'),
                           'mm', 1e3)
        # Hover shows the TRUE solver displacement, not the display-smoothed one
        _true = np.zeros(n_verts)
        for _vi in range(n_verts):
            _nid = self.converter._vert_to_node.get(_vi)
            if _nid is not None:
                _d = self.results.displacements[self.node_id_to_index[_nid]]
                _true[_vi] = (float(np.linalg.norm(_d[:3])) if component == 'magnitude'
                              else float(_d[{'ux': 0, 'uy': 1, 'uz': 2}.get(component, 0)]))
        self._bake_hover_data(obj_copy, component, _true)
        obj_copy["perm_result_viz"] = True
        obj_copy["perm_result_layer"] = layer
        perm_build_result_material(obj_copy, layer)
        return obj_copy

    # Human-readable outliner names for the generated result meshes. These match
    # the panel checkbox labels one-to-one, so the object list reads the same as
    # the panel list — except the displacement mesh is just "Displacement" (the
    # checkbox says "Show Displacement", but the object drops the "Show").
    FIELD_DISPLAY_NAMES = {
        'magnitude':     'Displacement',
        'von_mises':     'Von Mises (max of top/bot)',
        'von_mises_top': 'Von Mises top fiber',
        'von_mises_bot': 'Von Mises bottom fiber',
        'sigma1_top':    'σ1 top fiber (max principal)',
        'sigma2_top':    'σ2 top fiber (min principal)',
        'sigma1_bot':    'σ1 bottom fiber (max principal)',
        'sigma2_bot':    'σ2 bottom fiber (min principal)',
        'sx':            'σxx midplane',
        'sy':            'σyy midplane',
        'txy':           'τxy midplane',
        'sx_top':        'σxx top fiber',
        'sy_top':        'σyy top fiber',
        'txy_top':       'τxy top fiber',
        'sx_bot':        'σxx bottom fiber',
        'sy_bot':        'σyy bottom fiber',
        'txy_bot':       'τxy bottom fiber',
        'Mx':            'Bending moment Mxx',
        'My':            'Bending moment Myy',
        'Mxy':           'Twisting moment Mxy',
    }

    # Components supported by visualize_stress, with metadata.
    # 'signed' = uses signed (blue-white-red) colormap; otherwise jet.
    STRESS_COMPONENTS = {
        # Unsigned magnitudes — jet colormap
        'von_mises':       {'signed': False, 'desc': 'Von Mises stress (worse of top/bot)'},
        'von_mises_top':   {'signed': False, 'desc': 'Von Mises at top fiber'},
        'von_mises_bot':   {'signed': False, 'desc': 'Von Mises at bottom fiber'},
        # Signed quantities — signed colormap
        # Cartesian stress components (membrane only — no fiber distinction).
        # Reported in the GLOBAL frame so they are comparable across elements.
        'sx':              {'signed': True,  'desc': 'σxx membrane (global frame)'},
        'sy':              {'signed': True,  'desc': 'σyy membrane (global frame)'},
        'txy':             {'signed': True,  'desc': 'τxy membrane (global frame)'},
        # Cartesian stress components at top fiber (global frame)
        'sx_top':          {'signed': True,  'desc': 'σxx top fiber (global frame)'},
        'sy_top':          {'signed': True,  'desc': 'σyy top fiber (global frame)'},
        'txy_top':         {'signed': True,  'desc': 'τxy top fiber (global frame)'},
        # Cartesian stress components at bottom fiber (global frame)
        'sx_bot':          {'signed': True,  'desc': 'σxx bottom fiber (global frame)'},
        'sy_bot':          {'signed': True,  'desc': 'σyy bottom fiber (global frame)'},
        'txy_bot':         {'signed': True,  'desc': 'τxy bottom fiber (global frame)'},
        # Principal stresses at top fiber
        'sigma1_top':      {'signed': True,  'desc': 'σ1 (max principal) at top fiber'},
        'sigma2_top':      {'signed': True,  'desc': 'σ2 (min principal) at top fiber'},
        # Principal stresses at bottom fiber
        'sigma1_bot':      {'signed': True,  'desc': 'σ1 (max principal) at bottom fiber'},
        'sigma2_bot':      {'signed': True,  'desc': 'σ2 (min principal) at bottom fiber'},
        # Bending moments (signed, global frame)
        'Mx':              {'signed': True,  'desc': 'Bending moment Mxx (N·m/m, global frame)'},
        'My':              {'signed': True,  'desc': 'Bending moment Myy (N·m/m, global frame)'},
        'Mxy':             {'signed': True,  'desc': 'Bending moment Mxy (N·m/m, global frame)'},
    }

    def visualize_stress(self, component: str = 'von_mises') -> bpy.types.Object:
        """Create a stress field visualization in Blender.
        
        See STRESS_COMPONENTS for the full list of supported components.
        Signed quantities use a blue-white-red diverging colormap (white at zero).
        Unsigned quantities use the jet colormap.
        """
        if not self.results.von_mises:
            raise RuntimeError("No results. Run solve() first.")
        if component not in self.STRESS_COMPONENTS:
            available = list(self.STRESS_COMPONENTS.keys())
            raise ValueError(f"Unknown stress component '{component}'. Available: {available}")

        comp_info = self.STRESS_COMPONENTS[component]
        is_signed = comp_info['signed']

        # Replace this field's mesh from the previous run instead of stacking
        # "<name>.001" on top of it.
        layer = f"STRESS_{component}"
        for old in [o for o in bpy.data.objects
                    if o.get("perm_result_layer") == layer
                    and o.get("perm_source_name", self.converter.source_name) == self.converter.source_name]:
            _perm_remove_object(old)
        src = self.converter.working_object or self.converter.source_object
        obj_copy = src.copy()
        obj_copy.data = src.data.copy()
        obj_copy.name = self.FIELD_DISPLAY_NAMES.get(component, component)
        if "perm_working_copy" in obj_copy:
            del obj_copy["perm_working_copy"]
        obj_copy["perm_source_name"] = self.converter.source_name
        bpy.context.collection.objects.link(obj_copy)
        obj_copy.select_set(False)

        mesh = obj_copy.data
        face_values = np.zeros(len(mesh.polygons))
        # Which faces actually have a recovered value. Stress is only recovered
        # for QUAD4; triangles / unsupported elements have none. Those faces must
        # NOT be painted as "zero stress" (misleading) nor pollute the colour
        # scale — they are greyed out and excluded from the averaging + stats.
        face_valid = np.zeros(len(mesh.polygons), dtype=bool)

        for poly in mesh.polygons:
            # The converter's own map: poly.index + 1 is wrong after any skipped face
            elem_id = self.converter._poly_to_elem.get(poly.index)
            if elem_id is None:
                continue
            val = self._get_stress_value(elem_id, component)
            if val is not None:
                face_values[poly.index] = val
                face_valid[poly.index] = True

        n_skipped = int(np.sum(~face_valid))
        if n_skipped > 0:
            print(f"[PERM Solver] '{component}': {n_skipped} of {len(mesh.polygons)} "
                  f"faces have no recovered stress (non-QUAD4) — shown grey, "
                  f"excluded from the colour scale.")

        # The face values stay on the mesh, so the colours (concrete / steel
        # filter, colour range) can be redone later without solving again.
        _perm_write_face_attr(mesh, "perm_face_value", 'FLOAT',
                              np.where(face_valid, face_values, np.nan).astype(np.float32))
        layer = f"STRESS_{component}"
        _is_moment = component in ('Mx', 'My', 'Mxy')
        obj_copy["perm_result_viz"] = True
        obj_copy["perm_result_layer"] = layer
        obj_copy["perm_component"] = component
        obj_copy["perm_signed"] = bool(is_signed)
        obj_copy["perm_title"] = self.FIELD_DISPLAY_NAMES.get(component, comp_info['desc'])
        obj_copy["perm_unit"] = 'kN·m/m' if _is_moment else 'MPa'
        obj_copy["perm_factor"] = 1e-3 if _is_moment else 1e-6
        vert_values, valid_vert_mask = _perm_recolor_result(obj_copy, bpy.context.scene)
        self._set_vertex_color_shading()
        self._bake_hover_data(obj_copy, component, vert_values, valid_vert_mask)
        perm_build_result_material(obj_copy, layer)
        print(f"[PERM Solver] Stress visualization: '{obj_copy.name}'  "
              f"({comp_info['desc']})")
        return obj_copy

    def _get_stress_value(self, elem_id: int, component: str):
        """Look up a single stress-component value for an element. Returns None
        if missing (element not in dict, or principal stresses not computed)."""
        r = self.results
        if component == 'von_mises' and r.von_mises and elem_id in r.von_mises:
            return r.von_mises[elem_id]
        if component == 'von_mises_top' and r.von_mises_top and elem_id in r.von_mises_top:
            return r.von_mises_top[elem_id]
        if component == 'von_mises_bot' and r.von_mises_bot and elem_id in r.von_mises_bot:
            return r.von_mises_bot[elem_id]
        # Membrane cartesian — GLOBAL frame (averagable across elements)
        if component in ('sx', 'sy', 'txy') and r.stresses_global and elem_id in r.stresses_global:
            return r.stresses_global[elem_id][{'sx': 0, 'sy': 1, 'txy': 2}[component]]
        # Top fiber cartesian — GLOBAL frame
        if component in ('sx_top', 'sy_top', 'txy_top') and r.stresses_top_global and elem_id in r.stresses_top_global:
            return r.stresses_top_global[elem_id][{'sx_top': 0, 'sy_top': 1, 'txy_top': 2}[component]]
        # Bottom fiber cartesian — GLOBAL frame
        if component in ('sx_bot', 'sy_bot', 'txy_bot') and r.stresses_bot_global and elem_id in r.stresses_bot_global:
            return r.stresses_bot_global[elem_id][{'sx_bot': 0, 'sy_bot': 1, 'txy_bot': 2}[component]]
        # Top fiber principal
        if component in ('sigma1_top', 'sigma2_top') and r.principal_stresses_top \
                and elem_id in r.principal_stresses_top:
            return r.principal_stresses_top[elem_id][{'sigma1_top': 0, 'sigma2_top': 1}[component]]
        # Bottom fiber principal
        if component in ('sigma1_bot', 'sigma2_bot') and r.principal_stresses_bot \
                and elem_id in r.principal_stresses_bot:
            return r.principal_stresses_bot[elem_id][{'sigma1_bot': 0, 'sigma2_bot': 1}[component]]
        # Bending moments — GLOBAL frame (averagable across elements)
        if component in ('Mx', 'My', 'Mxy') and r.moments_global and elem_id in r.moments_global:
            return r.moments_global[elem_id][{'Mx': 0, 'My': 1, 'Mxy': 2}[component]]
        return None

    # ── SUPPORT REACTIONS ───────────────────────────────────────────────

    def _compute_reactions(self, u: np.ndarray) -> None:
        """R = K·u − F at every constrained DOF (GLOBAL frame).

        K and F are the matrices BEFORE the penalty was added, symmetrised the
        same way as the solved system. Loads applied directly at a support are
        part of F, so they are correctly netted out of the reaction there.
        """
        K = getattr(self, '_K_free', None)
        F = getattr(self, '_F_applied', None)
        n = self.n_nodes
        R = np.zeros((n, self.DOFS_PER_NODE))
        nodes: Dict[int, List[int]] = {}
        rdofs = getattr(self, '_reaction_dofs', None) or self._constrained_dofs
        if K is None or F is None or not rdofs:
            self.results.reaction_forces = R
            self.results.reaction_nodes = nodes
            return
        Ks = 0.5 * (K + K.T)
        r = Ks @ u - F
        flat = R.reshape(-1)
        for dof in rdofs:
            flat[dof] = r[dof]
            idx, local = divmod(dof, self.DOFS_PER_NODE)
            nodes.setdefault(self.index_to_node_id[idx], []).append(local)
        for nid in nodes:
            nodes[nid].sort()
        self.results.reaction_forces = R
        self.results.reaction_nodes = nodes
        tot_R = R[:, :3].sum(axis=0)
        tot_F = F.reshape(n, self.DOFS_PER_NODE)[:, :3].sum(axis=0)
        self.results.reaction_total = tot_R
        self.results.applied_total = tot_F
        err = np.linalg.norm(tot_R + tot_F) / max(np.linalg.norm(tot_F), 1e-30)
        print(f"[PERM Solver] Support reactions at {len(nodes)} node(s): "
              f"ΣR = ({tot_R[0]/1e3:.3f}, {tot_R[1]/1e3:.3f}, {tot_R[2]/1e3:.3f}) kN; "
              f"applied ΣF = ({tot_F[0]/1e3:.3f}, {tot_F[1]/1e3:.3f}, {tot_F[2]/1e3:.3f}) kN; "
              f"equilibrium error {err:.2e}")

    def visualize_reactions(self, size_scale: float = 1.0) -> Optional[bpy.types.Object]:
        """Classic FEA reaction plot at EVERY supported node.

        One thin line arrow per constrained force component (X red, Y green,
        Z blue), head at the node, pointing the way the support pushes on the
        structure, length proportional to the value (longest = 6 % of the model
        size x 'Arrow size'). The arrows are viewport overlay lines held by an
        Empty called 'Support reactions' -- hide it to hide the plot. Values:
        one label per node, hover (Inspect), and the text block
        'PERM_Reactions.csv' with all six components of every node.
        """
        res = self.results
        if res.reaction_forces is None or not res.reaction_nodes:
            print("[PERM Solver] No support reactions to show.")
            return None
        for old in [o for o in bpy.data.objects if o.get("perm_reaction_viz")]:
            _perm_remove_object(old)

        R = res.reaction_forces
        span = float(np.linalg.norm(self.node_coords.max(axis=0) - self.node_coords.min(axis=0))) or 1.0
        L_ref = 0.06 * span * size_scale
        fmax = max((abs(R[self.node_id_to_index[n], d])
                    for n, dofs in res.reaction_nodes.items() for d in dofs if d < 3), default=0.0)
        axes = np.eye(3)
        colors = [(0.95, 0.30, 0.30), (0.30, 0.85, 0.35), (0.35, 0.60, 1.00)]
        names = ("Rx", "Ry", "Rz")
        lines, labels, node_data = [], [], []
        head = 0.10 * L_ref
        for nid, dofs in sorted(res.reaction_nodes.items()):
            idx = self.node_id_to_index[nid]
            p = self.node_coords[idx]
            r6 = R[idx]
            node_data.append([int(nid), *map(float, p), *map(float, r6), list(map(int, dofs))])
            if fmax <= 0:
                continue
            best_tail, best = None, 0.0
            shown = []
            for d in (0, 1, 2):
                if d not in dofs:
                    continue
                length = L_ref * abs(r6[d]) / fmax
                if length < 0.01 * L_ref:          # below 1 % of the largest: not drawn
                    continue
                shown.append(d)
                dv = axes[d] * np.sign(r6[d])
                tail = p - dv * length
                col = colors[d]
                lines.append([*map(float, tail), *map(float, p), *col])
                h = min(head, 0.5 * length)
                e1 = axes[(d + 1) % 3]; e2 = axes[(d + 2) % 3]
                for e in (e1, -e1, e2, -e2):
                    q = p - dv * h + e * 0.35 * h
                    lines.append([*map(float, p), *map(float, q), *col])
                if abs(r6[d]) > best:
                    best, best_tail = abs(r6[d]), tail
            if shown:
                txt = "  ".join(f"{names[d]} {r6[d]/1e3:.3g}" for d in shown) + " kN"
                labels.append([*map(float, best_tail), txt, 0.93, 0.93, 0.93, float(best)])

        obj = bpy.data.objects.new("Support reactions", None)
        obj.empty_display_size = 0.01 * span
        bpy.context.collection.objects.link(obj)
        obj["perm_reaction_viz"] = True
        obj["perm_source_name"] = self.converter.source_name
        obj["perm_solve_id"] = getattr(self, '_solve_id', "")
        obj["perm_build_id"] = f"{time.time():.6f}"
        obj["perm_reaction_lines"] = json.dumps(lines)
        obj["perm_reaction_labels"] = json.dumps(labels)
        obj["perm_reaction_nodes"] = json.dumps(node_data)

        # Every supported node, all six components -> text block (Text Editor)
        tname = "PERM_Reactions.csv"
        txt = bpy.data.texts.get(tname) or bpy.data.texts.new(tname)
        rows = ["node,x_m,y_m,z_m,Rx_kN,Ry_kN,Rz_kN,Mx_kNm,My_kNm,Mz_kNm,restrained"]
        for n in node_data:
            rows.append(f"{n[0]},{n[1]:.4f},{n[2]:.4f},{n[3]:.4f}," +
                        ",".join(f"{v/1e3:.6g}" for v in n[4:10]) + "," +
                        "".join("xyzXYZ"[d] for d in n[10]))
        tot = res.reaction_total if res.reaction_total is not None else np.zeros(3)
        rows.append(f"TOTAL,,,,{tot[0]/1e3:.6g},{tot[1]/1e3:.6g},{tot[2]/1e3:.6g},,,,")
        txt.clear(); txt.write("\n".join(rows) + "\n")
        print(f"[PERM Solver] Reaction plot: {len(lines)//5} arrows at {len(res.reaction_nodes)} "
              f"supported node(s); ΣR = ({tot[0]/1e3:.3f}, {tot[1]/1e3:.3f}, {tot[2]/1e3:.3f}) kN. "
              f"All values: Text Editor -> '{tname}'")
        return obj

    # ── HOVER DATA BAKED ON RESULT MESHES ───────────────────────────────

    FIELD_UNITS = {   # component -> (unit label, multiplier from SI)
        'magnitude': ("mm", 1e3),
        'Mx': ("kN·m/m", 1e-3), 'My': ("kN·m/m", 1e-3), 'Mxy': ("kN·m/m", 1e-3),
    }

    def _field_unit(self, component: str) -> Tuple[str, float]:
        return self.FIELD_UNITS.get(component, ("MPa", 1e-6))

    def _bake_hover_data(self, obj, component: str, vert_values: np.ndarray,
                         valid_vert_mask: Optional[np.ndarray] = None) -> None:
        """Store per-vertex node id, displacement, rotation, reaction and the
        displayed field value on a result mesh, so the hover inspector can read
        them later -- even after the file is saved and reopened."""
        mesh = obj.data
        nv = len(mesh.vertices)
        nid_arr = np.full(nv, -1, dtype=np.int32)
        u = np.zeros((nv, 3)); rot = np.zeros((nv, 3))
        rf = np.zeros((nv, 3)); rm = np.zeros((nv, 3)); sup = np.zeros(nv, dtype=bool)
        R = self.results.reaction_forces
        rnodes = self.results.reaction_nodes or {}
        for vi in range(nv):
            nid = self.converter._vert_to_node.get(vi)
            if nid is None:
                continue
            idx = self.node_id_to_index.get(nid)
            if idx is None:
                continue
            nid_arr[vi] = nid
            d = self.results.displacements[idx]
            u[vi] = d[:3]; rot[vi] = d[3:6]
            if nid in rnodes and R is not None:
                sup[vi] = True
                rf[vi] = R[idx, :3]; rm[vi] = R[idx, 3:6]
        valid = (np.ones(nv, dtype=bool) if valid_vert_mask is None
                 else np.asarray(valid_vert_mask, dtype=bool))

        def put(name, typ, data, field):
            if name in mesh.attributes:
                mesh.attributes.remove(mesh.attributes[name])
            a = mesh.attributes.new(name, typ, 'POINT')
            a.data.foreach_set(field, np.ascontiguousarray(data).ravel())
        put("perm_node_id", 'INT', nid_arr, "value")
        put("perm_u", 'FLOAT_VECTOR', u.astype(np.float32), "vector")
        put("perm_rot", 'FLOAT_VECTOR', rot.astype(np.float32), "vector")
        put("perm_reac_f", 'FLOAT_VECTOR', rf.astype(np.float32), "vector")
        put("perm_reac_m", 'FLOAT_VECTOR', rm.astype(np.float32), "vector")
        put("perm_is_support", 'BOOLEAN', sup, "value")
        put("perm_field", 'FLOAT', np.asarray(vert_values, dtype=np.float32), "value")
        put("perm_field_valid", 'BOOLEAN', valid, "value")
        unit, mult = self._field_unit(component)
        obj["perm_field_component"] = component
        obj["perm_field_label"] = self.FIELD_DISPLAY_NAMES.get(component, component)
        obj["perm_field_unit"] = unit
        obj["perm_field_mult"] = mult
        obj["perm_solve_id"] = getattr(self, '_solve_id', "")

    def nodal_all_fields(self, nid: int) -> List[Tuple[str, float, str]]:
        """Every stress/moment field at a node: the average over the adjacent
        elements that have a value (the same averaging the heatmaps use)."""
        if not hasattr(self, '_node_elems'):
            ne: Dict[int, List[int]] = defaultdict(list)
            for eid, e in self.converter.elements.items():
                if _perm_is_line(e) or getattr(e, 'viz_only', False):
                    continue
                for n in e.node_ids:
                    ne[n].append(eid)
            self._node_elems = ne
        out = []
        for comp in self.STRESS_COMPONENTS:
            vals = [v for v in (self._get_stress_value(eid, comp)
                                for eid in self._node_elems.get(nid, [])) if v is not None]
            if vals:
                unit, mult = self._field_unit(comp)
                out.append((self.FIELD_DISPLAY_NAMES.get(comp, comp), float(np.mean(vals)) * mult, unit))
        return out

    # Colour used for faces/verts that have no recovered value (greyed out).
    NO_DATA_COLOR = (0.5, 0.5, 0.5, 1.0)

    def _apply_vertex_colors_jet(self, mesh: bpy.types.Mesh, values: np.ndarray,
                                  layer_name: str,
                                  clip_percentile: float = 2.0,
                                  valid_vert_mask: Optional[np.ndarray] = None,
                                  valid_face_mask: Optional[np.ndarray] = None) -> None:
        """Apply jet colormap to mesh with percentile-based color scale clipping.

        clip_percentile: % of extreme values to saturate (e.g. 2.0 means the
            color range spans the 2nd to 98th percentile of values; outliers
            beyond that saturate at the endpoint colors). Standard FEA convention
            to prevent a few bad elements from washing out the entire range.
            Set to 0.0 to use raw min/max.
        valid_vert_mask: optional bool array over vertices; only these verts feed
            the colour scale (min/max/percentile). Defaults to all verts.
        valid_face_mask: optional bool array over polygons; faces that are False
            are painted NO_DATA_COLOR (grey) instead of a colormap value.
        """
        if hasattr(mesh, 'color_attributes'):
            if layer_name not in mesh.color_attributes:
                mesh.color_attributes.new(name=layer_name, type='BYTE_COLOR', domain='CORNER')
            color_layer = mesh.color_attributes[layer_name]
            mesh.color_attributes.active_color = color_layer
        else:
            if layer_name not in mesh.vertex_colors:
                mesh.vertex_colors.new(name=layer_name)
            color_layer = mesh.vertex_colors[layer_name]

        # Build the colour scale from valid verts only, so unrecovered faces
        # (greyed out below) cannot skew the range.
        if valid_vert_mask is not None and np.any(valid_vert_mask):
            scale_vals = values[valid_vert_mask]
        else:
            scale_vals = values

        v_raw_min = float(np.min(scale_vals))
        v_raw_max = float(np.max(scale_vals))

        if clip_percentile > 0.0:
            v_min = float(np.percentile(scale_vals, clip_percentile))
            v_max = float(np.percentile(scale_vals, 100.0 - clip_percentile))
        else:
            v_min, v_max = v_raw_min, v_raw_max

        v_range = v_max - v_min if v_max > v_min else 1.0
        self._last_scale = ('jet', v_min, v_max, v_raw_min, v_raw_max, clip_percentile)

        # Report clip range so user can see what was suppressed
        n_clipped_low = int(np.sum(scale_vals < v_min))
        n_clipped_high = int(np.sum(scale_vals > v_max))
        if n_clipped_low + n_clipped_high > 0:
            print(f"[PERM Solver] Color scale '{layer_name}':")
            print(f"               raw range:    {v_raw_min:.4e} to {v_raw_max:.4e}")
            print(f"               clipped to:   {v_min:.4e} to {v_max:.4e} "
                  f"({clip_percentile}-{100-clip_percentile} percentile)")
            print(f"               saturated:    {n_clipped_low} verts at low end, "
                  f"{n_clipped_high} verts at high end")

        def jet(t: float) -> Tuple[float, float, float, float]:
            t = max(0.0, min(1.0, t))
            if t < 0.25:   s = t / 0.25;       return (0.0, s, 1.0, 1.0)
            elif t < 0.5:  s = (t - 0.25)/0.25; return (0.0, 1.0, 1.0 - s, 1.0)
            elif t < 0.75: s = (t - 0.5)/0.25;  return (s, 1.0, 0.0, 1.0)
            else:          s = (t - 0.75)/0.25; return (1.0, 1.0 - s, 0.0, 1.0)

        for poly in mesh.polygons:
            face_ok = True if valid_face_mask is None else bool(valid_face_mask[poly.index])
            for loop_idx in poly.loop_indices:
                if not face_ok:
                    color_layer.data[loop_idx].color = self.NO_DATA_COLOR
                    continue
                vi = mesh.loops[loop_idx].vertex_index
                t = (values[vi] - v_min) / v_range
                color_layer.data[loop_idx].color = jet(t)

        self._set_vertex_color_shading()

    @staticmethod
    def _set_vertex_color_shading():
        """Put the viewport where both the texture and the overlay are visible.

        SOLID shading with color_type='VERTEX' draws the colour attribute and
        nothing else: the object's texture is not rendered in that mode, so the
        Result Opacity slider would have nothing to reveal. MATERIAL preview
        renders the mix material built by perm_build_result_material, where
        opacity 1.0 gives an unlit colour readout and lower values let the
        texture through.

        color_type is still set, so dropping back to Solid by hand keeps
        showing the colour map rather than flat grey.
        """
        try:
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.color_type = 'VERTEX'
                            space.shading.type = 'MATERIAL'
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# BLENDER OPERATORS — Manual Support & Load Assignment
# (unchanged from v1, included for completeness)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-GROUP LOAD / SUPPORT INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
#
# Naming convention:
#   - Vertex groups starting with "PERM_Load_<name>" are load groups
#   - Vertex groups starting with "PERM_Support_<name>" are support groups
#   - Legacy "PERM_Loads" and "PERM_Supports" still work (single-group mode)
#
# Per-group force / BC values are stored as custom properties on the object:
#   obj["perm_load_<name>_fx"], obj["perm_load_<name>_fy"], obj["perm_load_<name>_fz"]
#   obj["perm_support_<name>_type"]  ("FIXED" | "PINNED" | "ROLLER_X" | ...)

PERM_DOF_NAMES = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')


def perm_support_dofs(obj, suffix) -> List[str]:
    """DOFs fixed by a CUSTOM support group, in the order ux ... rz."""
    raw = str(obj.get(f"perm_support_{suffix}_dofs", "") or "")
    got = {d.strip().lower() for d in raw.split(",")}
    return [d for d in PERM_DOF_NAMES if d in got]


def find_perm_load_groups(obj):
    """Return list of (group_name, suffix, fx, fy, fz) for all PERM load groups.
    suffix is the part after 'PERM_Load_' (or '' for legacy 'PERM_Loads')."""
    out = []
    for vg in obj.vertex_groups:
        if vg.name == "PERM_Loads":
            # Legacy single-group mode — values stored on obj custom props if present
            fx = float(obj.get("perm_legacy_load_fx", 0.0))
            fy = float(obj.get("perm_legacy_load_fy", 0.0))
            fz = float(obj.get("perm_legacy_load_fz", -1000.0))
            out.append(("PERM_Loads", "(legacy)", fx, fy, fz))
        elif vg.name.startswith("PERM_Load_"):
            suffix = vg.name[len("PERM_Load_"):]
            prop_prefix = f"perm_load_{suffix}_"
            fx = float(obj.get(prop_prefix + "fx", 0.0))
            fy = float(obj.get(prop_prefix + "fy", 0.0))
            fz = float(obj.get(prop_prefix + "fz", 0.0))
            out.append((vg.name, suffix, fx, fy, fz))
    return out


def find_perm_support_groups(obj):
    """Return list of (group_name, suffix, bc_type) for all PERM support groups.
    bc_type is one of 'FIXED', 'PINNED', 'ROLLER_X', 'ROLLER_Y', 'ROLLER_Z'."""
    out = []
    for vg in obj.vertex_groups:
        if vg.name == "PERM_Supports":
            bc_type = obj.get("perm_legacy_support_type", "FIXED")
            out.append(("PERM_Supports", "(legacy)", bc_type))
        elif vg.name.startswith("PERM_Support_"):
            suffix = vg.name[len("PERM_Support_"):]
            bc_type = obj.get(f"perm_support_{suffix}_type", "FIXED")
            out.append((vg.name, suffix, bc_type))
    return out


def get_group_vertex_indices(obj, group_name):
    """Return set of vertex indices belonging to the named vertex group."""
    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        return set()
    indices = set()
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                indices.add(v.index)
                break
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL MARKER GENERATION (load arrows, BC locks)
# ─────────────────────────────────────────────────────────────────────────────

PERM_MARKER_COLLECTION = "PERM_Markers"


def _ensure_marker_collection():
    """Create or get the PERM_Markers collection."""
    if PERM_MARKER_COLLECTION not in bpy.data.collections:
        coll = bpy.data.collections.new(PERM_MARKER_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
    return bpy.data.collections[PERM_MARKER_COLLECTION]


def clear_perm_markers():
    """Remove all PERM marker objects and their meshes."""
    coll = bpy.data.collections.get(PERM_MARKER_COLLECTION)
    if coll is None:
        return
    # Remove all child objects
    objs_to_remove = list(coll.objects)
    for o in objs_to_remove:
        mesh = o.data if hasattr(o, 'data') else None
        bpy.data.objects.remove(o, do_unlink=True)
        if mesh is not None and isinstance(mesh, bpy.types.Mesh) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _make_arrow_mesh(name, length, shaft_radius, head_radius, head_length, color):
    """Build a single arrow mesh pointing in +Z direction with given length.
    Returns the mesh data block. Tip of arrow is at z=0, tail at z=-length."""
    verts = []
    faces = []
    # Cylinder shaft, from z=-length to z=-head_length, low-poly (6-sided)
    n_sides = 6
    z_tail = -length
    z_neck = -head_length
    for k in range(n_sides):
        ang = 2 * math.pi * k / n_sides
        cx = shaft_radius * math.cos(ang)
        cy = shaft_radius * math.sin(ang)
        verts.append((cx, cy, z_tail))  # tail ring (idx 0..n_sides-1)
        verts.append((cx, cy, z_neck))  # neck ring (idx n_sides..2*n_sides-1)
    # Cone head: ring at z=-head_length, tip at z=0
    for k in range(n_sides):
        ang = 2 * math.pi * k / n_sides
        cx = head_radius * math.cos(ang)
        cy = head_radius * math.sin(ang)
        verts.append((cx, cy, z_neck))  # head ring (idx 2*n_sides..3*n_sides-1)
    verts.append((0, 0, 0))  # tip (idx 3*n_sides)

    # Shaft side faces
    for k in range(n_sides):
        k1 = (k + 1) % n_sides
        faces.append((k, k1, n_sides + k1, n_sides + k))  # tail ring → neck ring
    # Head ring → tip (triangular fan)
    tip_idx = 3 * n_sides
    for k in range(n_sides):
        k1 = (k + 1) % n_sides
        faces.append((2*n_sides + k, 2*n_sides + k1, tip_idx))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    # Vertex color
    if hasattr(mesh, 'color_attributes'):
        attr = mesh.color_attributes.new(name="Color", type='BYTE_COLOR', domain='CORNER')
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                attr.data[loop_idx].color = color
    return mesh


def _make_bc_cube_mesh(name, size, color):
    """FIXED support marker: a small cube."""
    half = size * 0.5
    verts = [
        Vector((-half, -half, -half)), Vector(( half, -half, -half)),
        Vector(( half,  half, -half)), Vector((-half,  half, -half)),
        Vector((-half, -half,  half)), Vector(( half, -half,  half)),
        Vector(( half,  half,  half)), Vector((-half,  half,  half)),
    ]
    faces = [
        (0,1,2,3), (4,7,6,5), (0,4,5,1),
        (1,5,6,2), (2,6,7,3), (3,7,4,0),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    # Color via material
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = color
    mesh.materials.append(mat)
    return mesh


def _make_bc_cone_mesh(name, size, color):
    """PINNED support marker: a cone pointing up."""
    n_segs = 8
    h = size
    r = size * 0.4
    verts = [Vector((0, 0, h * 0.5))]  # apex
    for i in range(n_segs):
        ang = 2 * math.pi * i / n_segs
        verts.append(Vector((r * math.cos(ang), r * math.sin(ang), -h * 0.5)))
    faces = []
    # Side triangles
    for i in range(n_segs):
        next_i = (i + 1) % n_segs
        faces.append((0, 1 + i, 1 + next_i))
    # Base (one polygon)
    faces.append(tuple(range(1, n_segs + 1)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = color
    mesh.materials.append(mat)
    return mesh


def _make_bc_cylinder_mesh(name, size, color):
    """ROLLER support marker: a horizontal cylinder."""
    n_segs = 12
    r = size * 0.35
    h = size * 0.8
    verts = []
    # Two circles at z = -h/2 and z = +h/2
    for z in (-h*0.5, h*0.5):
        for i in range(n_segs):
            ang = 2 * math.pi * i / n_segs
            verts.append(Vector((r * math.cos(ang), r * math.sin(ang), z)))
    # Bottom and top center verts
    verts.append(Vector((0, 0, -h*0.5)))   # idx 2*n_segs
    verts.append(Vector((0, 0,  h*0.5)))   # idx 2*n_segs+1
    faces = []
    # Side quads
    for i in range(n_segs):
        next_i = (i + 1) % n_segs
        bot1 = i
        bot2 = next_i
        top1 = n_segs + i
        top2 = n_segs + next_i
        faces.append((bot1, bot2, top2, top1))
    # Bottom and top caps
    for i in range(n_segs):
        next_i = (i + 1) % n_segs
        faces.append((2*n_segs, next_i, i))
        faces.append((2*n_segs+1, n_segs + i, n_segs + next_i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = color
    mesh.materials.append(mat)
    return mesh


def create_marker_objects(obj):
    """Generate visual marker objects for all loaded supports and loads.
    
    Loads: arrows pointing in force direction, sized by relative magnitude,
           colored per group (yellow/orange/red for different groups).
    Supports: small pyramid icons at each constrained vertex, color-coded
              by BC type (gray=fixed, light gray=pinned, blue=roller).
    
    Markers are placed in the PERM_Markers collection. Re-running clears
    previous markers.
    """
    # Clean up previous markers
    clear_perm_markers()
    marker_coll = _ensure_marker_collection()

    # Compute a scene scale to size markers proportionally
    bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in bbox_corners]
    ys = [c.y for c in bbox_corners]
    zs = [c.z for c in bbox_corners]
    scene_extent = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 0.1)
    marker_size = scene_extent * 0.02  # 2% of model extent (will be scaled by user)
    
    # Read scene-level marker scale (default 1.0 if not set)
    scn = bpy.context.scene
    user_scale = float(getattr(scn, 'perm_marker_scale', 1.0))

    # ── LOADS: arrows ────────────────────────────────────────────────────
    load_groups = find_perm_load_groups(obj)
    load_colors = [
        (1.0, 0.85, 0.0, 1.0),  # yellow
        (1.0, 0.5,  0.0, 1.0),  # orange
        (1.0, 0.2,  0.2, 1.0),  # red
        (0.7, 0.1,  0.7, 1.0),  # purple
        (0.0, 0.7,  0.7, 1.0),  # cyan
    ]
    # Compute reference force magnitude for arrow scaling
    max_force_mag = 0.0
    for _, _, fx, fy, fz in load_groups:
        mag = math.sqrt(fx*fx + fy*fy + fz*fz)
        max_force_mag = max(max_force_mag, mag)
    if max_force_mag < 1e-15:
        max_force_mag = 1.0  # avoid div by 0

    for gidx, (group_name, suffix, fx, fy, fz) in enumerate(load_groups):
        magnitude = math.sqrt(fx*fx + fy*fy + fz*fz)
        if magnitude < 1e-15:
            continue
        # Per-node force is total/N (matching current solver behavior)
        node_idx_set = get_group_vertex_indices(obj, group_name)
        if not node_idx_set:
            continue

        # Arrow direction is the force direction (unit vector)
        dir_x = fx / magnitude
        dir_y = fy / magnitude
        dir_z = fz / magnitude

        # Arrow length scales with relative magnitude (0.5–1.5 × scene_extent × 0.10)
        rel_mag = magnitude / max_force_mag
        arrow_length = scene_extent * 0.10 * (0.5 + rel_mag) * user_scale
        shaft_r = marker_size * 0.25 * user_scale
        head_r = marker_size * 0.5 * user_scale
        head_len = arrow_length * 0.3

        # Per-group color
        color = load_colors[gidx % len(load_colors)]

        # Create one arrow mesh, reuse for all nodes
        arrow_mesh = _make_arrow_mesh(
            f"PERM_LoadArrow_{suffix}_mesh", arrow_length, shaft_r, head_r, head_len, color)

        # For each node, instance the arrow oriented toward force direction.
        # Tip of arrow is at the node location; arrow points BACKWARD from
        # node (i.e., shows the force vector applied AT this node).
        # Default arrow points +Z; rotate to match (dir_x, dir_y, dir_z).
        for vidx in node_idx_set:
            vpos = obj.matrix_world @ obj.data.vertices[vidx].co
            arrow_obj = bpy.data.objects.new(
                f"PERM_LoadArrow_{suffix}_v{vidx}", arrow_mesh)
            arrow_obj.location = vpos
            # Rotate so arrow points in (dir_x, dir_y, dir_z) direction
            # Default points in +Z; build rotation from +Z to target
            target = Vector((dir_x, dir_y, dir_z))
            up = Vector((0, 0, 1))
            if (target - up).length < 1e-6:
                pass  # already pointing up
            elif (target + up).length < 1e-6:
                arrow_obj.rotation_euler = (math.pi, 0, 0)
            else:
                axis = up.cross(target)
                axis.normalize()
                angle = math.acos(max(-1.0, min(1.0, up.dot(target))))
                arrow_obj.rotation_mode = 'AXIS_ANGLE'
                arrow_obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
            marker_coll.objects.link(arrow_obj)

    # ── SUPPORTS: pyramid icons ──────────────────────────────────────────
    # ── SUPPORTS: shape per BC type, color per BC type ───────────────────
    support_groups = find_perm_support_groups(obj)
    bc_colors = {
        'FIXED':    (0.85, 0.15, 0.15, 1.0),  # red — fully fixed
        'PINNED':   (0.2,  0.4,  0.85, 1.0),  # blue — pinned
        'ROLLER_X': (0.95, 0.85, 0.2,  1.0),  # yellow — roller along x
        'ROLLER_Y': (0.3,  0.85, 0.3,  1.0),  # green — roller along y
        'ROLLER_Z': (0.85, 0.5,  0.85, 1.0),  # pink — roller along z
        'CUSTOM':   (0.95, 0.55, 0.1,  1.0),  # orange — custom set of DOFs
    }
    for gidx, (group_name, suffix, bc_type) in enumerate(support_groups):
        node_idx_set = get_group_vertex_indices(obj, group_name)
        if not node_idx_set:
            continue
        color = bc_colors.get(bc_type, (0.3, 0.3, 0.3, 1.0))
        # Shape: cube for FIXED, cone for PINNED, cylinder for ROLLER
        bc_size = marker_size * user_scale
        if bc_type == 'FIXED':
            shape_mesh = _make_bc_cube_mesh(
                f"PERM_BCLock_{suffix}_mesh", bc_size * 1.2, color)
        elif bc_type == 'PINNED':
            shape_mesh = _make_bc_cone_mesh(
                f"PERM_BCLock_{suffix}_mesh", bc_size * 1.5, color)
        elif bc_type in ('ROLLER_X', 'ROLLER_Y', 'ROLLER_Z'):
            shape_mesh = _make_bc_cylinder_mesh(
                f"PERM_BCLock_{suffix}_mesh", bc_size * 1.4, color)
        else:
            shape_mesh = _make_bc_cube_mesh(
                f"PERM_BCLock_{suffix}_mesh", bc_size, color)
        for vidx in node_idx_set:
            vpos = obj.matrix_world @ obj.data.vertices[vidx].co
            lock_obj = bpy.data.objects.new(
                f"PERM_BCLock_{suffix}_v{vidx}", shape_mesh)
            lock_obj.location = vpos
            # Orient roller cylinder along its rolling axis
            if bc_type == 'ROLLER_X':
                lock_obj.rotation_euler = (0, math.pi/2, 0)  # axis along X
            elif bc_type == 'ROLLER_Y':
                lock_obj.rotation_euler = (math.pi/2, 0, 0)  # axis along Y
            # ROLLER_Z is default (cylinder axis along Z)
            marker_coll.objects.link(lock_obj)

    # Set vertex color shading on viewport so markers are visible by color
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'

    n_loads = sum(len(get_group_vertex_indices(obj, n)) for n, _, _, _, _ in load_groups)
    n_supports = sum(len(get_group_vertex_indices(obj, n)) for n, _, _ in support_groups)
    print(f"[PERM Markers] Created {n_loads} load arrows + {n_supports} BC locks "
          f"({len(load_groups)} load groups, {len(support_groups)} support groups)")


# ─────────────────────────────────────────────────────────────────────────────
# PER-NODE FORCE EDITING (PropertyGroup, UIList, save/load helpers)
# ─────────────────────────────────────────────────────────────────────────────
# Each load group can optionally have per-node force overrides stored as JSON:
#     obj["perm_load_<suffix>_per_node"] = '{"42": [0, 0, -500], "43": [0, 0, -1500]}'
# Keys are node ids (string), values are [fx, fy, fz].
# When solving, the per-node value is used if present, else the group's default.

import json as _json

def _per_node_key(suffix):
    """Property key holding the per-node force JSON for a load group suffix."""
    return f"perm_load_{suffix}_per_node"


def get_per_node_forces(obj, suffix):
    """Return dict {vertex_id (int): [fx, fy, fz]} of per-node overrides.
    Returns empty dict if none stored."""
    raw = obj.get(_per_node_key(suffix), "")
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        # Normalize keys to int
        return {int(k): list(map(float, v)) for k, v in data.items()}
    except Exception:
        return {}


def set_per_node_forces(obj, suffix, per_node_dict):
    """Write per-node force overrides to the object as JSON.
    per_node_dict: {vertex_id: (fx, fy, fz)}"""
    if not per_node_dict:
        if _per_node_key(suffix) in obj.keys():
            del obj[_per_node_key(suffix)]
        return
    obj[_per_node_key(suffix)] = _json.dumps(
        {str(k): [float(f) for f in v] for k, v in per_node_dict.items()})


# ─────────────────────────────────────────────────────────────────────────────
# LOAD COORDINATES -- where each point load acts, for hand calculations
# ─────────────────────────────────────────────────────────────────────────────
# Every time a load is assigned, when "Show Load Coordinates" is clicked and at
# every analysis run, PERM prints for each point-load group:
#   * every loaded node in WORLD coordinates (m) and its force (N)
#   * the resultant point (force-weighted centre) of the group
#   * its position along the span measured from the support groups
#     (a, b, a/L, with L = distance between the two support centres)
#   * its sideways offset e from the line joining the support centres
#     (the lever arm for torsion of the girder)
# The same table goes to Text Editor -> 'PERM_Loads.txt' (copy/paste it), and
# each group's resultant point is stored on the object as
# obj["perm_load_<name>_at"] = [x, y, z] (shown under the load in the panel).
def _perm_report_object_loads(obj, only=None):
    """Report where the point loads of `obj` act (see block comment above).
    Object Mode only. Returns {load name: summary dict}."""
    me = obj.data
    R = np.array(obj.matrix_world.to_3x3()); T = np.array(obj.matrix_world.translation)
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
    W = co.reshape(-1, 3) @ R.T + T
    want = {vg.index: vg.name for vg in obj.vertex_groups
            if vg.name.startswith("PERM_Load_") or vg.name.startswith("PERM_Support_")}
    members = {n: [] for n in want.values()}
    if want:
        for v in me.vertices:
            for g in v.groups:
                n = want.get(g.group)
                if n is not None:
                    members[n].append(v.index)
    sups = [(sfx, bc, W[members[nm]].mean(axis=0), len(members[nm]))
            for nm, sfx, bc in find_perm_support_groups(obj) if members.get(nm)]
    if len(sups) >= 2:
        # measure along +X (or +Y): the first support is the one at the lower
        # coordinate, so a, a/L and the sign of e do not depend on group names
        d01 = sups[1][2] - sups[0][2]
        if (d01[0] if abs(d01[0]) >= abs(d01[1]) else d01[1]) < 0:
            sups[0], sups[1] = sups[1], sups[0]
        c1, c2 = sups[0][2], sups[1][2]
        ax = (c2 - c1) * np.array([1.0, 1.0, 0.0]); L = float(np.linalg.norm(ax))
        ax = ax / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        ext = W.max(axis=0) - W.min(axis=0)
        ax = np.array([1.0, 0.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0, 0.0])
        c1 = c2 = None; L = 0.0
    across = np.cross([0.0, 0.0, 1.0], ax)
    axn = "X" if abs(ax[0]) > 0.98 else ("Y" if abs(ax[1]) > 0.98 else "the line between the supports")
    trn = "Y" if axn == "X" else ("X" if axn == "Y" else "across")
    out = ["Where the point loads act (WORLD coordinates, m) -- share these for hand calculations"]
    for sfx, bc, c, n in sups:
        out.append(f"  support {sfx:<8} {bc:<9} centre x={c[0]:+.3f}  y={c[1]:+.3f}  z={c[2]:+.3f}  ({n} nodes)")
    if L > 0:
        out.append(f"  span between support centres: L = {L:.3f} m along {axn}")
    csv = ["load,vertex,x,y,z,Fx,Fy,Fz"]
    summary = {}
    for nm, sfx, fx, fy, fz in find_perm_load_groups(obj):
        if only is not None and sfx != only:
            continue
        ids = members.get(nm, [])
        if not ids:
            continue
        pn = get_per_node_forces(obj, sfx) if not sfx.startswith("(") else {}
        if pn:
            F = np.array([pn.get(i, [0.0, 0.0, 0.0]) for i in ids], float)
        else:                                   # uniform group: total split equally
            F = np.tile(np.array([fx, fy, fz], float) / len(ids), (len(ids), 1))
        Ft = F.sum(axis=0); wgt = np.linalg.norm(F, axis=1)
        p = (W[ids] * wgt[:, None]).sum(axis=0) / wgt.sum() if wgt.sum() > 0 else W[ids].mean(axis=0)
        obj[f"perm_load_{sfx}_at"] = [float(p[0]), float(p[1]), float(p[2])]
        out.append(f"  Load '{sfx}': {len(ids)} node(s), total F = ({Ft[0]:+.1f}, {Ft[1]:+.1f}, {Ft[2]:+.1f}) N")
        out.append(f"     resultant at x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}")
        s_info = {}
        if L > 0:
            a = float((p - c1) @ ax)
            line = c1 + (c2 - c1) * np.clip(a / L, 0.0, 1.0)
            e = float((p - line) @ across)
            out.append(f"     along {axn}: a = {a:.3f} m from {sups[0][0]} ({sups[0][1]}), "
                       f"b = {L - a:.3f} m from {sups[1][0]} ({sups[1][1]})  ->  a/L = {a / L:.3f}")
            out.append(f"     across ({trn}): e = {e:+.3f} m from the line joining the support centres "
                       f"(+ = towards +{trn}; lever arm for torsion)")
            s_info = dict(a=a, b=L - a, L=L, e=e)
        for k, i in enumerate(ids):
            x, y, z = W[i]
            csv.append(f"{sfx},{i},{x:.4f},{y:.4f},{z:.4f},{F[k, 0]:.3f},{F[k, 1]:.3f},{F[k, 2]:.3f}")
            if k < 12:
                out.append(f"     node {i}: ({x:+.3f}, {y:+.3f}, {z:+.3f})  F = ({F[k, 0]:+.1f}, {F[k, 1]:+.1f}, {F[k, 2]:+.1f}) N")
        if len(ids) > 12:
            out.append(f"     ... {len(ids) - 12} more node(s): see Text Editor -> PERM_Loads.txt")
        summary[sfx] = dict(at=p, n=len(ids), F=Ft, **s_info)
    if not summary:
        out.append("  (no point loads)")
    for ln in out:
        print("[PERM Loads] " + ln)
    try:
        txt = bpy.data.texts.get("PERM_Loads.txt") or bpy.data.texts.new("PERM_Loads.txt")
        txt.clear(); txt.write("\n".join(out) + "\n\n" + "\n".join(csv) + "\n")
    except Exception as e:
        print(f"[PERM Loads] could not write the text block: {e}")
    return summary


class PERM_PerNodeForce(bpy.types.PropertyGroup):
    """One row in the per-node force UIList editor."""
    vert_idx: bpy.props.IntProperty(name="Vertex", default=-1)
    fx: bpy.props.FloatProperty(name="Fx (N)", default=0.0)
    fy: bpy.props.FloatProperty(name="Fy (N)", default=0.0)
    fz: bpy.props.FloatProperty(name="Fz (N)", default=0.0)
    # Position for display (helps user identify which vert is which)
    px: bpy.props.FloatProperty(default=0.0)
    py: bpy.props.FloatProperty(default=0.0)
    pz: bpy.props.FloatProperty(default=0.0)


class PERM_UL_PerNodeForces(bpy.types.UIList):
    """UIList rendering one row per node with editable Fx/Fy/Fz."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=f"v{item.vert_idx}", icon='VERTEXSEL')
            row.label(text=f"({item.px:.2f}, {item.py:.2f}, {item.pz:.2f})")
            row.prop(item, "fx", text="")
            row.prop(item, "fy", text="")
            row.prop(item, "fz", text="")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=f"v{item.vert_idx}")


class PERM_OT_OpenPerNodeEditor(bpy.types.Operator):
    """PERM: Open the per-node force editor for a load group.
    
    Populates a scene-level collection from the group's stored per-node JSON
    (or from the group's default force if no overrides exist yet), then opens
    a UIList dialog where each node can be edited individually.
    """
    bl_idname = "perm.open_per_node_editor"
    bl_label = "PERM: Edit Per-Node Forces"
    bl_options = {'REGISTER'}
    
    group_name: bpy.props.StringProperty(default="")
    
    def invoke(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or not self.group_name.startswith("PERM_Load_") \
                or self.group_name not in obj.vertex_groups:
            self.report({'ERROR'}, "Select a mesh and a PERM_Load_* group")
            return {'CANCELLED'}
        self._fill(context, obj)
        # Show dialog
        return context.window_manager.invoke_props_dialog(self, width=600)

    def _fill(self, context, obj):
        """Scene list of the editor <- the group's stored per-node forces."""
        suffix = self.group_name[len("PERM_Load_"):]
        
        # Get group's default force
        prefix = f"perm_load_{suffix}_"
        default_fx = float(obj.get(prefix + "fx", 0.0))
        default_fy = float(obj.get(prefix + "fy", 0.0))
        default_fz = float(obj.get(prefix + "fz", 0.0))
        N = len(get_group_vertex_indices(obj, self.group_name))
        # If group default is "TOTAL", per-node would be default/N
        per_node_default_x = default_fx / max(N, 1)
        per_node_default_y = default_fy / max(N, 1)
        per_node_default_z = default_fz / max(N, 1)
        
        existing_overrides = get_per_node_forces(obj, suffix)
        
        # Populate the scene collection
        scn = context.scene
        scn.perm_per_node_list.clear()
        scn.perm_per_node_group = self.group_name
        
        vert_indices = sorted(get_group_vertex_indices(obj, self.group_name))
        for vidx in vert_indices:
            item = scn.perm_per_node_list.add()
            item.vert_idx = vidx
            v = obj.data.vertices[vidx]
            item.px = v.co.x
            item.py = v.co.y
            item.pz = v.co.z
            if vidx in existing_overrides:
                ovr = existing_overrides[vidx]
                item.fx = ovr[0]; item.fy = ovr[1]; item.fz = ovr[2]
            else:
                # Default to group default / N (per-node interpretation)
                item.fx = per_node_default_x
                item.fy = per_node_default_y
                item.fz = per_node_default_z
    
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        layout.label(text=f"Per-node forces for '{self.group_name}':", icon='FORCE_FORCE')
        layout.label(text="Each row = one vertex. Edit Fx, Fy, Fz individually.",
                     icon='INFO')
        layout.separator()
        row = layout.row()
        row.template_list("PERM_UL_PerNodeForces", "",
                          scn, "perm_per_node_list",
                          scn, "perm_per_node_index",
                          rows=10)
        layout.separator()
        # Bulk actions
        row = layout.row(align=True)
        row.operator("perm.bulk_set_per_node", text="Set all Fz", icon='COPYDOWN').axis = 'z'
        row.operator("perm.bulk_set_per_node", text="Set all Fx", icon='COPYDOWN').axis = 'x'
        row.operator("perm.bulk_set_per_node", text="Set all Fy", icon='COPYDOWN').axis = 'y'
    
    def execute(self, context):
        # Save the per-node values back to the object
        obj = context.active_object
        scn = context.scene
        if not obj or obj.type != 'MESH' or not self.group_name.startswith("PERM_Load_") \
                or self.group_name not in obj.vertex_groups:
            self.report({'ERROR'}, "Select a mesh and a PERM_Load_* group")
            return {'CANCELLED'}
        suffix = self.group_name[len("PERM_Load_"):]
        # The scene list must be this group's: run without the dialog (a script, Redo) it may hold
        # another group's rows or none, and saving it would wipe this group's forces.
        if scn.perm_per_node_group != self.group_name or len(scn.perm_per_node_list) == 0:
            self._fill(context, obj)

        per_node = {}
        for item in scn.perm_per_node_list:
            per_node[item.vert_idx] = (item.fx, item.fy, item.fz)
        
        set_per_node_forces(obj, suffix, per_node)
        
        # Also update the group "default" props to reflect the sum (so the panel
        # listing still shows a meaningful total).
        total_fx = sum(p[0] for p in per_node.values())
        total_fy = sum(p[1] for p in per_node.values())
        total_fz = sum(p[2] for p in per_node.values())
        obj[f"perm_load_{suffix}_fx"] = total_fx
        obj[f"perm_load_{suffix}_fy"] = total_fy
        obj[f"perm_load_{suffix}_fz"] = total_fz
        
        # Auto-refresh markers (Feature 2)
        try:
            create_marker_objects(obj)
        except Exception as e:
            print(f"[PERM] Marker refresh failed after per-node edit: {e}")
        
        self.report({'INFO'},
            f"PERM: Saved per-node forces for '{self.group_name}' "
            f"(total = {total_fx:.1f}, {total_fy:.1f}, {total_fz:.1f} N)")
        return {'FINISHED'}


class PERM_OT_BulkSetPerNode(bpy.types.Operator):
    """PERM: Bulk-set one axis (Fx/Fy/Fz) for all rows in the per-node editor."""
    bl_idname = "perm.bulk_set_per_node"
    bl_label = "Bulk Set Force"
    bl_options = {'REGISTER', 'UNDO'}
    
    axis: bpy.props.EnumProperty(items=[('x', "X", ""), ('y', "Y", ""), ('z', "Z", "")],
                                  default='z')
    value: bpy.props.FloatProperty(name="Value (N)", default=0.0)
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)
    
    def execute(self, context):
        scn = context.scene
        attr = f"f{self.axis}"
        for item in scn.perm_per_node_list:
            setattr(item, attr, self.value)
        self.report({'INFO'}, f"PERM: Set F{self.axis.upper()}={self.value} on all rows")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL REMOVAL OPERATORS
# Three ways to remove individual nodes/faces from PERM groups:
#   1. PERM_OT_RemoveSelectedFromGroups: edit-mode, removes selected verts from
#      all PERM groups they belong to (handles accidental face inclusion).
#   2. PERM_OT_RemoveMarker: object-mode, click marker(s), remove that node
#      from its group.
#   3. PERM_OT_DeleteGroup: existing — removes whole group.
# ─────────────────────────────────────────────────────────────────────────────

class PERM_OT_RemoveSelectedFromGroups(bpy.types.Operator):
    """PERM: Remove selected vertices/faces from any PERM group(s) they belong to.
    
    Useful when you accidentally included extra faces in a group — select just
    the unwanted ones (in Edit Mode) and click this. The other vertices in the
    group stay assigned.
    
    If a group becomes empty after removal, it's deleted automatically.
    """
    bl_idname = "perm.remove_from_groups"
    bl_label = "PERM: Remove Selected from PERM Groups"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select vertices/faces first")
            return {'CANCELLED'}
        
        bpy.ops.object.mode_set(mode='OBJECT')
        selected = [v.index for v in obj.data.vertices if v.select]
        if not selected:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "No vertices selected"); return {'CANCELLED'}
        
        # Find all PERM groups these verts belong to
        perm_groups = [vg for vg in obj.vertex_groups
                       if vg.name == "PERM_Loads" or vg.name == "PERM_Supports"
                       or vg.name.startswith("PERM_Load_") or vg.name.startswith("PERM_Support_")]
        
        n_removed_total = 0
        groups_emptied = []
        for vg in perm_groups:
            # Remove selected verts from this group
            verts_in_group = get_group_vertex_indices(obj, vg.name)
            to_remove = [v for v in selected if v in verts_in_group]
            if not to_remove:
                continue
            vg.remove(to_remove)
            n_removed_total += len(to_remove)
            # Check if group is now empty
            remaining = get_group_vertex_indices(obj, vg.name)
            if not remaining:
                # Delete empty group + its custom properties
                groups_emptied.append(vg.name)
                _cleanup_group_props(obj, vg.name)
                obj.vertex_groups.remove(vg)
        
        # Also clean up per-node JSON for removed verts (in load groups)
        for vg in obj.vertex_groups:
            if not vg.name.startswith("PERM_Load_"):
                continue
            suffix = vg.name[len("PERM_Load_"):]
            pn = get_per_node_forces(obj, suffix)
            if not pn:
                continue
            changed = False
            for vidx in selected:
                if vidx in pn:
                    del pn[vidx]
                    changed = True
            if changed:
                set_per_node_forces(obj, suffix, pn)
        
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Auto-refresh markers
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            create_marker_objects(obj)
        except Exception as e:
            print(f"[PERM] Marker refresh failed: {e}")
        bpy.ops.object.mode_set(mode='EDIT')
        
        msg = f"PERM: Removed {n_removed_total} vertex-group assignments"
        if groups_emptied:
            msg += f"; deleted empty groups: {', '.join(groups_emptied)}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class PERM_OT_RemoveMarker(bpy.types.Operator):
    """PERM: Remove the assignment(s) shown by the selected marker object(s).
    
    Select arrow or pyramid marker objects (in Object Mode), click this.
    The corresponding nodes are unassigned from their groups, and the
    markers are deleted. If a group becomes empty, it's deleted too.
    """
    bl_idname = "perm.remove_marker"
    bl_label = "PERM: Remove Selected Marker(s)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_markers = [o for o in context.selected_objects
                            if o.name.startswith("PERM_LoadArrow_") 
                            or o.name.startswith("PERM_BCLock_")]
        if not selected_markers:
            self.report({'ERROR'},
                "Select one or more PERM_LoadArrow_* or PERM_BCLock_* objects in Object Mode")
            return {'CANCELLED'}
        
        # Find the parent mesh: the active object if it's a mesh, OR the most
        # recently used mesh (heuristic: any mesh in scene with PERM groups)
        target = None
        for o in bpy.data.objects:
            if o.type == 'MESH' and any(
                vg.name.startswith("PERM_Load_") or vg.name.startswith("PERM_Support_")
                or vg.name in ("PERM_Loads", "PERM_Supports")
                for vg in o.vertex_groups):
                target = o
                break
        if target is None:
            self.report({'ERROR'}, "No PERM mesh found")
            return {'CANCELLED'}
        
        # Parse each marker's name: PERM_LoadArrow_<suffix>_v<idx> or PERM_BCLock_<suffix>_v<idx>
        n_unassigned = 0
        affected_groups = set()
        for marker in selected_markers:
            name = marker.name
            if name.startswith("PERM_LoadArrow_"):
                rest = name[len("PERM_LoadArrow_"):]
                vg_prefix = "PERM_Load_"
            elif name.startswith("PERM_BCLock_"):
                rest = name[len("PERM_BCLock_"):]
                vg_prefix = "PERM_Support_"
            else:
                continue
            
            # Split out vertex index: <suffix>_v<idx>
            if "_v" not in rest:
                continue
            suffix_part, _, vidx_str = rest.rpartition("_v")
            try:
                vidx = int(vidx_str)
            except ValueError:
                continue
            
            # Find group: PERM_Load_<suffix_part> or PERM_Support_<suffix_part>
            # (Legacy groups may have no suffix → handled by "(legacy)" string)
            if suffix_part == "(legacy)":
                group_name = "PERM_Loads" if vg_prefix == "PERM_Load_" else "PERM_Supports"
            else:
                group_name = vg_prefix + suffix_part
            
            vg = target.vertex_groups.get(group_name)
            if vg is None:
                continue
            vg.remove([vidx])
            # Also remove from per-node JSON if a load
            if group_name.startswith("PERM_Load_"):
                suffix = group_name[len("PERM_Load_"):]
                pn = get_per_node_forces(target, suffix)
                if vidx in pn:
                    del pn[vidx]
                    set_per_node_forces(target, suffix, pn)
            affected_groups.add(group_name)
            n_unassigned += 1
        
        # Delete empty groups
        for group_name in list(affected_groups):
            if not get_group_vertex_indices(target, group_name):
                _cleanup_group_props(target, group_name)
                target.vertex_groups.remove(target.vertex_groups[group_name])
        
        # Refresh markers (this also deletes the selected ones since their nodes
        # are no longer in groups)
        try:
            create_marker_objects(target)
        except Exception as e:
            print(f"[PERM] Marker refresh failed: {e}")
        
        self.report({'INFO'},
            f"PERM: Removed {n_unassigned} node assignment(s) from {len(affected_groups)} group(s)")
        return {'FINISHED'}


def _cleanup_group_props(obj, group_name):
    """Remove custom properties associated with a PERM vertex group."""
    if group_name == "PERM_Loads":
        for key in ("perm_legacy_load_fx", "perm_legacy_load_fy", "perm_legacy_load_fz"):
            if key in obj.keys(): del obj[key]
    elif group_name == "PERM_Supports":
        if "perm_legacy_support_type" in obj.keys():
            del obj["perm_legacy_support_type"]
    elif group_name.startswith("PERM_Load_"):
        suffix = group_name[len("PERM_Load_"):]
        for axis in ('fx', 'fy', 'fz'):
            k = f"perm_load_{suffix}_{axis}"
            if k in obj.keys(): del obj[k]
        pn_key = f"perm_load_{suffix}_per_node"
        if pn_key in obj.keys(): del obj[pn_key]
    elif group_name.startswith("PERM_Support_"):
        suffix = group_name[len("PERM_Support_"):]
        k = f"perm_support_{suffix}_type"
        if k in obj.keys(): del obj[k]


class PERM_OT_ShowMarkers(bpy.types.Operator):
    """PERM: Generate / refresh visual markers for all loads and supports"""
    bl_idname = "perm.show_markers"
    bl_label = "PERM: Show Load/BC Markers"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object"); return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            create_marker_objects(obj)
            self.report({'INFO'}, "PERM: Markers refreshed in 'PERM_Markers' collection")
        except Exception as e:
            self.report({'ERROR'}, f"PERM: marker generation failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        return {'FINISHED'}


class PERM_OT_ClearMarkers(bpy.types.Operator):
    """PERM: Remove all load/BC visual markers"""
    bl_idname = "perm.clear_markers"
    bl_label = "PERM: Clear Markers"
    bl_options = {'REGISTER'}

    def execute(self, context):
        clear_perm_markers()
        self.report({'INFO'}, "PERM: Markers cleared")
        return {'FINISHED'}


class PERM_OT_ClearAssignments(bpy.types.Operator):
    """Remove every restraint and every load of the model (point loads, surface loads) and their markers. Materials, sections, element types, offsets and connections are kept"""
    bl_idname = "perm.clear_assignments"
    bl_label = "Clear All Supports and Loads"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        # Remove legacy single groups
        for name in ["PERM_Supports", "PERM_Loads"]:
            if name in obj.vertex_groups:
                obj.vertex_groups.remove(obj.vertex_groups[name])
        # Remove all named PERM_Load_* and PERM_Support_* groups
        to_remove = [vg for vg in obj.vertex_groups
                     if vg.name.startswith("PERM_Load_")
                     or vg.name.startswith("PERM_Support_")]
        for vg in to_remove:
            obj.vertex_groups.remove(vg)
        # Strip the custom properties of the supports and loads ONLY. The id tables of the face
        # materials / sections / element types (perm_face_*_ids) must stay: without them the faces keep
        # their numbers, the next assignment re-uses number 1 for another name, and faces change material
        # without any message.
        keys_to_remove = [k for k in obj.keys()
                          if k.startswith(("perm_load_", "perm_support_", "perm_legacy_", "perm_surface_load"))]
        for k in keys_to_remove:
            del obj[k]
        if obj.mode == 'OBJECT':                           # surface loads
            _a = obj.data.attributes.get(PERM_SURFLOAD_ATTR)
            if _a is not None:
                obj.data.attributes.remove(_a)
        else:
            _bm = bmesh.from_edit_mesh(obj.data)
            _lay = _bm.faces.layers.float_vector.get(PERM_SURFLOAD_ATTR)
            if _lay is not None:
                _bm.faces.layers.float_vector.remove(_lay)
                bmesh.update_edit_mesh(obj.data)
        # Clean markers
        clear_perm_markers()
        self.report({'INFO'}, "PERM: all supports and loads cleared (materials, sections, element types kept)")
        return {'FINISHED'}


class PERM_OT_SelectGroup(bpy.types.Operator):
    """PERM: Select vertices in a PERM group"""
    bl_idname = "perm.select_group"
    bl_label = "Select Group"
    bl_options = {'REGISTER', 'UNDO'}
    group_name: bpy.props.StringProperty(default="PERM_Supports")
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH': return {'CANCELLED'}
        if self.group_name not in obj.vertex_groups: return {'CANCELLED'}
        vg_idx = obj.vertex_groups[self.group_name].index
        if obj.mode != 'EDIT': bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg_idx: v.select = True; break
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED ASSIGN / SELECT / DESELECT / REMOVE OPERATORS
# The operators behind the Restraints and Loads sections of the panel.
# Each section has 4 operations: Assign (with selection), Select,
# Deselect, Remove. They all operate on the currently selected vertices.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# NODE LIST DISPLAY (toggle expansion + draw helper)
# Each BC/Load assignment row has an expand button. When expanded, the panel
# shows the assigned vertex indices (first 10 then "...and N more").
# State is stored in a scene-level comma-separated string of expanded groups.
# ─────────────────────────────────────────────────────────────────────────────

class PERM_OT_ToggleNodeList(bpy.types.Operator):
    """PERM: Toggle the expanded node list for a group."""
    bl_idname = "perm.toggle_node_list"
    bl_label = "PERM: Toggle Node List"
    bl_options = {'INTERNAL'}
    
    group_name: bpy.props.StringProperty(default="")
    
    def execute(self, context):
        scn = context.scene
        current = getattr(scn, 'perm_expanded_groups', '')
        expanded = set(s for s in current.split(',') if s)
        if self.group_name in expanded:
            expanded.discard(self.group_name)
        else:
            expanded.add(self.group_name)
        scn.perm_expanded_groups = ','.join(sorted(expanded))
        return {'FINISHED'}


def _is_group_expanded(scn, group_name):
    """Check if a group's node list is currently expanded."""
    current = getattr(scn, 'perm_expanded_groups', '')
    return group_name in current.split(',')


def _draw_node_list_for_group(box, obj, group_name, scn, max_shown=10):
    """Draw the expand button + (if expanded) the vertex index list for a group.
    
    If the group has more than max_shown nodes, show first max_shown + "...and N more".
    """
    is_expanded = _is_group_expanded(scn, group_name)
    # The expand toggle row
    row = box.row(align=True)
    icon = 'TRIA_DOWN' if is_expanded else 'TRIA_RIGHT'
    op = row.operator("perm.toggle_node_list", text="Nodes" if not is_expanded else "Hide nodes",
                      icon=icon, emboss=False)
    op.group_name = group_name
    
    if not is_expanded:
        return
    
    # Get vertex indices (sorted for stable display)
    verts = sorted(get_group_vertex_indices(obj, group_name))
    if not verts:
        box.label(text="    (empty group)", icon='ERROR')
        return
    
    sub = box.box()
    sub.scale_y = 0.7  # compact display
    
    N = len(verts)
    shown = verts[:max_shown]
    # Display in groups of 5 per line
    text_lines = []
    line = []
    for v in shown:
        line.append(str(v))
        if len(line) == 5:
            text_lines.append(", ".join(line))
            line = []
    if line:
        text_lines.append(", ".join(line))
    for tl in text_lines:
        sub.label(text=tl)
    if N > max_shown:
        sub.label(text=f"...and {N - max_shown} more (total: {N} nodes)", icon='INFO')
    else:
        sub.label(text=f"(total: {N} nodes)", icon='INFO')


class PERM_OT_AssignBC(bpy.types.Operator):
    """PERM: Assign selected vertices as a new BC/restraint group.
    
    Combines selection and assignment in one step. Pick the BC type
    in the dialog. The current selection becomes the BC's nodes.
    """
    bl_idname = "perm.assign_bc"
    bl_label = "PERM: Assign BC"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_name: bpy.props.StringProperty(name="Name", default="")
    bc_type: bpy.props.EnumProperty(
        name="BC Type",
        items=[
            ('FIXED', "FIXED (all DOFs)", "Constrain all 6 DOFs"),
            ('PINNED', "PINNED (no rotation)", "Constrain ux, uy, uz"),
            ('ROLLER_X', "ROLLER X (free in X)", "Constrain uy, uz only"),
            ('ROLLER_Y', "ROLLER Y (free in Y)", "Constrain ux, uz only"),
            ('ROLLER_Z', "ROLLER Z (free in Z)", "Constrain ux, uy only"),
            ('CUSTOM', "CUSTOM (choose the DOFs)", "Constrain any set of the six degrees of freedom, e.g. a "
                                                   "symmetry plane with normal Y: uy, rx, rz"),
        ],
        default='PINNED')
    # the DOFs of a CUSTOM restraint (global axes)
    fix_ux: bpy.props.BoolProperty(name="ux", default=False, description="Fix the translation along global X")
    fix_uy: bpy.props.BoolProperty(name="uy", default=False, description="Fix the translation along global Y")
    fix_uz: bpy.props.BoolProperty(name="uz", default=True, description="Fix the translation along global Z")
    fix_rx: bpy.props.BoolProperty(name="rx", default=False, description="Fix the rotation about global X")
    fix_ry: bpy.props.BoolProperty(name="ry", default=False, description="Fix the rotation about global Y")
    fix_rz: bpy.props.BoolProperty(name="rz", default=False, description="Fix the rotation about global Z")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "group_name")
        layout.prop(self, "bc_type")
        if self.bc_type == 'CUSTOM':
            layout.label(text="Fixed degrees of freedom (global axes):")
            row = layout.row(align=True)
            for d in ("ux", "uy", "uz"):
                row.prop(self, "fix_" + d, toggle=True)
            row = layout.row(align=True)
            for d in ("rx", "ry", "rz"):
                row.prop(self, "fix_" + d, toggle=True)

    def invoke(self, context, event):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh first"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select vertices first")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        n_sel = sum(1 for v in obj.data.vertices if v.select)
        bpy.ops.object.mode_set(mode='EDIT')
        if n_sel == 0:
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        # Auto-name
        existing = {vg.name for vg in obj.vertex_groups if vg.name.startswith("PERM_Support_")}
        i = 1
        while f"BC_{i}" in [n[len('PERM_Support_'):] for n in existing]:
            i += 1
        self.group_name = f"BC_{i}"
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh first"); return {'CANCELLED'}
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        _dofs = [d for d in PERM_DOF_NAMES if getattr(self, "fix_" + d)]
        if self.bc_type == 'CUSTOM' and not _dofs:
            self.report({'ERROR'}, "CUSTOM restraint: tick at least one degree of freedom")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        selected = [v.index for v in obj.data.vertices if v.select]
        if not selected:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        # Sanitize name
        name = self.group_name.strip()
        if not name:
            name = f"BC_{len(selected)}"
        group_name = f"PERM_Support_{name}"
        if group_name in obj.vertex_groups:
            # Reuse existing if user provided same name
            vg = obj.vertex_groups[group_name]
        else:
            vg = obj.vertex_groups.new(name=group_name)
        vg.add(selected, 1.0, 'REPLACE')
        # Store BC type as custom property
        obj[f"perm_support_{name}_type"] = self.bc_type
        if self.bc_type == 'CUSTOM':
            obj[f"perm_support_{name}_dofs"] = ",".join(_dofs)
        elif f"perm_support_{name}_dofs" in obj.keys():
            del obj[f"perm_support_{name}_dofs"]
        bpy.ops.object.mode_set(mode='EDIT')
        # Auto-refresh markers
        bpy.ops.object.mode_set(mode='OBJECT')
        try: create_marker_objects(obj)
        except Exception as e: print(f"[PERM] Marker refresh failed: {e}")
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'},
            f"PERM: BC '{name}' ({self.bc_type}) → {len(selected)} vertices")
        return {'FINISHED'}


class PERM_OT_AssignLoadPerNode(bpy.types.Operator):
    """PERM: Assign selected vertices as a new load group.
    
    Default convention: per-node (each selected vertex receives the entered Fx/Fy/Fz).
    NOT divided across the selection. To make different nodes carry different forces,
    you can edit individual values after creation via the per-node editor.
    """
    bl_idname = "perm.assign_load_per_node"
    bl_label = "PERM: Assign Load (per-node)"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_name: bpy.props.StringProperty(name="Name", default="")
    fx: bpy.props.FloatProperty(name="Fx (N) per node", default=0.0)
    fy: bpy.props.FloatProperty(name="Fy (N) per node", default=0.0)
    fz: bpy.props.FloatProperty(name="Fz (N) per node", default=-1000.0)
    
    def invoke(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh first"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select vertices first")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        n_sel = sum(1 for v in obj.data.vertices if v.select)
        bpy.ops.object.mode_set(mode='EDIT')
        if n_sel == 0:
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        existing = {vg.name for vg in obj.vertex_groups if vg.name.startswith("PERM_Load_")}
        i = 1
        while f"Load_{i}" in [n[len('PERM_Load_'):] for n in existing]:
            i += 1
        self.group_name = f"Load_{i}"
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        selected = [v.index for v in obj.data.vertices if v.select]
        if not selected:
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        name = self.group_name.strip()
        if not name:
            name = f"Load_{len(selected)}"
        group_name = f"PERM_Load_{name}"
        if group_name in obj.vertex_groups:
            vg = obj.vertex_groups[group_name]
        else:
            vg = obj.vertex_groups.new(name=group_name)
        vg.add(selected, 1.0, 'REPLACE')
        # Store as TOTAL (sum across all selected nodes) so the legacy display
        # shows a sensible value, but also store explicit per-node JSON so
        # the solver applies fx/fy/fz at EACH node.
        N = len(selected)
        total_fx = self.fx * N
        total_fy = self.fy * N
        total_fz = self.fz * N
        obj[f"perm_load_{name}_fx"] = total_fx
        obj[f"perm_load_{name}_fy"] = total_fy
        obj[f"perm_load_{name}_fz"] = total_fz
        # Build per-node dict: each selected vertex gets the same fx/fy/fz
        per_node = {vidx: (self.fx, self.fy, self.fz) for vidx in selected}
        set_per_node_forces(obj, name, per_node)
        # Print WHERE the load acts (world coordinates + position on the span)
        # for hand calculations; also written to Text Editor -> PERM_Loads.txt
        where = ""
        try:
            info = _perm_report_object_loads(obj, only=name).get(name)
            if info:
                p = info['at']
                where = f" at x={p[0]:+.3f}, y={p[1]:+.3f}, z={p[2]:+.3f} m"
                if 'a' in info:
                    where += f" (a/L={info['a'] / info['L']:.3f}, e={info['e']:+.3f} m)"
        except Exception as e:
            print(f"[PERM Loads] coordinate report failed: {e}")
        bpy.ops.object.mode_set(mode='EDIT')
        # Auto-refresh markers
        bpy.ops.object.mode_set(mode='OBJECT')
        try: create_marker_objects(obj)
        except Exception as e: print(f"[PERM] Marker refresh failed: {e}")
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'},
            f"PERM: Load '{name}' → {N} verts, "
            f"Fz={self.fz:.1f} N/node = {total_fz:.1f} N total{where}")
        return {'FINISHED'}


class PERM_OT_ReportLoads(bpy.types.Operator):
    """PERM: Print where every point load acts -- world coordinates of each
    loaded node, the resultant point, its distance along the span from the
    supports (a, b, a/L) and its sideways offset e -- and write the table to
    Text Editor -> PERM_Loads.txt (copy/paste it for hand calculations)"""
    bl_idname = "perm.report_loads"
    bl_label = "PERM: Show Load Coordinates"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = _perm_resolve_source(context)
        if obj is None:
            self.report({'ERROR'}, "Select the model mesh first"); return {'CANCELLED'}
        was_edit = obj.mode == 'EDIT'
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            info = _perm_report_object_loads(obj)
        finally:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
        if not info:
            self.report({'WARNING'}, "No point loads on this model"); return {'CANCELLED'}
        msg = "; ".join(f"{k}: x={v['at'][0]:+.3f} y={v['at'][1]:+.3f} z={v['at'][2]:+.3f}"
                        + (f" (a/L={v['a'] / v['L']:.3f})" if 'a' in v else "") for k, v in info.items())
        self.report({'INFO'}, f"PERM loads: {msg}  -- full table in Text Editor: PERM_Loads.txt")
        return {'FINISHED'}


class PERM_OT_SelectAssignment(bpy.types.Operator):
    """PERM: Select the vertices of a BC or load group in edit mode."""
    bl_idname = "perm.select_assignment"
    bl_label = "PERM: Select Group"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_name: bpy.props.StringProperty(default="")
    
    def execute(self, context):
        obj = context.active_object
        if not obj or self.group_name not in obj.vertex_groups:
            return {'CANCELLED'}
        vg = obj.vertex_groups[self.group_name]
        # Enter edit mode and deselect all, then select group's verts
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    v.select = True
                    break
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        self.report({'INFO'}, f"PERM: Selected '{self.group_name}' vertices")
        return {'FINISHED'}


class PERM_OT_DeselectAssignment(bpy.types.Operator):
    """PERM: Deselect the vertices of a BC or load group in edit mode."""
    bl_idname = "perm.deselect_assignment"
    bl_label = "PERM: Deselect Group"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_name: bpy.props.StringProperty(default="")
    
    def execute(self, context):
        obj = context.active_object
        if not obj or self.group_name not in obj.vertex_groups:
            return {'CANCELLED'}
        vg = obj.vertex_groups[self.group_name]
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    v.select = False
                    break
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"PERM: Deselected '{self.group_name}' vertices")
        return {'FINISHED'}


class PERM_OT_RemoveAssignment(bpy.types.Operator):
    """PERM: Remove an entire BC or load group (and its markers)."""
    bl_idname = "perm.remove_assignment"
    bl_label = "PERM: Remove Group"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_name: bpy.props.StringProperty(default="")
    
    def execute(self, context):
        obj = context.active_object
        if not obj or self.group_name not in obj.vertex_groups:
            return {'CANCELLED'}
        # Use the existing cleanup logic
        _cleanup_group_props(obj, self.group_name)
        obj.vertex_groups.remove(obj.vertex_groups[self.group_name])
        # Auto-refresh markers
        try: create_marker_objects(obj)
        except Exception as e: print(f"[PERM] Marker refresh failed: {e}")
        self.report({'INFO'}, f"PERM: Removed '{self.group_name}'")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# MATERIALS & SECTIONS — PropertyGroups + Operators
# ─────────────────────────────────────────────────────────────────────────────
# Materials and sections are stored as Scene-level CollectionProperty lists.
# At solve time, the operator builds the converter's materials/sections dicts
# from these scene-level lists.

class PERM_MaterialPG(bpy.types.PropertyGroup):
    """A named material definition (mechanical properties only).
    
    Following Abaqus/ANSYS/SAP2000 convention: material carries E, ν, ρ.
    Geometric properties (thickness for shells, area for trusses) belong
    in the cross-section, not the material.
    """
    name: bpy.props.StringProperty(name="Name", default="Concrete")
    youngs_modulus: bpy.props.FloatProperty(
        name="E (Pa)", default=30e9, min=1e6,
        description="Young's modulus")
    poissons_ratio: bpy.props.FloatProperty(
        name="ν", default=0.2, min=0.0, max=0.499,
        description="Poisson's ratio")
    density: bpy.props.FloatProperty(
        name="ρ (kg/m³)", default=2400.0, min=0.0,
        description="Density")
    # The thickness belongs to the cross-section (PERM_SectionPG), not to the material.


class PERM_SectionPG(bpy.types.PropertyGroup):
    """A named cross-section definition.
    For shells: thickness is the relevant field (area, Ix, Iy, J ignored).
    For trusses: area is the relevant field (thickness, Ix, Iy, J ignored).
    For future beams: area, Ix, Iy, J all matter; thickness ignored.
    """
    name: bpy.props.StringProperty(name="Name", default="default")
    thickness: bpy.props.FloatProperty(
        name="Shell t (m)", default=0.3, min=1e-6,
        description="Thickness — used when this section is assigned to shell faces")
    area: bpy.props.FloatProperty(
        name="A (m²)", default=1e-3, min=1e-12,
        description="Cross-sectional area — used for trusses and beams")
    Ix: bpy.props.FloatProperty(
        name="Ix (m⁴)", default=1e-8, min=0.0,
        description="Not used (kept so files from older versions open unchanged)")
    Iy: bpy.props.FloatProperty(
        name="Iy (m⁴)", default=1e-8, min=0.0,
        description="Beams: second moment about local y, i.e. vertical bending of a horizontal member "
                    "(local z points up)")
    J:  bpy.props.FloatProperty(
        name="J (m⁴)",  default=1e-8, min=0.0,
        description="Beams: torsion constant")
    # ── beam sections ──────────────────────────────────────────────────
    Iz: bpy.props.FloatProperty(
        name="Iz (m⁴)", default=1e-8, min=0.0,
        description="Beams: second moment about local z, i.e. bending in the horizontal plane of a "
                    "horizontal member")
    beam_shape: bpy.props.EnumProperty(
        name="Beam shape",
        items=[('GENERIC', "Generic (type A, Iy, Iz, J)", "Use the typed section properties"),
               ('RECT', "Rectangle b × h", "Solid rectangle: b across (local y), h deep (local z)"),
               ('RHS', "Rectangular tube b × h × t", "Hollow rectangle with wall thickness t"),
               ('CIRC', "Circle d", "Solid round bar"),
               ('CHS', "Circular tube d × t", "Pipe with wall thickness t")],
        default='GENERIC',
        description="Beams: section shape; the properties are computed from its dimensions")
    dim_b: bpy.props.FloatProperty(name="b (m)", default=0.10, min=0.0, precision=4, unit='LENGTH',
                                   description="Width across the member (local y)")
    dim_h: bpy.props.FloatProperty(name="h (m)", default=0.20, min=0.0, precision=4, unit='LENGTH',
                                   description="Depth (local z, up for a horizontal member)")
    dim_t: bpy.props.FloatProperty(name="t (m)", default=0.005, min=0.0, precision=4, unit='LENGTH',
                                   description="Wall thickness of a tube")
    dim_d: bpy.props.FloatProperty(name="d (m)", default=0.05, min=0.0, precision=4, unit='LENGTH',
                                   description="Outer diameter")
    shear_factor: bpy.props.FloatProperty(
        name="Shear factor κ", default=5.0 / 6.0, min=0.0, max=1.0,
        description="Beams (generic section): shear area = κ A in both directions; 0 = no shear "
                    "deformation (Euler-Bernoulli)")
    fibre_y: bpy.props.FloatProperty(
        name="c_y (m)", default=0.0, min=0.0, precision=4,
        description="Beams (generic section): distance from the centroid to the extreme fibre along "
                    "local y, for the stresses (0 = forces only)")
    fibre_z: bpy.props.FloatProperty(
        name="c_z (m)", default=0.0, min=0.0, precision=4,
        description="Beams (generic section): distance to the extreme fibre along local z")
    roll_deg: bpy.props.FloatProperty(
        name="Roll (deg)", default=0.0, soft_min=-180.0, soft_max=180.0,
        description="Beams: turn the section about the member axis (0 = local z up)")
    # ── Reinforced concrete (linear transformed section) ────────────────────
    # Leave ρ steel = 0 for plain concrete / non-RC shells (no effect at all).
    # These describe smeared reinforcement in a shell; the solver folds them
    # into transformed membrane + bending stiffness (see _rebar_factors).
    rebar_ratio: bpy.props.FloatProperty(
        name="ρ steel (A_s/A_c)", default=0.0, min=0.0, max=0.2,
        description="Total reinforcement ratio A_s/(b·t) summed over top+bottom "
                    "mats. 0 = plain concrete (rebar has no effect).")
    rebar_cover: bpy.props.FloatProperty(
        name="c: face → bar centre (m)", default=0.0, min=0.0,
        description="NOT the clear cover: distance from the shell face to the CENTRE "
                    "of the bars, through the shell thickness = clear cover + stirrup + "
                    "bar Ø/2 (each "
                    "mat). Drives the bending-stiffness boost; unused if ρ=0.")
    steel_modulus: bpy.props.FloatProperty(
        name="E_s (Pa)", default=200e9, min=0.0,
        description="Reinforcement Young's modulus (steel ≈ 200e9 Pa). "
                    "Only used when ρ steel > 0.")


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT TYPES — unified shell + truss element-type management
# ─────────────────────────────────────────────────────────────────────────────
# Each element type entry has a name and a kind. Kinds:
#   - QUAD4_MITC4: 4-node bending shell (default for quad faces)
#   - QUAD4_MEMBRANE: 4-node membrane-only quad (no bending)
#   - QUAD4_PLATE: 4-node MITC4 plate, no membrane (transverse load only)
#   - TRI3: 3-node shell triangle (CST + MITC3)
#   - TRI6: 6-node membrane triangle (not yet implemented in solver)
#   - TRUSS_T2: 2-node truss bar
# 
# Assignment storage:
#   - Shell kinds: vertex group `PERM_Element_<typename>` containing the
#     selected faces' vertices. A face's element type is determined by
#     which PERM_Element_<typename> all its vertices belong to.
#   - TRUSS_T2: numbered vertex groups `PERM_Element_<typename>_t<n>` each
#     containing exactly 2 vertices (one truss instance).
#
# When solver runs, it inspects each face's verts and finds the matching
# element type group. Defaults: QUAD4_MITC4 for quad faces, TRI3 for triangles.


def _element_kind_items(self, context):
    """Enum items for element kind selector."""
    return [
        ('QUAD4_MITC4',
         "Bending shell (QUAD4)",
         "Full 4-node shell: QM6 membrane + MITC4 bending & transverse shear + "
         "Hughes-Brezzi drilling. The default, and the validated element"),
        ('QUAD4_MEMBRANE',
         "Membrane shell (QUAD4)",
         "4-node quadrilateral, QM6 membrane only — NO bending stiffness. "
         "Acts as a hinge out of plane; use only where that is intended"),
        ('QUAD4_PLATE',
         "Plate, MITC4 only (QUAD4)",
         "MITC4 bending & transverse shear + drilling, NO membrane — the exact "
         "complement of Membrane shell. Carries transverse load only; has no "
         "in-plane stiffness (5 zero-energy modes per element). Valid for a "
         "FLAT plate; wrong for folded structures, whose bending is carried by "
         "membrane action"),
        ('TRI3',
         "Triangle shell (TRI3)",
         "3-node shell: CST membrane + MITC3 bending & transverse shear + "
         "Hughes-Brezzi drilling"),
        ('TRUSS_T2',
         "Truss bar (T2)",
         "2-node axial bar: one per selected mesh edge (two selected vertices with no edge between them "
         "give one bar). Uses the active material and the area of the active cross-section"),
        ('BEAM_B2',
         "Beam (B2, 3D frame)",
         "2-node 3D Timoshenko beam on mesh edges: axial force, two shears, torsion and bending about "
         "two axes. Section: a shape (rectangle, tube, ...) or A, Iy, Iz and J"),
    ]


class PERM_ElementTypePG(bpy.types.PropertyGroup):
    """A named element type definition that can be assigned to faces or vertex pairs."""
    name: bpy.props.StringProperty(name="Name", default="ShellBending")
    kind: bpy.props.EnumProperty(
        name="Kind",
        items=_element_kind_items,
        default=0)
    # For TRUSS_T2 only: track how many truss instances are using this type
    # (informational only; actual instances live in PERM_Element_<n>_t<i> groups)
    truss_count: bpy.props.IntProperty(default=0)


class PERM_UL_ElementTypes(bpy.types.UIList):
    """List view for element types."""
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            # Show different icons per kind
            kind_icon = {
                'QUAD4_MITC4': 'MESH_PLANE',
                'QUAD4_MEMBRANE': 'OUTLINER_OB_MESH',
                'QUAD4_PLATE': 'MOD_LATTICE',
                'TRI3': 'MESH_DATA',
                'TRUSS_T2': 'IPO_LINEAR',
                'BEAM_B2': 'MOD_WIREFRAME',
            }.get(item.kind, 'MESH_DATA')
            row.label(text=item.name, icon=kind_icon)
            # Human-readable kind label
            kind_label = {
                'QUAD4_MITC4': "QUAD4+MITC4",
                'QUAD4_MEMBRANE': "QUAD4 membrane",
                'QUAD4_PLATE': "QUAD4 plate (MITC4)",
                'TRI3': "TRI3",
                'TRUSS_T2': "TRUSS_T2",
                'BEAM_B2': "BEAM_B2",
            }.get(item.kind, item.kind)
            row.label(text=kind_label)


class PERM_OT_AddElementType(bpy.types.Operator):
    """PERM: Add a new element type."""
    bl_idname = "perm.add_element_type"
    bl_label = "PERM: Add Element Type"
    bl_options = {'REGISTER', 'UNDO'}
    
    name: bpy.props.StringProperty(name="Name", default="NewElementType")
    kind: bpy.props.EnumProperty(name="Kind", items=_element_kind_items, default=0)
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def execute(self, context):
        scn = context.scene
        # Reject duplicate names
        for et in scn.perm_element_types:
            if et.name == self.name.strip():
                self.report({'ERROR'},
                    f"Element type '{self.name}' already exists")
                return {'CANCELLED'}
        item = scn.perm_element_types.add()
        item.name = self.name.strip() or f"ElementType_{len(scn.perm_element_types)}"
        item.kind = self.kind
        scn.perm_element_type_index = len(scn.perm_element_types) - 1
        self.report({'INFO'}, f"PERM: Added element type '{item.name}' ({self.kind})")
        return {'FINISHED'}


class PERM_OT_DeleteElementType(bpy.types.Operator):
    """PERM: Delete the active element type."""
    bl_idname = "perm.delete_element_type"
    bl_label = "PERM: Delete Element Type"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scn = context.scene
        obj = context.active_object
        idx = scn.perm_element_type_index
        if idx < 0 or idx >= len(scn.perm_element_types):
            return {'CANCELLED'}
        et = scn.perm_element_types[idx]
        name = et.name
        # Remove associated vertex groups, face tags and truss edges. Names are
        # matched EXACTLY: a prefix test on "Web" also hit "Web_thin".
        if obj and obj.type == 'MESH':
            to_remove = [vg.name for vg in obj.vertex_groups
                         if vg.name == f"PERM_Element_{name}"
                         or _is_type_truss_group(vg.name, name)]
            for vg_name in to_remove:
                obj.vertex_groups.remove(obj.vertex_groups[vg_name])
            _clear_faces_tag(obj, 'elemtype', _face_tag_id(obj, 'elemtype', name, create=False))
            _clear_edges_truss(obj, _truss_ids_of_type(obj, name))
        scn.perm_element_types.remove(idx)
        if scn.perm_element_type_index >= len(scn.perm_element_types):
            scn.perm_element_type_index = max(0, len(scn.perm_element_types) - 1)
        self.report({'INFO'}, f"PERM: Removed element type '{name}'")
        return {'FINISHED'}


class PERM_OT_AssignElementType(bpy.types.Operator):
    """PERM: Assign the active element type to current selection.
    
    For shell kinds (QUAD4_MITC4, QUAD4_MEMBRANE, QUAD4_PLATE, TRI3): selected faces' vertices
    are added to a single group PERM_Element_<typename>. Validates that selected
    faces' topology matches the element kind (e.g. QUAD4 needs quad faces).
    
    For TRUSS_T2: requires exactly 2 selected vertices. Creates a NEW numbered
    group PERM_Element_<typename>_t<n> for each truss instance.
    """
    bl_idname = "perm.assign_element_type"
    bl_label = "PERM: Assign Element Type"
    bl_options = {'REGISTER', 'UNDO'}

    replace_existing: bpy.props.BoolProperty(
        name="Replace existing assignment", default=True,
        description="Move faces that already carry a different element type over "
                    "to this one. Turn off to skip them instead")

    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        scn = context.scene
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and make a selection")
            return {'CANCELLED'}
        if scn.perm_element_type_index < 0 or scn.perm_element_type_index >= len(scn.perm_element_types):
            self.report({'ERROR'}, "Pick an element type in the Element Types list first")
            return {'CANCELLED'}
        et = scn.perm_element_types[scn.perm_element_type_index]
        kind = et.kind
        name = et.name.strip()
        if not name:
            self.report({'ERROR'}, "Active element type has empty name"); return {'CANCELLED'}
        
        # ─── TRUSS_T2 branch ─────────────────────────────────────────────
        # One bar per selected mesh EDGE, carrying the ACTIVE material and the
        # ACTIVE cross-section (both by name; area read at solve time).
        # Exactly two selected vertices with no edge between them still create a
        # single legacy bar (vertex-group instance) -- e.g. diagonal bracing.
        if kind in _PERM_LINE_TYPES:
            _bar = "beam" if kind == 'BEAM_B2' else "truss bar"
            if not scn.perm_sections or not (0 <= scn.perm_section_index < len(scn.perm_sections)):
                self.report({'ERROR'},
                    f"Pick a Cross-Section first - a {_bar} needs one")
                return {'CANCELLED'}
            if not scn.perm_materials or not (0 <= scn.perm_material_index < len(scn.perm_materials)):
                self.report({'ERROR'},
                    "Pick a Material (e.g. Steel) first - each truss bar keeps its own material")
                return {'CANCELLED'}
            section = scn.perm_sections[scn.perm_section_index]
            section_name = section.name.strip()
            area = float(section.area)
            mat_name = scn.perm_materials[scn.perm_material_index].name.strip()
            if kind == 'BEAM_B2':
                _bp = _perm_beam_props(section)
                if min(_bp['A'], _bp['Iy'], _bp['Iz'], _bp['J']) <= 0.0:
                    self.report({'ERROR'}, f"Section '{section_name}' has no beam properties: pick a beam shape "
                                "(rectangle, tube, ...) or give A, Iy, Iz and J")
                    return {'CANCELLED'}
                area = _bp['A']

            bpy.ops.object.mode_set(mode='OBJECT')
            sel_edges = [e.index for e in obj.data.edges if e.select]
            selected = [v.index for v in obj.data.vertices if v.select]
            bpy.ops.object.mode_set(mode='EDIT')

            if sel_edges:
                cid = _truss_combo_id(obj, name, mat_name, section_name)
                n, n_over = _set_selected_edges_truss(obj, cid)
                et.truss_count = _count_truss_edges(obj, name)
                msg = (f"PERM: {n} {_bar}(s) '{name}' on selected edges - "
                       f"material '{mat_name}', section '{section_name}' (A={area:.3e} m²)")
                if n_over:
                    msg += f"; {n_over} edge(s) re-assigned from another truss"
                self.report({'INFO'}, msg)
                return {'FINISHED'}

            if len(selected) != 2:
                self.report({'ERROR'},
                    f"{_bar}: select mesh EDGES (one per edge), or exactly 2 "
                    f"vertices for a bar between unconnected vertices (got {len(selected)} verts)")
                return {'CANCELLED'}
            # Legacy single bar between two vertices that share no edge
            existing = [vg.name for vg in obj.vertex_groups
                        if _is_type_truss_group(vg.name, name)]
            next_idx = len(existing) + 1
            while f"PERM_Element_{name}_t{next_idx}" in [vg.name for vg in obj.vertex_groups]:
                next_idx += 1
            group_name = f"PERM_Element_{name}_t{next_idx}"
            bpy.ops.object.mode_set(mode='OBJECT')
            vg = obj.vertex_groups.new(name=group_name)
            vg.add(selected, 1.0, 'REPLACE')
            obj[f"perm_truss_{name}_t{next_idx}_area"] = area            # fallback only
            obj[f"perm_truss_{name}_t{next_idx}_section"] = section_name  # authoritative
            obj[f"perm_truss_{name}_t{next_idx}_material"] = mat_name
            c1 = obj.data.vertices[selected[0]].co
            c2 = obj.data.vertices[selected[1]].co
            L = (c2 - c1).length
            try: create_marker_objects(obj)
            except Exception as e: print(f"[PERM] Marker refresh failed: {e}")
            bpy.ops.object.mode_set(mode='EDIT')
            et.truss_count = _count_truss_edges(obj, name) + len(existing) + 1
            self.report({'INFO'},
                f"PERM: Created truss '{group_name}' (L={L:.3f} m) - material "
                f"'{mat_name}', section '{section_name}' (A={area:.3e} m²)")
            return {'FINISHED'}
        
        # ─── SHELL branch (QUAD4_MITC4, QUAD4_MEMBRANE, QUAD4_PLATE, TRI3) ─────
        # Filter and skip:
        #   1. Take all selected faces
        #   2. Classify each face: matches kind / wrong topology / already assigned to other type
        #   3. Assign only the matching unassigned faces; report skipped counts
        # This lets the user "select all, assign QUAD4" then "select all, assign TRI3"
        # without getting refused.
        bpy.ops.object.mode_set(mode='OBJECT')
        selected_faces = [p for p in obj.data.polygons if p.select]
        if not selected_faces:
            # Fall back: infer from vertex selection
            sel_verts = {v.index for v in obj.data.vertices if v.select}
            if not sel_verts:
                bpy.ops.object.mode_set(mode='EDIT')
                self.report({'ERROR'}, "Select faces (or vertices) first")
                return {'CANCELLED'}
            for p in obj.data.polygons:
                if all(vi in sel_verts for vi in p.vertices):
                    selected_faces.append(p)
            if not selected_faces:
                bpy.ops.object.mode_set(mode='EDIT')
                self.report({'ERROR'},
                    "No complete faces in selection (need to select all 3-4 verts of a face)")
                return {'CANCELLED'}
        
        # Build a map: vertex_index → set of shell element-type group names it belongs to
        # (so we can detect "already assigned" faces).
        # We only consider shell groups (PERM_Element_<name> with no _t suffix).
        my_group_name = f"PERM_Element_{name}"
        other_shell_groups = []
        for vg in obj.vertex_groups:
            if (vg.name.startswith("PERM_Element_")
                and not _is_truss_instance_group(vg.name[len("PERM_Element_"):])
                and vg.name != my_group_name):
                other_shell_groups.append(vg)
        
        # Compute the set of vertices currently in OTHER shell groups (= "already assigned")
        already_assigned_verts = set()
        for vg in other_shell_groups:
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index:
                        already_assigned_verts.add(v.index)
                        break
        
        # Classify each selected face
        # Every quad KIND must be listed here. A kind missing from this tuple
        # makes face_topology_matches return False for all four-vertex faces,
        # so the operator reports "wrong topology" for the entire mesh and
        # assigns nothing. Keep it in step with _element_kind_items.
        QUAD_KINDS = ('QUAD4_MITC4', 'QUAD4_MEMBRANE', 'QUAD4_PLATE')
        TRI_KINDS = ('TRI3',)

        def face_topology_matches(face, kind):
            n = len(face.vertices)
            if kind in QUAD_KINDS:
                return n == 4
            elif kind in TRI_KINDS:
                return n == 3
            return False
        
        # Face tags are authoritative once the mesh has them: a face is
        # "already assigned" when it carries ANOTHER element type's tag.
        my_et_id = _face_tag_id(obj, 'elemtype', name)
        et_tags = _read_face_tags(obj.data, 'elemtype')

        def face_is_already_assigned(face):
            if et_tags is not None:
                return int(et_tags[face.index]) not in (0, my_et_id)
            # Legacy: ALL its verts are in some other shell group
            return all(vi in already_assigned_verts for vi in face.vertices)
        
        accepted_faces = []
        skipped_wrong_topology = []
        skipped_already_assigned = []
        reassigned_faces = []
        for p in selected_faces:
            # Topology is checked FIRST. A face of the wrong topology can never be
            # accepted, so reporting it as "already assigned" would name the wrong
            # reason and send the user looking in the wrong place.
            if not face_topology_matches(p, kind):
                skipped_wrong_topology.append(p.index)
                continue
            if face_is_already_assigned(p):
                if not self.replace_existing:
                    skipped_already_assigned.append(p.index)
                    continue
                reassigned_faces.append(p)
            accepted_faces.append(p)
        
        if not accepted_faces:
            bpy.ops.object.mode_set(mode='EDIT')
            wt = len(skipped_wrong_topology)
            aa = len(skipped_already_assigned)
            msg = f"PERM: No faces assigned to '{name}' ({kind})."
            if wt:
                want = "4 vertices (quad)" if kind in QUAD_KINDS else "3 vertices (triangle)"
                msg += (f" {wt} face(s) have the wrong topology - {kind} needs faces "
                        f"with {want}.")
            if aa:
                msg += (f" {aa} face(s) already carry another element type; enable "
                        f"'Replace existing assignment' (F9) to move them.")
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}
        
        # Collect vertices from accepted faces only
        verts_to_assign = set()
        for p in accepted_faces:
            for vi in p.vertices:
                verts_to_assign.add(vi)
        
        # Take the reassigned faces out of whichever other shell group held them.
        #
        # Vertex groups are per-VERTEX but a face counts as assigned only when ALL
        # of its vertices are in the group, so a vertex cannot simply be dropped:
        # it may still be needed by a neighbouring face that is staying where it
        # is. Remove a vertex from group G only when no face REMAINING in G uses
        # it -- otherwise reassigning one patch silently un-assigns the faces
        # around its border.
        n_moved_from = defaultdict(int)
        if reassigned_faces:
            reassigned_idx = {p.index for p in reassigned_faces}
            for vg_other in other_shell_groups:
                members = set()
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == vg_other.index:
                            members.add(v.index)
                            break
                if not members:
                    continue
                keep = set()
                for p in obj.data.polygons:
                    if p.index in reassigned_idx:
                        continue
                    if all(vi in members for vi in p.vertices):   # stays in G
                        keep.update(p.vertices)
                drop = {vi for p in reassigned_faces for vi in p.vertices
                        if vi in members and vi not in keep}
                if drop:
                    vg_other.remove(list(drop))
                    n_moved_from[vg_other.name[len("PERM_Element_"):]] += len(drop)

        # Add to this element type's group
        if my_group_name in obj.vertex_groups:
            vg = obj.vertex_groups[my_group_name]
        else:
            vg = obj.vertex_groups.new(name=my_group_name)
        vg.add(list(verts_to_assign), 1.0, 'REPLACE')
        # Authoritative per-face element type (see PER-FACE ... TAGS)
        _set_faces_tag_object_mode(obj, 'elemtype',
                                   [p.index for p in accepted_faces], my_et_id)
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Build report message
        report_parts = [f"Assigned {len(accepted_faces)} face(s) to '{name}' ({kind})"]
        if reassigned_faces:
            src = ", ".join(f"'{k}'" for k in sorted(n_moved_from)) or "another type"
            report_parts.append(f"{len(reassigned_faces)} moved from {src}")
        if skipped_wrong_topology:
            report_parts.append(f"skipped {len(skipped_wrong_topology)} (wrong topology)")
        if skipped_already_assigned:
            report_parts.append(f"skipped {len(skipped_already_assigned)} (already assigned)")
        self.report({'INFO'}, "PERM: " + "; ".join(report_parts))
        
        # Print a more detailed breakdown to console
        print(f"[PERM] Element type assignment '{name}' ({kind}):")
        print(f"  Accepted: {len(accepted_faces)} faces ({len(verts_to_assign)} verts)")
        for k, n in sorted(n_moved_from.items()):
            print(f"  Moved from '{k}': {n} vertices released")
        if skipped_wrong_topology:
            sample = skipped_wrong_topology[:5]
            more = "" if len(skipped_wrong_topology) <= 5 else f" (and {len(skipped_wrong_topology)-5} more)"
            print(f"  Skipped (wrong topology): {len(skipped_wrong_topology)} faces, e.g. {sample}{more}")
        if skipped_already_assigned:
            sample = skipped_already_assigned[:5]
            more = "" if len(skipped_already_assigned) <= 5 else f" (and {len(skipped_already_assigned)-5} more)"
            print(f"  Skipped (already assigned): {len(skipped_already_assigned)} faces, e.g. {sample}{more}")
        return {'FINISHED'}


class PERM_OT_SelectElementTypeFaces(bpy.types.Operator):
    """PERM: Select all faces/verts assigned to the active element type."""
    bl_idname = "perm.select_element_type_faces"
    bl_label = "PERM: Select Element Type's Faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_element_type_index < 0 or scn.perm_element_type_index >= len(scn.perm_element_types):
            return {'CANCELLED'}
        et = scn.perm_element_types[scn.perm_element_type_index]
        name = et.name
        # Shell types: face tags when present
        if et.kind not in _PERM_LINE_TYPES:
            tag = _face_tag_id(obj, 'elemtype', name, create=False)
            if tag and _select_faces_with_tag(obj, 'elemtype', tag):
                return {'FINISHED'}
        # Truss edges of this type
        truss_ids = _truss_ids_of_type(obj, name) if et.kind in _PERM_LINE_TYPES else set()
        # Legacy groups (exact names)
        matching = [vg for vg in obj.vertex_groups
                    if vg.name == f"PERM_Element_{name}"
                    or _is_type_truss_group(vg.name, name)]
        if not matching and not truss_ids:
            self.report({'WARNING'}, f"No assignments for '{name}' yet")
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        if truss_ids:
            tags = _read_edge_truss_tags(obj.data)
            if tags is not None:
                for e in obj.data.edges:
                    if int(tags[e.index]) in truss_ids:
                        e.select = True
                        for vi in e.vertices:
                            obj.data.vertices[vi].select = True
        for vg in matching:
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index:
                        v.select = True
                        break
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class PERM_OT_RemoveTrussFromEdges(bpy.types.Operator):
    """PERM: Remove truss bars (of any type) from the selected edges."""
    bl_idname = "perm.remove_truss_from_edges"
    bl_label = "PERM: Remove Trusses from Selected Edges"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select edges first")
            return {'CANCELLED'}
        n = _clear_edges_truss(obj, set(_truss_combos(obj).keys()), only_selected=True)
        for et in context.scene.perm_element_types:
            if et.kind in _PERM_LINE_TYPES:
                et.truss_count = _count_truss_edges(obj, et.name)
        self.report({'INFO'}, f"PERM: Removed {n} truss bar(s) from selected edges")
        return {'FINISHED'}


class PERM_OT_UnassignElementType(bpy.types.Operator):
    """Remove the active element type from ALL its faces and edges (faces become full bending shells again)"""
    bl_idname = "perm.unassign_element_type"
    bl_label = "PERM: Unassign Element Type"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_element_type_index < 0 or scn.perm_element_type_index >= len(scn.perm_element_types):
            return {'CANCELLED'}
        et = scn.perm_element_types[scn.perm_element_type_index]
        name = et.name
        to_remove = [vg.name for vg in obj.vertex_groups
                     if vg.name == f"PERM_Element_{name}"
                     or _is_type_truss_group(vg.name, name)]
        for vg_name in to_remove:
            obj.vertex_groups.remove(obj.vertex_groups[vg_name])
        # Remove custom props for legacy trusses (exact type name only)
        key_re = re.compile(r"perm_truss_" + re.escape(name) + r"_t\d+_")
        keys_to_del = [k for k in obj.keys() if key_re.match(k)]
        for k in keys_to_del:
            del obj[k]
        n_faces = _clear_faces_tag(obj, 'elemtype',
                                   _face_tag_id(obj, 'elemtype', name, create=False))
        n_edges = _clear_edges_truss(obj, _truss_ids_of_type(obj, name))
        if n_faces or n_edges:
            print(f"[PERM] Unassigned '{name}': {n_faces} face tag(s), {n_edges} truss edge(s)")
        et.truss_count = 0
        try: create_marker_objects(obj)
        except Exception as e: print(f"[PERM] Marker refresh failed: {e}")
        self.report({'INFO'},
            f"PERM: Unassigned all instances of '{name}' "
            f"({len(to_remove)} groups removed)")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL / SECTION ASSIGNMENT TO ELEMENTS (vertex-group based)
# Convention:
#   - PERM_Material_<name> vertex group: faces whose vertices are entirely in
#     this group use the named material (E, ν, ρ, etc.)
#   - PERM_Section_<name> vertex group: faces whose vertices are entirely in
#     this group use the named section (thickness for shells, area for trusses)
# At solve time, the operator reads these groups and builds per-element material
# and section assignments.
# ─────────────────────────────────────────────────────────────────────────────

def find_perm_material_groups(obj):
    """Return list of (suffix, group_name) for all PERM_Material_* groups."""
    out = []
    if not obj or obj.type != 'MESH': return out
    for vg in obj.vertex_groups:
        if vg.name.startswith("PERM_Material_"):
            out.append((vg.name[len("PERM_Material_"):], vg.name))
    return out


def find_perm_section_groups(obj):
    """Return list of (suffix, group_name) for all PERM_Section_* groups (shell sections)."""
    out = []
    if not obj or obj.type != 'MESH': return out
    for vg in obj.vertex_groups:
        if vg.name.startswith("PERM_Section_"):
            out.append((vg.name[len("PERM_Section_"):], vg.name))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-FACE MATERIAL / SECTION TAGS
# ─────────────────────────────────────────────────────────────────────────────
# Material and section assignments are stored on the FACE itself, as an integer
# Blender attribute (domain FACE). Vertex groups alone cannot do it: a face would
# be assigned only when ALL its vertices sat in one group, and the vertices on
# the line where two parts meet (deck/web, web/railing) can belong to only one
# group, so every element touching such a line would fall back to material #0
# and the default thickness.
#
# An integer id per object maps to the material / section NAME, so reordering
# the scene lists does not change results. The attribute survives the
# working-copy duplicate and the clean / quadrangulate steps (BMesh carries face
# custom data through them), and the converter's _element_to_poly map takes
# each element straight to its face.
#
# Vertex groups are still written, so older files and the rest of the UI keep
# working; where a face carries a tag, the tag wins.
PERM_FACE_ATTR = {'material': "perm_face_material", 'section': "perm_face_section",
                  'elemtype': "perm_face_elemtype"}
_PERM_FACE_IDS_PROP = {'material': "perm_face_material_ids",
                       'section': "perm_face_section_ids",
                       'elemtype': "perm_face_elemtype_ids"}


def _face_tag_table(obj, kind) -> Dict[str, int]:
    raw = obj.get(_PERM_FACE_IDS_PROP[kind])
    if raw is None:
        return {}
    raw = raw.to_dict() if hasattr(raw, 'to_dict') else dict(raw)
    return {str(k): int(v) for k, v in raw.items()}


def _face_tag_id(obj, kind, name, create=True) -> int:
    """Stable per-object integer id for a material/section name (0 = untagged)."""
    table = _face_tag_table(obj, kind)
    if name in table:
        return table[name]
    if not create:
        return 0
    new_id = max(table.values(), default=0) + 1
    table[name] = new_id
    obj[_PERM_FACE_IDS_PROP[kind]] = table
    return new_id


def _face_tag_names(obj, kind) -> Dict[int, str]:
    return {v: k for k, v in _face_tag_table(obj, kind).items()}


def _set_selected_faces_tag(obj, kind, value) -> int:
    """EDIT MODE: write `value` into the face attribute of every selected face."""
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    name = PERM_FACE_ATTR[kind]
    layer = bm.faces.layers.int.get(name) or bm.faces.layers.int.new(name)
    n = 0
    for f in bm.faces:
        if f.select:
            f[layer] = value
            n += 1
    bmesh.update_edit_mesh(me)
    return n


def _read_face_tags(mesh, kind):
    """OBJECT MODE: per-polygon tag array, or None if the mesh has no tags."""
    attr = mesh.attributes.get(PERM_FACE_ATTR[kind])
    if attr is None or attr.domain != 'FACE' or attr.data_type != 'INT':
        return None
    vals = np.zeros(len(mesh.polygons), dtype=np.int32)
    if len(vals):
        attr.data.foreach_get('value', vals)
    return vals


def _clear_faces_tag(obj, kind, value) -> int:
    """Set every face tagged `value` back to 0. Works from Object or Edit Mode."""
    was_edit = obj.mode == 'EDIT'
    if was_edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    n = 0
    attr = obj.data.attributes.get(PERM_FACE_ATTR[kind])
    if attr is not None and attr.domain == 'FACE' and value > 0:
        for d in attr.data:
            if d.value == value:
                d.value = 0
                n += 1
    if was_edit:
        bpy.ops.object.mode_set(mode='EDIT')
    return n


def _select_faces_with_tag(obj, kind, value) -> int:
    """Select (in Edit Mode) the faces tagged `value`. Returns the face count."""
    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    layer = bm.faces.layers.int.get(PERM_FACE_ATTR[kind])
    if layer is None or value <= 0:
        return 0
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    n = 0
    for f in bm.faces:
        if f[layer] == value:
            f.select_set(True)
            n += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(me)
    return n


def _has_face_tags(converter, kind) -> bool:
    """True when the mesh carries per-face tags of this kind. Once it does, the
    tags are the ONLY source for that kind: an untagged face is unassigned, so
    'Unassign' really reverts it, and stale vertex groups cannot leak back in."""
    wobj = getattr(converter, 'working_object', None)
    return wobj is not None and _read_face_tags(wobj.data, kind) is not None


def _element_face_tag_names(converter, kind) -> Dict[int, str]:
    """FE element id -> assigned material/section NAME from face tags.

    Reads the converter's working copy (the mesh the elements were built from),
    so polygon indices match _element_to_poly exactly.
    """
    wobj = getattr(converter, 'working_object', None)
    if wobj is None:
        return {}
    tags = _read_face_tags(wobj.data, kind)
    if tags is None:
        return {}
    names = _face_tag_names(converter.source_object, kind) or _face_tag_names(wobj, kind)
    out = {}
    for eid, poly in converter._element_to_poly.items():
        if 0 <= poly < len(tags):
            name = names.get(int(tags[poly]))
            if name:
                out[eid] = name
    return out


def _source_to_fe_maps(src_obj, converter, tol: float = 1.0e-3):
    """Map the SOURCE object's vertex indices onto FE nodes.

    Vertex groups (supports, loads, legacy material/section/element-type groups)
    are stored on the source object, but the FE model is built from the working
    copy AFTER clean_mesh(). When that merges duplicate vertices -- which it does
    whenever deck, webs and railings were modelled as separate pieces -- vertex
    indices shift, and looking source indices up in the working copy's
    _vert_to_node put supports and loads on unrelated nodes.

    Each source vertex is matched to the working-copy vertex at the same
    location (KD-tree, within `tol`, comfortably above the 1e-4 merge distance).

    Returns (src_v2n, src_n2v, src_n2vs):
        src_v2n   source vertex index -> FE node id
        src_n2v   FE node id -> one source vertex index (representative)
        src_n2vs  FE node id -> all source vertex indices merged into it
    """
    v2n_w = converter._vert_to_node
    wobj = getattr(converter, 'working_object', None)
    sme = src_obj.data
    src_v2n: Dict[int, int] = {}
    if wobj is None or wobj is src_obj:
        src_v2n = dict(v2n_w)
    else:
        wme = wobj.data
        n_s, n_w = len(sme.vertices), len(wme.vertices)
        same = False
        if n_s == n_w:
            a = np.zeros(3 * n_s); b = np.zeros(3 * n_w)
            sme.vertices.foreach_get('co', a); wme.vertices.foreach_get('co', b)
            same = bool(np.all(np.abs(a - b) <= tol))
        if same:
            src_v2n = dict(v2n_w)
        else:
            from mathutils import kdtree
            kd = kdtree.KDTree(n_w)
            for v in wme.vertices:
                kd.insert(v.co, v.index)
            kd.balance()
            for v in sme.vertices:
                co, idx, dist = kd.find(v.co)
                if idx is not None and dist <= tol and idx in v2n_w:
                    src_v2n[v.index] = v2n_w[idx]
            print(f"[PERM Solver] Source mesh had {n_s - n_w} duplicate/removed "
                  f"vertices; vertex groups remapped by position "
                  f"({len(src_v2n)}/{n_s} source vertices matched)")
    src_n2vs: Dict[int, List[int]] = defaultdict(list)
    for sv, nid in src_v2n.items():
        src_n2vs[nid].append(sv)
    src_n2v = {nid: vs[0] for nid, vs in src_n2vs.items()}
    return src_v2n, src_n2v, dict(src_n2vs)


# ─────────────────────────────────────────────────────────────────────────────
# TRUSS BARS ON MESH EDGES
# ─────────────────────────────────────────────────────────────────────────────
# A truss bar is a mesh EDGE carrying an integer Blender attribute (domain EDGE).
# The integer indexes a per-object table of (element type, material, section)
# NAMES, so each bar keeps its own material and section, independent of the
# faces around it, and the area is read from the Section at SOLVE time (editing
# the Section updates every bar). One bar per edge means a strand along a web is
# bonded at every node it passes, exactly like the shell mesh it lies on.
#
# The legacy 2-vertex workflow (vertex group PERM_Element_<type>_t<n>) is kept
# for bars between vertices that share no edge, e.g. diagonal bracing.
PERM_EDGE_TRUSS_ATTR = "perm_edge_truss"
_PERM_EDGE_TRUSS_IDS_PROP = "perm_edge_truss_ids"

# Legacy truss instance groups are named PERM_Element_<type>_t<number>. Match
# that EXACT suffix: a plain `"_t" in name` test also caught shell types such as
# "Web_thin" or "Deck_top" and treated them as trusses.
_TRUSS_GROUP_RE = re.compile(r"^(.*)_t(\d+)$")


def _is_truss_instance_group(rest: str) -> bool:
    """`rest` is a vertex-group name with the 'PERM_Element_' prefix removed."""
    return _TRUSS_GROUP_RE.match(rest) is not None


def _is_type_truss_group(vg_name: str, type_name: str) -> bool:
    """True for PERM_Element_<type_name>_t<n> exactly (not a longer type name)."""
    return re.fullmatch(r"PERM_Element_" + re.escape(type_name) + r"_t\d+",
                        vg_name) is not None


def _truss_combo_table(obj) -> Dict[str, int]:
    raw = obj.get(_PERM_EDGE_TRUSS_IDS_PROP)
    if raw is None:
        return {}
    raw = raw.to_dict() if hasattr(raw, 'to_dict') else dict(raw)
    return {str(k): int(v) for k, v in raw.items()}


def _truss_combo_id(obj, type_name: str, material: str, section: str,
                    create: bool = True) -> int:
    key = json.dumps([type_name, material, section])
    table = _truss_combo_table(obj)
    if key in table:
        return table[key]
    if not create:
        return 0
    new_id = max(table.values(), default=0) + 1
    table[key] = new_id
    obj[_PERM_EDGE_TRUSS_IDS_PROP] = table
    return new_id


def _truss_combos(obj) -> Dict[int, Tuple[str, str, str]]:
    """id -> (element type name, material name, section name)."""
    out = {}
    for key, cid in _truss_combo_table(obj).items():
        try:
            t, m, sname = json.loads(key)
            out[cid] = (str(t), str(m), str(sname))
        except Exception:
            continue
    return out


def _truss_ids_of_type(obj, type_name: str) -> Set[int]:
    return {cid for cid, (t, _, _) in _truss_combos(obj).items() if t == type_name}


def _read_edge_truss_tags(mesh):
    """OBJECT MODE: per-edge truss tag array, or None if the mesh has none."""
    attr = mesh.attributes.get(PERM_EDGE_TRUSS_ATTR)
    if attr is None or attr.domain != 'EDGE' or attr.data_type != 'INT':
        return None
    vals = np.zeros(len(mesh.edges), dtype=np.int32)
    if len(vals):
        attr.data.foreach_get('value', vals)
    return vals


def _set_selected_edges_truss(obj, value: int) -> Tuple[int, int]:
    """EDIT MODE: tag every selected edge. Returns (n_tagged, n_overwritten)."""
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    layer = (bm.edges.layers.int.get(PERM_EDGE_TRUSS_ATTR)
             or bm.edges.layers.int.new(PERM_EDGE_TRUSS_ATTR))
    n = n_over = 0
    for e in bm.edges:
        if e.select:
            if e[layer] not in (0, value):
                n_over += 1
            e[layer] = value
            n += 1
    bmesh.update_edit_mesh(me)
    return n, n_over


def _clear_edges_truss(obj, ids: Set[int], only_selected: bool = False) -> int:
    """Set truss tags in `ids` back to 0 (all edges, or only selected ones).
    Works from Object or Edit Mode."""
    if not ids:
        return 0
    was_edit = obj.mode == 'EDIT'
    if was_edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    n = 0
    attr = obj.data.attributes.get(PERM_EDGE_TRUSS_ATTR)
    if attr is not None and attr.domain == 'EDGE':
        edges = obj.data.edges
        for i, d in enumerate(attr.data):
            if d.value in ids and (not only_selected or edges[i].select):
                d.value = 0
                n += 1
    if was_edit:
        bpy.ops.object.mode_set(mode='EDIT')
    return n


def _count_truss_edges(obj, type_name: str) -> int:
    was_edit = obj.mode == 'EDIT'
    if was_edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    tags = _read_edge_truss_tags(obj.data)
    ids = _truss_ids_of_type(obj, type_name)
    n = int(np.isin(tags, list(ids)).sum()) if (tags is not None and ids) else 0
    if was_edit:
        bpy.ops.object.mode_set(mode='EDIT')
    return n


def _set_faces_tag_object_mode(obj, kind: str, face_indices, value: int) -> None:
    """OBJECT MODE: write `value` into the face attribute of the given faces."""
    me = obj.data
    name = PERM_FACE_ATTR[kind]
    attr = me.attributes.get(name)
    if attr is None:
        attr = me.attributes.new(name=name, type='INT', domain='FACE')
    for i in face_indices:
        attr.data[i].value = value


def _get_selected_face_vertices(obj):
    """Return set of vertex indices belonging to currently-selected faces (Edit Mode)."""
    if obj.mode != 'EDIT':
        return set()
    bpy.ops.object.mode_set(mode='OBJECT')
    verts = set()
    for poly in obj.data.polygons:
        if poly.select:
            for vidx in poly.vertices:
                verts.add(vidx)
    # If no face selected, fall back to vertex selection
    if not verts:
        for v in obj.data.vertices:
            if v.select:
                verts.add(v.index)
    bpy.ops.object.mode_set(mode='EDIT')
    return verts


class PERM_OT_AssignMaterialToFaces(bpy.types.Operator):
    """PERM: Assign the currently-active material to selected faces.
    
    Workflow:
      1. Pick a material in the Materials list (section 1 of panel)
      2. Enter Edit Mode, select the faces (or vertices) to receive that material
      3. Click 'Assign Material to Faces'
      4. All those faces' vertices go into a 'PERM_Material_<material_name>' vertex group.
    
    At solve time, each shell element whose vertices ARE ALL in a material group
    uses that material's E, ν, ρ, etc.
    Elements with vertices in multiple material groups: first match wins.
    Elements with no material group: use scene material #0 (default).
    """
    bl_idname = "perm.assign_material_to_faces"
    bl_label = "PERM: Assign Material to Faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        scn = context.scene
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select faces first")
            return {'CANCELLED'}
        if scn.perm_material_index < 0 or scn.perm_material_index >= len(scn.perm_materials):
            self.report({'ERROR'}, "Pick a material in the Materials list first")
            return {'CANCELLED'}
        mat = scn.perm_materials[scn.perm_material_index]
        mat_name = mat.name.strip()
        if not mat_name:
            self.report({'ERROR'}, "Active material has empty name"); return {'CANCELLED'}
        
        verts = _get_selected_face_vertices(obj)
        if not verts:
            self.report({'ERROR'}, "No faces/vertices selected"); return {'CANCELLED'}
        
        group_name = f"PERM_Material_{mat_name}"
        bpy.ops.object.mode_set(mode='OBJECT')
        # Remove these verts from OTHER material groups (one material per face)
        for vg in obj.vertex_groups:
            if vg.name.startswith("PERM_Material_") and vg.name != group_name:
                vg.remove(list(verts))
        # Add to this material's group
        if group_name in obj.vertex_groups:
            vg = obj.vertex_groups[group_name]
        else:
            vg = obj.vertex_groups.new(name=group_name)
        vg.add(list(verts), 1.0, 'REPLACE')
        bpy.ops.object.mode_set(mode='EDIT')
        # Authoritative per-face assignment (see PER-FACE MATERIAL / SECTION TAGS)
        n_faces = _set_selected_faces_tag(obj, 'material',
                                          _face_tag_id(obj, 'material', mat_name))
        
        self.report({'INFO'},
            f"PERM: Assigned material '{mat_name}' to {n_faces} faces")
        return {'FINISHED'}


class PERM_OT_AssignSectionToFaces(bpy.types.Operator):
    """PERM: Assign the currently-active section to selected faces.
    
    For shells: section.thickness becomes the element's thickness.
    For trusses (this won't apply — trusses use their own per-element section).
    """
    bl_idname = "perm.assign_section_to_faces"
    bl_label = "PERM: Assign Section to Faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        scn = context.scene
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh"); return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select faces first")
            return {'CANCELLED'}
        if scn.perm_section_index < 0 or scn.perm_section_index >= len(scn.perm_sections):
            self.report({'ERROR'}, "Pick a section in the Sections list first")
            return {'CANCELLED'}
        sec = scn.perm_sections[scn.perm_section_index]
        sec_name = sec.name.strip()
        if not sec_name:
            self.report({'ERROR'}, "Active section has empty name"); return {'CANCELLED'}
        
        verts = _get_selected_face_vertices(obj)
        if not verts:
            self.report({'ERROR'}, "No faces/vertices selected"); return {'CANCELLED'}
        
        group_name = f"PERM_Section_{sec_name}"
        bpy.ops.object.mode_set(mode='OBJECT')
        for vg in obj.vertex_groups:
            if vg.name.startswith("PERM_Section_") and vg.name != group_name:
                vg.remove(list(verts))
        if group_name in obj.vertex_groups:
            vg = obj.vertex_groups[group_name]
        else:
            vg = obj.vertex_groups.new(name=group_name)
        vg.add(list(verts), 1.0, 'REPLACE')
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Authoritative per-face assignment (see PER-FACE MATERIAL / SECTION TAGS)
        n_faces = _set_selected_faces_tag(obj, 'section',
                                          _face_tag_id(obj, 'section', sec_name))
        # The face tag IS the assignment (sections belong to elements, not nodes);
        # the vertex group above is only kept so files from older versions load.
        self.report({'INFO'},
            f"PERM: Assigned section '{sec_name}' to {n_faces} faces")
        return {'FINISHED'}


class PERM_OT_SelectMaterialFaces(bpy.types.Operator):
    """PERM: Select all faces assigned to the active material."""
    bl_idname = "perm.select_material_faces"
    bl_label = "PERM: Select Material's Faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_material_index < 0 or scn.perm_material_index >= len(scn.perm_materials):
            return {'CANCELLED'}
        mat_name = scn.perm_materials[scn.perm_material_index].name.strip()
        tag = _face_tag_id(obj, 'material', mat_name, create=False)
        if tag and _select_faces_with_tag(obj, 'material', tag):
            return {'FINISHED'}
        group_name = f"PERM_Material_{mat_name}"
        if group_name not in obj.vertex_groups:
            self.report({'WARNING'}, f"No faces assigned to '{mat_name}' yet")
            return {'CANCELLED'}
        vg = obj.vertex_groups[group_name]
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    v.select = True
                    break
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class PERM_OT_SelectSectionFaces(bpy.types.Operator):
    """PERM: Select all faces assigned to the active section."""
    bl_idname = "perm.select_section_faces"
    bl_label = "PERM: Select Section's Faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_section_index < 0 or scn.perm_section_index >= len(scn.perm_sections):
            return {'CANCELLED'}
        sec_name = scn.perm_sections[scn.perm_section_index].name.strip()
        tag = _face_tag_id(obj, 'section', sec_name, create=False)
        if tag and _select_faces_with_tag(obj, 'section', tag):
            return {'FINISHED'}
        group_name = f"PERM_Section_{sec_name}"
        if group_name not in obj.vertex_groups:
            self.report({'WARNING'}, f"No faces assigned to '{sec_name}' yet")
            return {'CANCELLED'}
        vg = obj.vertex_groups[group_name]
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    v.select = True
                    break
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class PERM_OT_UnassignMaterial(bpy.types.Operator):
    """Remove the active material from ALL its faces (they fall back to the first material in the list)"""
    bl_idname = "perm.unassign_material"
    bl_label = "PERM: Unassign Material"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_material_index < 0 or scn.perm_material_index >= len(scn.perm_materials):
            return {'CANCELLED'}
        mat_name = scn.perm_materials[scn.perm_material_index].name.strip()
        group_name = f"PERM_Material_{mat_name}"
        n_cleared = _clear_faces_tag(obj, 'material',
                                     _face_tag_id(obj, 'material', mat_name, create=False))
        if group_name in obj.vertex_groups or n_cleared:
            if group_name in obj.vertex_groups:
                obj.vertex_groups.remove(obj.vertex_groups[group_name])
            self.report({'INFO'}, f"PERM: Unassigned '{mat_name}' from all faces")
        return {'FINISHED'}


class PERM_OT_UnassignSection(bpy.types.Operator):
    """Remove the active cross-section from ALL its faces (they fall back to the default thickness)"""
    bl_idname = "perm.unassign_section"
    bl_label = "PERM: Unassign Section"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = context.active_object
        scn = context.scene
        if not obj or scn.perm_section_index < 0 or scn.perm_section_index >= len(scn.perm_sections):
            return {'CANCELLED'}
        sec_name = scn.perm_sections[scn.perm_section_index].name.strip()
        group_name = f"PERM_Section_{sec_name}"
        n_cleared = _clear_faces_tag(obj, 'section',
                                     _face_tag_id(obj, 'section', sec_name, create=False))
        if group_name in obj.vertex_groups or n_cleared:
            if group_name in obj.vertex_groups:
                obj.vertex_groups.remove(obj.vertex_groups[group_name])
            self.report({'INFO'}, f"PERM: Unassigned '{sec_name}' from all faces")
        return {'FINISHED'}


class PERM_OT_AddMaterial(bpy.types.Operator):
    """PERM: Add a new material to the scene's material list."""
    bl_idname = "perm.add_material"
    bl_label = "PERM: Add Material"
    bl_options = {'REGISTER', 'UNDO'}
    
    name: bpy.props.StringProperty(name="Name", default="NewMaterial")
    youngs_modulus: bpy.props.FloatProperty(name="E (Pa)", default=30e9, min=1e6)
    poissons_ratio: bpy.props.FloatProperty(name="ν", default=0.2, min=0.0, max=0.499)
    density: bpy.props.FloatProperty(name="ρ (kg/m³)", default=2400.0, min=0.0)
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def execute(self, context):
        scn = context.scene
        item = scn.perm_materials.add()
        item.name = self.name.strip() or f"Material_{len(scn.perm_materials)}"
        item.youngs_modulus = self.youngs_modulus
        item.poissons_ratio = self.poissons_ratio
        item.density = self.density
        scn.perm_material_index = len(scn.perm_materials) - 1
        self.report({'INFO'}, f"PERM: Added material '{item.name}'")
        return {'FINISHED'}


class PERM_OT_DeleteMaterial(bpy.types.Operator):
    """PERM: Delete the active material from the list."""
    bl_idname = "perm.delete_material"
    bl_label = "PERM: Delete Material"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scn = context.scene
        idx = scn.perm_material_index
        if idx < 0 or idx >= len(scn.perm_materials):
            return {'CANCELLED'}
        name = scn.perm_materials[idx].name
        scn.perm_materials.remove(idx)
        if scn.perm_material_index >= len(scn.perm_materials):
            scn.perm_material_index = max(0, len(scn.perm_materials) - 1)
        self.report({'INFO'}, f"PERM: Removed material '{name}'")
        return {'FINISHED'}


class PERM_UL_Materials(bpy.types.UIList):
    """List view for materials."""
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=item.name, icon='MATERIAL')
            row.label(text=f"E {item.youngs_modulus / 1e9:.4g} GPa")


class PERM_OT_AddSection(bpy.types.Operator):
    """PERM: Add a new cross-section to the scene's section list."""
    bl_idname = "perm.add_section"
    bl_label = "PERM: Add Section"
    bl_options = {'REGISTER', 'UNDO'}
    
    name: bpy.props.StringProperty(name="Name", default="NewSection")
    thickness: bpy.props.FloatProperty(name="Shell t (m)", default=0.3, min=1e-6)
    area: bpy.props.FloatProperty(name="A (m²)", default=1e-3, min=1e-12)
    Ix:   bpy.props.FloatProperty(name="Ix (m⁴)", default=1e-8, min=0.0)
    Iy:   bpy.props.FloatProperty(name="Iy (m⁴)", default=1e-8, min=0.0)
    J:    bpy.props.FloatProperty(name="J (m⁴)",  default=1e-8, min=0.0)
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def execute(self, context):
        scn = context.scene
        item = scn.perm_sections.add()
        item.name = self.name.strip() or f"Section_{len(scn.perm_sections)}"
        item.thickness = self.thickness
        item.area = self.area
        item.Ix = self.Ix; item.Iy = self.Iy; item.J = self.J
        scn.perm_section_index = len(scn.perm_sections) - 1
        self.report({'INFO'}, f"PERM: Added section '{item.name}'")
        return {'FINISHED'}


class PERM_OT_DeleteSection(bpy.types.Operator):
    """PERM: Delete the active section from the list."""
    bl_idname = "perm.delete_section"
    bl_label = "PERM: Delete Section"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scn = context.scene
        idx = scn.perm_section_index
        if idx < 0 or idx >= len(scn.perm_sections):
            return {'CANCELLED'}
        name = scn.perm_sections[idx].name
        scn.perm_sections.remove(idx)
        if scn.perm_section_index >= len(scn.perm_sections):
            scn.perm_section_index = max(0, len(scn.perm_sections) - 1)
        self.report({'INFO'}, f"PERM: Removed section '{name}'")
        return {'FINISHED'}


class PERM_UL_Sections(bpy.types.UIList):
    """List view for sections."""
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=item.name, icon='MESH_CYLINDER')
            row.label(text=f"t {item.thickness * 1e3:.4g} mm")


# ─────────────────────────────────────────────────────────────────────────────
# TRUSS OPERATORS (add/delete truss elements; section management)
# ─────────────────────────────────────────────────────────────────────────────
# Truss data model on the mesh object:
#   - Each truss element is stored as a vertex group `PERM_Truss_<n>` containing
#     exactly 2 vertices (the two end nodes).
#   - The cross-section area and section name are stored as custom properties
#     on the object: `perm_truss_<n>_area`, `perm_truss_<n>_section`.
#   - The faces (if any) whose vertices are all inside a PERM_Truss_* group
#     are marked viz-only during FEA assembly.

def find_perm_element_type_truss_groups(obj):
    """Find all truss instances created via the new Element Types workflow.
    Returns list of (type_name, group_name, area, [v1, v2]).
    
    These are vertex groups named PERM_Element_<typename>_t<n>, each containing
    exactly 2 vertices.
    """
    results = []
    if not obj or obj.type != 'MESH':
        return results
    for vg in obj.vertex_groups:
        name = vg.name
        if not name.startswith("PERM_Element_"):
            continue
        # Match PERM_Element_<typename>_t<n>
        rest = name[len("PERM_Element_"):]
        m_t = _TRUSS_GROUP_RE.match(rest)
        if m_t is None:
            continue
        type_name, suffix = m_t.group(1), m_t.group(2)
        # Collect the 2 vertices
        verts = []
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    verts.append(v.index)
                    break
        if len(verts) != 2:
            continue
        # Read area from custom property
        area = float(obj.get(f"perm_truss_{type_name}_t{suffix}_area", 1e-3))
        results.append((type_name, name, area, verts))
    return results


def find_perm_element_type_shell_groups(obj):
    """Find shell element-type assignments from the new Element Types workflow.
    Returns dict {type_name: set_of_vertex_indices}.
    
    These are vertex groups named PERM_Element_<typename> (no _t suffix).
    """
    results = {}
    if not obj or obj.type != 'MESH':
        return results
    for vg in obj.vertex_groups:
        name = vg.name
        if not name.startswith("PERM_Element_"):
            continue
        rest = name[len("PERM_Element_"):]
        if _is_truss_instance_group(rest):  # skip truss instances (<type>_t<n>)
            continue
        type_name = rest
        verts = set()
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    verts.add(v.index)
                    break
        if verts:
            results[type_name] = verts
    return results


def find_perm_truss_groups(obj):
    """Find all PERM_Truss_* vertex groups on an object.
    Returns list of (suffix, group_name, area, [vert_id1, vert_id2]).
    """
    results = []
    if not obj or obj.type != 'MESH':
        return results
    for vg in obj.vertex_groups:
        if not vg.name.startswith("PERM_Truss_"):
            continue
        suffix = vg.name[len("PERM_Truss_"):]
        # Get the two vertices in this group
        verts = []
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    verts.append(v.index)
                    break
        if len(verts) != 2:
            continue  # invalid truss (must have exactly 2 nodes)
        area = float(obj.get(f"perm_truss_{suffix}_area", 1e-3))
        results.append((suffix, vg.name, area, verts))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RAILING-POST BASES AS STEEL BASE PLATES (shell)
# ─────────────────────────────────────────────────────────────────────────────
# What a reality-captured mesh typically has where a steel post stands on a concrete deck:
#   * the deck's top surface has a HOLE under the post: inside the post footprint there is no deck surface
#     (at most a lower layer, where the deck is modelled by two offset surfaces);
#   * the steel tube flares out to the hole edge in one row of thin faces (the 'skirt'), whose bottom nodes are
#     shared with the deck (the 'ring'). The skirt is the scanner's blend of tube, weld and plate; concrete faces
#     thinner than 2 cm next to the post are treated as part of it.
# The joint is built as it is made: the hole is closed with deck concrete (the same section as the layer
# around it), a steel base plate (PERM_PLATE_SIZE square, PERM_PLATE_THICKNESS) is bonded on the deck top
# (shell elements on the deck nodes, offset so the steel lies above the surface), the skirt row is taken out
# and the tube walls are carried down to the plate at their own perimeter (welded). No rigid links.
# The new nodes and elements are FE-only (not in the Blender mesh): results on them are in the report, not in
# the colour plots.

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIONS BETWEEN MESH PARTS (fixed / pinned / hinge) AND CONTACT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# A connection joins two PARTS of the model at a joint. A part is a loose part of the mesh (faces and edges linked
# through shared vertices, as Blender's 'Separate by Loose Parts'); a frame member drawn as edges is a part too. Side A and side B are the vertex groups PERM_Conn_<name>_A / _B: the joint vertices of each part.
#   FIXED   every node of side B is tied rigidly (all six components) to the nearest node of side A: a welded or
#           monolithic joint. Vertices that coincide are merged when the model is built, so a FIXED joint without a
#           gap is continuous already and needs no link.
#   PINNED  each side is made rigid up to the joint centre and the two are joined there by the translations only: a
#           ball joint that transmits force but no moment.
#   HINGE   the rotation about the hinge axis is free, everything else is tied. When the joint nodes lie on a line
#           along the axis (a hinge line between two plates, for example) the node pairs are linked directly, so
#           the parts stay flexible along the line; otherwise the joint works through its centre, as PINNED.
# A pinned or hinged joint needs the two parts to be separate in the mesh. Where their vertices coincide they are
# merged into one FE node when the model is built; that node is split again, side B getting its own copy.
# Links are penalty constraints (PERM_CONN_PENALTY × the largest diagonal stiffness of the same DOF kind, as the
# rigid ties of the connection details); the force each joint carries is recovered from them and reported.
PERM_PART_ATTR = "perm_part"
PERM_CONN_TYPES = [
    ('FIXED', "Fixed", "Rigid joint: every node of side B tied to the nearest node of side A in all six "
                       "components (welded or monolithic). Transmits force and moment"),
    ('PINNED', "Pinned", "Ball joint at the joint centre: transmits force, no moment. Each side is made rigid up "
                         "to the joint centre"),
    ('HINGE', "Hinge", "Free rotation about the hinge axis, everything else tied. A hinge line when the joint "
                       "nodes lie along the axis, otherwise a hinge at the joint centre"),
]
PERM_CONN_AXES = [
    ('LINE', "Along the joint", "The direction in which the joint nodes are spread (a hinge line)"),
    ('X', "Global X", "Hinge axis parallel to global X"),
    ('Y', "Global Y", "Hinge axis parallel to global Y"),
    ('Z', "Global Z", "Hinge axis parallel to global Z"),
]


def _perm_mesh_parts(mesh):
    """Loose parts of the mesh (vertices linked by edges, as Blender's 'Separate by Loose Parts').
    Returns (vertex part, face part, number of parts)."""
    nv, nf, ne = len(mesh.vertices), len(mesh.polygons), len(mesh.edges)
    if nv == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(nf, dtype=np.int64), 0
    ev = np.zeros(2 * ne, dtype=np.int64)
    if ne:
        mesh.edges.foreach_get('vertices', ev)
    ev = ev.reshape(-1, 2)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(ev)), (ev[:, 0], ev[:, 1])), shape=(nv, nv))
    n, vpart = connected_components(g, directed=False)
    fpart = np.zeros(nf, dtype=np.int64)
    if nf:
        ls = np.zeros(nf, dtype=np.int64); mesh.polygons.foreach_get('loop_start', ls)
        lv = np.zeros(len(mesh.loops), dtype=np.int64); mesh.loops.foreach_get('vertex_index', lv)
        fpart = vpart[lv[ls]].astype(np.int64)
    return vpart.astype(np.int64), fpart, int(n)


def _perm_write_part_attr(obj):
    """Store each face's part number as a face attribute (the FE working copy inherits it). Returns the vertex parts."""
    me = obj.data
    vpart, fpart, n = _perm_mesh_parts(me)
    a = me.attributes.get(PERM_PART_ATTR)
    if a is not None and (a.domain != 'FACE' or a.data_type != 'INT'):
        me.attributes.remove(a); a = None
    if a is None:
        a = me.attributes.new(PERM_PART_ATTR, 'INT', 'FACE')
    if len(fpart):
        a.data.foreach_set('value', fpart.astype(np.int32))
    return vpart, n


def _perm_groups_verts(obj, names) -> Dict[str, List[int]]:
    """Vertex indices of several vertex groups in one pass over the mesh."""
    idx = {obj.vertex_groups[n].index: n for n in names if n and n in obj.vertex_groups}
    out = {n: [] for n in names if n}
    if not idx:
        return out
    for v in obj.data.vertices:
        for g in v.groups:
            nm = idx.get(g.group)
            if nm is not None:
                out[nm].append(v.index)
    return out


def _perm_conn_axis(kind: str, pts: np.ndarray) -> np.ndarray:
    if kind in ('X', 'Y', 'Z'):
        return np.eye(3)[{'X': 0, 'Y': 1, 'Z': 2}[kind]]
    if len(pts) < 2:
        return np.array([1.0, 0.0, 0.0])
    d = pts - pts.mean(0)
    w, v = np.linalg.eigh(d.T @ d)
    return v[:, -1] / np.linalg.norm(v[:, -1])


def _perm_build_connections(converter, obj, src_v2n, vpart) -> list:
    """Resolve the object's connections on the FE model: side node sets, split shared nodes of pinned / hinged
    joints, add the two joint-centre nodes of a point joint. Runs after the model is built and before the solver
    numbers the nodes. Stores converter._connections, _conn_nodes (every joint node) and _joint_masters."""
    conns = []
    converter._connections = conns
    converter._conn_nodes = set()
    converter._joint_masters = set()
    items = list(getattr(obj, 'perm_connections', []) or [])
    if not items:
        return conns
    nodes, els = converter.nodes, converter.elements
    converter._next_node_id = max(int(getattr(converter, '_next_node_id', 1)), max(nodes) + 1)

    def new_node(xyz):
        nid = converter._next_node_id; converter._next_node_id += 1
        nodes[nid] = FENode(id=nid, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        return nid
    # part of every element: faces from the working copy's face attribute, bars from their two source vertices
    fp = None
    wobj = getattr(converter, 'working_object', None)
    if wobj is not None:
        a = wobj.data.attributes.get(PERM_PART_ATTR)
        if a is not None and a.domain == 'FACE' and len(wobj.data.polygons):
            fp = np.zeros(len(wobj.data.polygons), dtype=np.int32); a.data.foreach_get('value', fp)
    epart = {}
    node_els = defaultdict(list)
    for eid, e in els.items():
        if getattr(e, 'viz_only', False):
            continue
        for n in e.node_ids:
            node_els[n].append(eid)
        if _perm_is_line(e):
            sv = getattr(e, 'src_verts', None)
            if sv is not None and vpart is not None:
                ps = {int(vpart[v]) for v in sv if v is not None and 0 <= v < len(vpart) and vpart[v] >= 0}
                epart[eid] = ps.pop() if len(ps) == 1 else -1
            continue
        poly = converter._element_to_poly.get(eid)
        if fp is not None and poly is not None and 0 <= poly < len(fp):
            epart[eid] = int(fp[poly])
    gverts = _perm_groups_verts(obj, [c.group_a for c in items] + [c.group_b for c in items])

    def major(vs):
        ps = [int(vpart[v]) for v in vs if vpart is not None and 0 <= v < len(vpart) and vpart[v] >= 0]
        return Counter(ps).most_common(1)[0][0] if ps else -1
    for c in items:
        A_src, B_src = gverts.get(c.group_a, []), gverts.get(c.group_b, [])
        A = {src_v2n[v] for v in A_src if v in src_v2n}
        B = {src_v2n[v] for v in B_src if v in src_v2n}
        if not A or not B:
            print(f"[PERM] WARNING: joint '{c.name}' skipped - side A has {len(A)}, side B {len(B)} nodes")
            continue
        pa, pb = major(A_src), major(B_src)
        n_split = 0
        if c.ctype in ('PINNED', 'HINGE'):
            if pa < 0 or pb < 0 or pa == pb:
                print(f"[PERM] WARNING: joint '{c.name}' ({c.ctype}) skipped - both sides are in the same mesh part, "
                      f"which is continuous there. Separate the parts (e.g. Edge Split) for a pinned or hinged joint.")
                continue
            for n in sorted(A | B):
                es = node_els.get(n, [])
                in_b = [e for e in es if epart.get(e) == pb]
                if not in_b or len(in_b) == len(es):
                    continue                                   # used by side B only: nothing shared
                nn = new_node((nodes[n].x, nodes[n].y, nodes[n].z))
                for e in in_b:
                    els[e].node_ids = [nn if q == n else q for q in els[e].node_ids]
                node_els[nn] = in_b; node_els[n] = [e for e in es if e not in in_b]
                B.discard(n); B.add(nn); A.add(n); n_split += 1
        pts = np.array([[nodes[n].x, nodes[n].y, nodes[n].z] for n in sorted(A | B)])
        ctr = pts.mean(0)
        ctype = c.ctype
        axis = _perm_conn_axis(c.axis, pts) if ctype == 'HINGE' else None
        if ctype == 'HINGE' and c.axis == 'LINE' and float(np.ptp((pts - ctr) @ axis)) < 1e-6:
            print(f"[PERM] WARNING: joint '{c.name}' is a single point, so 'along the joint' gives no hinge axis: "
                  f"it is treated as PINNED (all rotations free). Pick global X, Y or Z for a hinge about one axis.")
            ctype, axis = 'PINNED', None
        if ctype == 'FIXED':
            method = 'pairs'
        elif ctype == 'PINNED':
            method = 'point'
        else:
            d = pts - ctr
            ext = max(float(np.ptp(d @ axis)), 1e-9)
            off = float(np.linalg.norm(d - np.outer(d @ axis, axis), axis=1).max()) if len(d) else 0.0
            method = 'pairs' if off <= max(0.002, PERM_CONN_LINE_TOL * ext) else 'point'
            if c.axis == 'LINE' and method == 'point':
                print(f"[PERM] NOTE: joint '{c.name}': the joint nodes do not lie on a line, so 'along the joint' "
                      f"is only their main direction; pick a global axis for a hinge at a compact joint.")
        spec = dict(name=c.name, ctype=ctype, axis_kind=c.axis, axis=axis, A=sorted(A), B=sorted(B),
                    centre=ctr, method=method, split=n_split, parts=(pa, pb))
        if method == 'point':
            spec['MA'], spec['MB'] = new_node(ctr), new_node(ctr)
            converter._joint_masters.update((spec['MA'], spec['MB']))
        conns.append(spec)
        converter._conn_nodes.update(A); converter._conn_nodes.update(B)
        print(f"[PERM] joint '{c.name}': {ctype}" + (f" about {c.axis}" if ctype == 'HINGE' else "")
              + f", {method}; side A {len(A)} nodes (part {pa}), side B {len(B)} nodes (part {pb})"
              + (f"; {n_split} shared node(s) split" if n_split else ""))
    return conns


def _perm_new_conn_name(obj, prefix):
    used = {c.name for c in obj.perm_connections}
    k = 1
    while f"{prefix}{k}" in used or f"PERM_Conn_{prefix}{k}_A" in obj.vertex_groups:
        k += 1
    return f"{prefix}{k}"


def _perm_add_connection(obj, name, A, B, ctype='FIXED', axis='LINE', detected=False):
    """Create the two vertex groups and the list entry (OBJECT MODE)."""
    ga, gb = f"PERM_Conn_{name}_A", f"PERM_Conn_{name}_B"
    for g in (ga, gb):
        if g in obj.vertex_groups:
            obj.vertex_groups.remove(obj.vertex_groups[g])
    obj.vertex_groups.new(name=ga).add([int(v) for v in A], 1.0, 'REPLACE')
    obj.vertex_groups.new(name=gb).add([int(v) for v in B], 1.0, 'REPLACE')
    it = obj.perm_connections.add()
    it.name = name; it.ctype = ctype; it.axis = axis; it.group_a = ga; it.group_b = gb
    it.detected = detected; it.n_a = len(A); it.n_b = len(B)
    obj.perm_connection_index = len(obj.perm_connections) - 1
    return it


def _perm_remove_connection(obj, i) -> str:
    it = obj.perm_connections[i]
    name = it.name
    for g in (it.group_a, it.group_b):
        if g and g in obj.vertex_groups:
            obj.vertex_groups.remove(obj.vertex_groups[g])
    obj.perm_connections.remove(i)
    obj.perm_connection_index = min(max(0, obj.perm_connection_index), max(0, len(obj.perm_connections) - 1))
    return name


def perm_detect_contacts(obj, tol=0.005, ctype='FIXED', replace=True, axis='LINE') -> Tuple[int, int]:
    """Find where separate parts of the mesh touch or come within `tol` and add one connection per contact zone.
    Side A is the larger part (more vertices). Returns (connections added, part pairs). OBJECT MODE."""
    me = obj.data
    vpart, fpart, n = _perm_mesh_parts(me)
    if replace:
        for i in reversed(range(len(obj.perm_connections))):
            if obj.perm_connections[i].detected:
                _perm_remove_connection(obj, i)
    if n < 2:
        return 0, 0
    co = np.zeros(3 * len(me.vertices)); me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]
    psize = np.bincount(vpart, minlength=n)
    from scipy.spatial import cKDTree
    on = np.nonzero(vpart >= 0)[0]
    pairs = cKDTree(co[on]).query_pairs(max(float(tol), 1e-7), output_type='ndarray')
    if not len(pairs):
        return 0, 0
    pairs = on[pairs]
    pa, pb = vpart[pairs[:, 0]], vpart[pairs[:, 1]]
    keep = pa != pb
    pairs, pa, pb = pairs[keep], pa[keep], pb[keep]
    by_pair = defaultdict(set)
    partners = defaultdict(set)
    for (u, v), a, b in zip(pairs, pa, pb):
        key = (min(a, b), max(a, b))
        by_pair[key].update((int(u), int(v)))
        partners[int(u)].add(int(v)); partners[int(v)].add(int(u))
    # Contact vertices of one part that are neighbours in the mesh (an edge or a face in common) belong to the same
    # joint when their partners in the other part are neighbours too (or the same vertex): a seam between two
    # plates stays one joint however coarse the mesh, while a member touching the same part at both of its ends
    # still gives two joints.
    inC = np.zeros(len(me.vertices), dtype=bool); inC[list(partners)] = True
    adj = set()
    ne = len(me.edges)
    if ne:
        ev = np.zeros(2 * ne, dtype=np.int64); me.edges.foreach_get('vertices', ev); ev = ev.reshape(-1, 2)
        for u, v in ev[inC[ev[:, 0]] & inC[ev[:, 1]]]:
            adj.add((min(int(u), int(v)), max(int(u), int(v))))
    nf = len(me.polygons)
    if nf:
        lt = np.zeros(nf, dtype=np.int64); me.polygons.foreach_get('loop_total', lt)
        lv = np.zeros(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', lv)
        fl = np.repeat(np.arange(nf), lt); m = inC[lv]
        per_face = defaultdict(list)
        for f, v in zip(fl[m], lv[m]):
            per_face[int(f)].append(int(v))
        for vs_ in per_face.values():
            for i_ in range(len(vs_)):
                for j_ in range(i_ + 1, len(vs_)):
                    adj.add((min(vs_[i_], vs_[j_]), max(vs_[i_], vs_[j_])))

    def neighbours(p, q):
        return p == q or (min(p, q), max(p, q)) in adj
    link = max(10.0 * float(tol), 0.05)
    added = 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    for (a, b), vs in sorted(by_pair.items()):
        vs = np.array(sorted(vs)); pos = {int(v): k for k, v in enumerate(vs)}
        pp = cKDTree(co[vs]).query_pairs(link, output_type='ndarray')
        ii = [int(x) for x in pp[:, 0]] if len(pp) else []
        jj = [int(x) for x in pp[:, 1]] if len(pp) else []
        for u, v in adj:
            if u in pos and v in pos and vpart[u] == vpart[v]:
                other = b if vpart[u] == a else a
                pu = [p for p in partners[u] if vpart[p] == other]
                pv = [q for q in partners[v] if vpart[q] == other]
                if any(neighbours(p, q) for p in pu for q in pv):
                    ii.append(pos[u]); jj.append(pos[v])
        g = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(vs), len(vs)))
        _, lab = connected_components(g, directed=False)
        big, small = (a, b) if psize[a] >= psize[b] else (b, a)
        for k in range(int(lab.max()) + 1):
            z = vs[lab == k]
            A = [int(v) for v in z if vpart[v] == big]; B = [int(v) for v in z if vpart[v] == small]
            if A and B:
                _perm_add_connection(obj, _perm_new_conn_name(obj, "C"), A, B, ctype=ctype, axis=axis, detected=True)
                added += 1
    return added, len(by_pair)


class PERM_ConnectionPG(bpy.types.PropertyGroup):
    """One joint between two mesh parts."""
    name: bpy.props.StringProperty(name="Name", default="J1")
    ctype: bpy.props.EnumProperty(name="Type", items=PERM_CONN_TYPES, default='FIXED')
    axis: bpy.props.EnumProperty(name="Hinge axis", items=PERM_CONN_AXES, default='LINE')
    group_a: bpy.props.StringProperty(default="")
    group_b: bpy.props.StringProperty(default="")
    detected: bpy.props.BoolProperty(default=False)
    n_a: bpy.props.IntProperty(default=0)
    n_b: bpy.props.IntProperty(default=0)


class PERM_UL_Connections(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon='LINKED')
        row.prop(item, "ctype", text="")
        if item.ctype == 'HINGE':
            row.prop(item, "axis", text="")
        row.label(text=f"{item.n_a}/{item.n_b}")


class PERM_OT_AddConnection(bpy.types.Operator):
    """PERM: Make a joint from the selected vertices of two mesh parts (side A = the larger part)"""
    bl_idname = "perm.add_connection"
    bl_label = "PERM: Add Connection"
    bl_options = {'REGISTER', 'UNDO'}
    name: bpy.props.StringProperty(name="Name", default="")
    ctype: bpy.props.EnumProperty(name="Type", items=PERM_CONN_TYPES, default='FIXED')
    axis: bpy.props.EnumProperty(name="Hinge axis", items=PERM_CONN_AXES, default='LINE')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=360)

    def execute(self, context):
        obj = context.active_object
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        if _perm_need_scipy(self):
            return {'CANCELLED'}
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'ERROR'}, "Edit Mode: select the joint vertices of both parts first"); return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            sel = np.array([v.index for v in obj.data.vertices if v.select], dtype=np.int64)
            vpart, fpart, n = _perm_mesh_parts(obj.data)
            ps = Counter(int(vpart[v]) for v in sel if vpart[v] >= 0)
            if len(ps) < 2:
                self.report({'ERROR'}, "The selection lies in one mesh part, which is continuous there. Select the joint "
                            "vertices of TWO separate parts (Edge Split or Detect Contacts for touching parts)")
                return {'CANCELLED'}
            (p1, _), (p2, _) = ps.most_common(2)
            nf = np.bincount(vpart, minlength=n)
            big, small = (p1, p2) if nf[p1] >= nf[p2] else (p2, p1)
            A = [int(v) for v in sel if vpart[v] == big]; B = [int(v) for v in sel if vpart[v] == small]
            name = self.name.strip() or _perm_new_conn_name(obj, "J")
            if any(c.name == name for c in obj.perm_connections):
                self.report({'ERROR'}, f"A connection '{name}' exists already"); return {'CANCELLED'}
            _perm_add_connection(obj, name, A, B, ctype=self.ctype, axis=self.axis)
            msg = f"PERM: joint '{name}' ({self.ctype}): side A {len(A)} vertices, side B {len(B)} vertices"
            if len(ps) > 2:
                msg += f"; {len(ps) - 2} more part(s) in the selection ignored"
            self.report({'INFO'}, msg)
            return {'FINISHED'}
        finally:
            bpy.ops.object.mode_set(mode='EDIT')


class PERM_OT_DetectContacts(bpy.types.Operator):
    """PERM: Find where separate mesh parts touch (within the tolerance) and add a connection for each contact zone"""
    bl_idname = "perm.detect_contacts"
    bl_label = "PERM: Detect Contacts"
    bl_options = {'REGISTER', 'UNDO'}
    tolerance: bpy.props.FloatProperty(name="Tolerance", default=0.005, min=0.0, soft_max=0.1, precision=4,
                                       unit='LENGTH', description="Largest gap between two parts that counts as contact")
    ctype: bpy.props.EnumProperty(name="Type", items=PERM_CONN_TYPES, default='FIXED')
    axis: bpy.props.EnumProperty(name="Hinge axis", items=PERM_CONN_AXES, default='LINE',
                                 description="For HINGE joints: the axis of free rotation")
    replace: bpy.props.BoolProperty(name="Replace earlier detected joints", default=True)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=360)

    def execute(self, context):
        if _perm_need_scipy(self):
            return {'CANCELLED'}
        obj = _perm_resolve_source(context) or context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select the model mesh"); return {'CANCELLED'}
        was_edit = obj.mode == 'EDIT'
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            added, npairs = perm_detect_contacts(obj, self.tolerance, self.ctype, self.replace, self.axis)
        finally:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"PERM: {added} contact zone(s) between {npairs} pair(s) of parts "
                              f"(gap ≤ {self.tolerance * 1e3:.1f} mm) added as {self.ctype} joints")
        return {'FINISHED'}


class PERM_OT_ConnectionsApplyAll(bpy.types.Operator):
    """PERM: Give every joint of the object the type (and hinge axis) of the active joint"""
    bl_idname = "perm.connections_apply_all"
    bl_label = "PERM: Apply Type to All Joints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or not (0 <= obj.perm_connection_index < len(obj.perm_connections)):
            return {'CANCELLED'}
        src = obj.perm_connections[obj.perm_connection_index]
        for c in obj.perm_connections:
            c.ctype = src.ctype; c.axis = src.axis
        self.report({'INFO'}, f"PERM: {len(obj.perm_connections)} joint(s) set to {src.ctype}"
                              + (f" about {src.axis}" if src.ctype == 'HINGE' else ""))
        return {'FINISHED'}


class PERM_OT_RemoveConnection(bpy.types.Operator):
    """PERM: Remove the active connection (and its two vertex groups)"""
    bl_idname = "perm.remove_connection"
    bl_label = "PERM: Remove Connection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _perm_resolve_source(context) or context.active_object
        if obj is None or not (0 <= obj.perm_connection_index < len(obj.perm_connections)):
            return {'CANCELLED'}
        self.report({'INFO'}, f"PERM: removed joint '{_perm_remove_connection(obj, obj.perm_connection_index)}'")
        return {'FINISHED'}


class PERM_OT_ClearConnections(bpy.types.Operator):
    """PERM: Remove every connection of the model"""
    bl_idname = "perm.clear_connections"
    bl_label = "PERM: Remove All Connections"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _perm_resolve_source(context) or context.active_object
        if obj is None:
            return {'CANCELLED'}
        n = len(obj.perm_connections)
        for i in reversed(range(n)):
            _perm_remove_connection(obj, i)
        self.report({'INFO'}, f"PERM: removed {n} joint(s)")
        return {'FINISHED'}


class PERM_OT_SelectConnection(bpy.types.Operator):
    """PERM: Select the vertices of the active connection (side A, side B or both) in Edit Mode"""
    bl_idname = "perm.select_connection"
    bl_label = "PERM: Select Connection"
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.EnumProperty(items=[('BOTH', "Both", ""), ('A', "Side A", ""), ('B', "Side B", "")], default='BOTH')

    def execute(self, context):
        obj = _perm_resolve_source(context) or context.active_object
        if obj is None or not (0 <= obj.perm_connection_index < len(obj.perm_connections)):
            return {'CANCELLED'}
        it = obj.perm_connections[obj.perm_connection_index]
        groups = [g for s_, g in (('A', it.group_a), ('B', it.group_b)) if self.side in ('BOTH', s_)]
        context.view_layer.objects.active = obj
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        vs = _perm_groups_verts(obj, groups)
        for g in groups:
            for v in vs.get(g, []):
                obj.data.vertices[v].select = True
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


def _perm_has_wire_bars(obj) -> bool:
    """True when a truss or beam bar ends at a vertex of no face (a frame of bars). clean_mesh() would delete
    such loose vertices, and the bars with them, so the run keeps loose geometry then."""
    me = obj.data
    used = np.zeros(len(me.vertices), dtype=bool)
    if len(me.loops):
        lv = np.zeros(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', lv); used[lv] = True
    tags = _read_edge_truss_tags(me)
    if tags is not None and tags.any():
        ev = np.zeros(2 * len(me.edges), dtype=np.int64); me.edges.foreach_get('vertices', ev)
        ev = ev.reshape(-1, 2)[tags > 0]
        if len(ev) and not used[ev].all():
            return True
    for _t, _g, _a, verts in find_perm_element_type_truss_groups(obj) + find_perm_truss_groups(obj):
        if any(0 <= v < len(used) and not used[v] for v in verts):
            return True
    return False


def _point_in_poly(pt, poly):
    """point-in-polygon (ray casting), poly (n, 2)"""
    x, y = pt; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xc > x:
                inside = not inside
    return inside


def _dist_to_poly(pt, poly):
    p = np.asarray(pt, float); best = 1e9
    for i in range(len(poly)):
        a = poly[i]; b = poly[(i + 1) % len(poly)]; ab = b - a; L2 = float(ab @ ab)
        t = 0.0 if L2 <= 0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _cross2(a, b):
    """z of the 2-D cross product"""
    return float(a[0] * b[1] - a[1] * b[0])


def _poly_normal(Q):
    """PERM's element normal: quads the diagonal cross product, triangles (p1-p0) x (p2-p0)"""
    nv = np.cross(Q[2] - Q[0], Q[3] - Q[1]) if len(Q) >= 4 else np.cross(Q[1] - Q[0], Q[2] - Q[0])
    L = float(np.linalg.norm(nv))
    return nv / L if L > 0 else np.array([0.0, 0.0, 1.0])


def _perm_build_post_bases(converter) -> dict:
    """Post bases as bonded steel base plates (see the block comment above). Modifies converter.nodes/elements
    before the solver is built. Returns counts; also stored as converter._plate_info."""
    info = dict(posts=0, done=0, failed=0, fill_el=0, fill_nodes=0, plate_el=0, walls=0, skirt=0, enlarged=0,
                fill_area=0.0, plate_mass=0.0)
    converter._plate_info = info
    converter._plate_sections = {}
    converter._plate_fill = set()
    converter._plates = []
    converter._plate_ties = []                    # (master node id, slave node id): rigid, for near-coincident tube/plate nodes
    if not (PERM_POST_PLATES and str(PERM_PLATE_MODEL).lower() == "shell"):
        return info
    from scipy.spatial import Delaunay, cKDTree
    nodes, els, mats = converter.nodes, converter.elements, converter.materials
    P = {nid: np.array([n.x, n.y, n.z], float) for nid, n in nodes.items()}
    sec_names = _element_face_tag_names(converter, 'section')
    rows = {}                                   # eid -> (corner ids, centre, normal, kind)  kind: 'steel' | 'top' | 'low' | 'conc'
    for eid, e in els.items():
        tn = getattr(e.element_type, 'name', '')
        if tn not in ('QUAD4', 'TRI3') or getattr(e, 'viz_only', False):
            continue
        ids = list(e.node_ids[:4 if tn == 'QUAD4' else 3])
        Q = np.array([P[i] for i in ids]); n = _poly_normal(Q)
        m = mats.get(e.material_id, mats[0])
        if m.youngs_modulus >= 1.0e11:
            kind = 'steel'
        elif m.density > 0:
            kind = 'conc'
            if float(e.thickness) < 0.02:
                kind = 'thin'               # concrete-tagged faces with a railing-like thickness: part of the scanned post base
            elif abs(n[2]) >= 0.5:
                kind = 'top' if float(getattr(e, 'offset', 0.0)) * np.sign(n[2]) <= 0.0 else 'low'
        else:
            continue
        rows[eid] = (ids, Q.mean(0), n, kind)
    st_n, co_n = set(), set()
    for ids, c, n, k in rows.values():
        (st_n if k in ('steel', 'thin') else co_n).update(ids)
    joint = np.array(sorted(st_n & co_n))
    if not len(joint):
        return info
    # posts = clusters of steel-concrete nodes (single linkage within 0.10 m)
    J = np.array([P[i] for i in joint]); parent = list(range(len(joint)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in cKDTree(J).query_pairs(0.10):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    clusters = defaultdict(list)
    for k in range(len(joint)):
        clusters[find(k)].append(int(joint[k]))
    # span axis of the deck (plan principal direction of the top concrete layer)
    tc = np.array([c[:2] for ids, c, n, k in rows.values() if k == 'top'])
    tc = tc - tc.mean(0); w_, v_ = np.linalg.eigh(tc.T @ tc); ax = v_[:, -1]; ac = np.array([-ax[1], ax[0]])
    node_el = defaultdict(list)
    for eid, (ids, c, n, k) in rows.items():
        for i in ids:
            node_el[i].append(eid)
    lowpts = np.array([i for ids, c, n, k in rows.values() if k == 'low' for i in ids])
    lowpts = np.unique(lowpts) if len(lowpts) else lowpts
    LP = np.array([P[i] for i in lowpts]) if len(lowpts) else np.zeros((0, 3))
    # the plate material
    pmid = max(mats) + 1
    mats[pmid] = MaterialProperty(id=pmid, name="base plate", youngs_modulus=PERM_PLATE_E, poissons_ratio=PERM_PLATE_NU,
                                  density=PERM_PLATE_DENSITY)

    def new_node(xyz):
        nid = converter._next_node_id; converter._next_node_id += 1
        nodes[nid] = FENode(id=nid, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])); P[nid] = np.asarray(xyz, float)
        return nid

    def new_el(etype, ids, like, mat_id=None, thickness=None, offset=0.0, rebar=True):
        eid = converter._next_elem_id; converter._next_elem_id += 1
        e = FEElement(id=eid, element_type=etype, node_ids=list(ids), material_id=like.material_id if mat_id is None else mat_id,
                      thickness=float(like.thickness if thickness is None else thickness), offset=float(offset))
        if rebar:
            e.rebar_ratio = float(getattr(like, 'rebar_ratio', 0.0)); e.rebar_cover = float(getattr(like, 'rebar_cover', 0.0))
            e.steel_modulus = float(getattr(like, 'steel_modulus', 0.0))
        els[eid] = e
        return eid

    conn_node_set = getattr(converter, '_conn_nodes', None) or set()
    for ring in clusters.values():
        if conn_node_set and set(ring) & conn_node_set:
            continue                                   # an explicit connection is defined at this post
        info['posts'] += 1
        rset = set(ring)
        c0 = np.mean([P[i] for i in ring], axis=0)
        st2 = lambda p: np.array([(p[:2] - c0[:2]) @ ax, (p[:2] - c0[:2]) @ ac])
        adj_top = sorted({eid for i in ring for eid in node_el[i] if rows[eid][3] == 'top'})
        skirt = {eid for i in ring for eid in node_el[i] if rows[eid][3] in ('steel', 'thin')}
        skirt |= {eid for eid, r in rows.items() if r[3] == 'thin' and np.linalg.norm(r[1][:2] - c0[:2]) < 0.12}
        skirt = sorted(skirt); skirt_set = set(skirt)
        tmpl_pool = [eid for eid in adj_top if els[eid].thickness >= 0.02]
        if not tmpl_pool or not skirt:
            info['failed'] += 1; continue
        key = lambda eid: (els[eid].material_id, round(els[eid].thickness, 5), round(float(els[eid].offset) * np.sign(rows[eid][2][2]), 5))
        cnt = defaultdict(int)
        for eid in tmpl_pool:
            cnt[key(eid)] += 1
        best = max(cnt, key=cnt.get); tmpl = els[[eid for eid in tmpl_pool if key(eid) == best][0]]; d_tmpl = best[2]
        # hole boundary: ring-to-ring edges used by exactly one concrete face
        ecount = defaultdict(int)
        for eid in {eid for i in ring for eid in node_el[i] if rows[eid][3] in ('top', 'conc')}:
            ids = rows[eid][0]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                if a in rset and b in rset:
                    ecount[tuple(sorted((a, b)))] += 1
        nb = defaultdict(list)
        for (a, b), k in ecount.items():
            if k == 1:
                nb[a].append(b); nb[b].append(a)
        loop = []
        if nb and all(len(v) == 2 for v in nb.values()):
            start = next(iter(nb)); loop = [start]; prev = None; cur = start
            while True:
                nxt = [v for v in nb[cur] if v != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                if cur == start:
                    break
                loop.append(cur)
            if len(loop) != len(nb):
                loop = []
        if len(loop) < 3:                     # fall back: the ring ordered by angle
            ang = [np.arctan2(*st2(P[i])[::-1]) for i in ring]
            loop = [ring[k] for k in np.argsort(ang)]
        poly = np.array([st2(P[i]) for i in loop])
        if 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])) < 0:
            loop = loop[::-1]; poly = poly[::-1]
        A_poly = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
        # deck plane through the ring
        Mz = np.c_[np.ones(len(loop)), poly]; zz = np.array([P[i][2] for i in loop])
        coef = np.linalg.lstsq(Mz, zz, rcond=None)[0] if len(loop) >= 3 else np.array([zz.mean(), 0.0, 0.0])
        zpl = lambda q: float(coef[0] + coef[1] * q[0] + coef[2] * q[1])
        ax_3 = np.array([ax[0], ax[1], 0.0]); ac_3 = np.array([ac[0], ac[1], 0.0])
        to3 = lambda q: c0 + ax_3 * q[0] + ac_3 * q[1] + np.array([0.0, 0.0, zpl(q) - c0[2]])
        # the tube's bottom: the skirt nodes that are not ring nodes
        # the tube's bottom: nodes of the removed faces that the remaining tube faces also use
        keep_steel = lambda i: [e for e in node_el[i] if rows[e][3] == 'steel' and e not in skirt_set]
        inner = sorted({i for eid in skirt for i in rows[eid][0] if i not in rset and keep_steel(i)})
        inner_set = set(inner)
        orphans = sorted({i for eid in skirt for i in rows[eid][0]
                          if i not in rset and i not in inner_set and all(e in skirt_set for e in node_el[i])})
        inner_edges = set()
        for eid in skirt:
            ids = rows[eid][0]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                if a in inner_set and b in inner_set and set(keep_steel(a)) & set(keep_steel(b)):
                    inner_edges.add(tuple(sorted((a, b))))
        pts, owner = [q for q in poly], list(loop)         # fill vertices (plan) and their node ids (None = new)
        target = {}                                         # inner node -> fill vertex node (itself, a new node or a ring node)
        for i in inner:
            q = st2(P[i]); dz = P[i][2] - zpl(q)
            inside = _point_in_poly(q, poly) and _dist_to_poly(q, poly) >= 0.004
            if not inside:
                target[i] = loop[int(np.argmin(np.linalg.norm(poly - q, axis=1)))]
            elif dz < 0.006:
                target[i] = i; pts.append(q); owner.append(i)
            else:
                target[i] = ('new', len(pts)); pts.append(q); owner.append(None)
        # bottom-layer nodes under the footprint: fill vertices right above them (so the overhang ties reach them)
        if len(LP):
            near = np.nonzero(np.linalg.norm(LP[:, :2] - c0[:2], axis=1) < 0.20)[0]
            for k in near:
                q = st2(LP[k]); rise = zpl(q) - LP[k][2]
                if not (0.02 < rise <= PERM_TIE_MAX_RISE) or not _point_in_poly(q, poly) or _dist_to_poly(q, poly) < 0.006:
                    continue
                # no fill vertex yet -> nothing to be close to (a post can have no inner node before its
                # first bottom-layer node)
                _fv = np.array(pts[len(loop):], dtype=float).reshape(-1, 2)
                if len(_fv) and float(np.linalg.norm(_fv - q, axis=1).min()) < 0.008:
                    continue
                pts.append(q); owner.append(None)
        if len(pts) == len(loop):
            g = poly.mean(0)
            if _point_in_poly(g, poly):
                pts.append(g); owner.append(None)
        Pt = np.array(pts)
        tris = None
        try:
            tri = Delaunay(Pt)
            keep = [s for s in tri.simplices if _point_in_poly(Pt[s].mean(0), poly)]
            area = sum(0.5 * abs(_cross2(Pt[s[1]] - Pt[s[0]], Pt[s[2]] - Pt[s[0]])) for s in keep)
            edges = defaultdict(int)
            for s in keep:
                for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                    edges[tuple(sorted((a, b)))] += 1
            ok = abs(area - A_poly) <= 1e-6 + 1e-4 * A_poly and all(edges.get(tuple(sorted((k, (k + 1) % len(loop)))), 0) == 1
                                                                    for k in range(len(loop)))
            if ok:
                tris = keep
        except Exception:
            tris = None
        if tris is None:
            info['failed'] += 1
            print(f"[PERM] post base at ({c0[0]:+.3f}, {c0[1]:+.3f}): fill not built (post left as scanned)")
            continue
        # nodes of the fill
        nid_of = []
        for k, (q, o) in enumerate(zip(pts, owner)):
            nid_of.append(o if o is not None else new_node(to3(q)))
            if o is None:
                info['fill_nodes'] += 1
        for i, t in target.items():
            if isinstance(t, tuple):
                target[i] = nid_of[t[1]]
        fill_ids = []
        for s in tris:
            s = list(s)
            if _cross2(Pt[s[1]] - Pt[s[0]], Pt[s[2]] - Pt[s[0]]) < 0:
                s = [s[0], s[2], s[1]]
            ids = [nid_of[k] for k in s]
            n = _poly_normal(np.array([P[i] for i in ids]))
            eid = new_el(ElementType.TRI3, ids, tmpl, offset=d_tmpl * np.sign(n[2]))
            fill_ids.append(eid); converter._plate_fill.add(eid)
            converter._plate_sections[eid] = "post-base fill"
            info['fill_el'] += 1
            info['fill_area'] += 0.5 * abs(float(_cross2(Pt[s[1]] - Pt[s[0]], Pt[s[2]] - Pt[s[0]])))
        # tube walls carried down to the plate; a tube node within 4 mm of its plate node is tied to it instead
        tied = set()
        for i, t in target.items():
            if t != i and float(np.linalg.norm(P[i] - P[t])) < 0.004:
                converter._plate_ties.append((t, i)); tied.add(i)
        for a, b in sorted(inner_edges):
            ta, tb = target[a], target[b]
            ids = [a, b] + [v for v, src in ((tb, b), (ta, a)) if v != src and src not in tied]
            ids = list(dict.fromkeys(ids))
            if len(ids) < 3:
                continue
            ref = sorted(set(keep_steel(a)) & set(keep_steel(b)))
            like = els[ref[0]]; n_ref = rows[ref[0]][2]
            n_w = _poly_normal(np.array([P[i] for i in ids]))
            if float(n_w @ n_ref) < 0:
                ids = ids[::-1]
            et = ElementType.QUAD4 if len(ids) == 4 else ElementType.TRI3
            eid = new_el(et, ids, like, offset=float(getattr(like, 'offset', 0.0)), rebar=False)
            converter._plate_sections[eid] = sec_names.get(ref[0], "railing")
            info['walls'] += 1
        # nodes that only the removed faces used: tied to the nearest fill node (they carry nothing)
        if orphans:
            FN = np.array([P[i] for i in nid_of])
            for i in orphans:
                converter._plate_ties.append((nid_of[int(np.argmin(np.linalg.norm(FN - P[i], axis=1)))], i))
            info['orphans'] = info.get('orphans', 0) + len(orphans)
        for eid in skirt:                      # the scanned flare is replaced by the plate and the walls
            els[eid].viz_only = True
            info['skirt'] += 1
        # the base plate on the deck top
        sc = np.mean([st2(P[i]) for i in inner], axis=0) if inner else np.zeros(2)
        h = 0.5 * PERM_PLATE_SIZE
        need = float(np.max(np.abs(poly - sc))) + 0.010
        if need > h:
            h = need; info['enlarged'] += 1
        under = list(fill_ids)
        for eid, (ids, c, n, k) in rows.items():
            if k == 'top' and abs(c[2] - zpl(st2(c))) <= 0.05:
                q = st2(c) - sc
                if abs(q[0]) <= h and abs(q[1]) <= h:
                    under.append(eid)
        pl = []
        for eid in under:
            e = els[eid]
            ids = list(e.node_ids[:4 if getattr(e.element_type, 'name', '') == 'QUAD4' else 3])
            n = _poly_normal(np.array([P[i] for i in ids]))
            pe = new_el(e.element_type, ids, e, mat_id=pmid, thickness=PERM_PLATE_THICKNESS,
                        offset=0.5 * PERM_PLATE_THICKNESS * np.sign(n[2]), rebar=False)
            converter._plate_sections[pe] = "base plate"; pl.append(pe)
            Q = np.array([P[i] for i in ids])
            ar = 0.5 * float(np.linalg.norm(np.cross(Q[2] - Q[0], Q[3] - Q[1]) if len(Q) == 4 else np.cross(Q[1] - Q[0], Q[2] - Q[0])))
            info['plate_mass'] += PERM_PLATE_DENSITY * PERM_PLATE_THICKNESS * ar
        converter._plates.append(dict(centre=(c0 + ax_3 * sc[0] + ac_3 * sc[1]).tolist(), half=h, elements=pl,
                                          fill=fill_ids, ring=list(loop)))
        info['plate_el'] += len(pl); info['done'] += 1
    print(f"[PERM] post bases: {info['done']} of {info['posts']} posts on {PERM_PLATE_SIZE:.2f} m x "
          f"{PERM_PLATE_THICKNESS * 1e3:.0f} mm steel base plates ({info['plate_el']} plate shells, {info['enlarged']} enlarged "
          f"to cover the scanned base); deck holes filled with {info['fill_el']} triangles ({info['fill_area']:.4f} m2, "
          f"{info['fill_nodes']} new nodes); {info['skirt']} flare faces replaced by {info['walls']} tube-wall elements"
          + (f"; {len(converter._plate_ties)} tube nodes tied to the plate" if converter._plate_ties else "")
          + (f"; {info['failed']} posts left as scanned" if info['failed'] else ""))
    return info


class PERM_OT_RunSolver(bpy.types.Operator):
    """PERM: Run FEA with manually assigned supports and loads"""
    bl_idname = "perm.run_solver"
    bl_label = "PERM: Solve FEA"
    bl_options = {'REGISTER', 'UNDO'}

    youngs_modulus: bpy.props.FloatProperty(name="Young's Modulus (Pa)", default=30e9, min=1e6)
    poissons_ratio: bpy.props.FloatProperty(name="Poisson's Ratio", default=0.2, min=0.0, max=0.499)
    density: bpy.props.FloatProperty(name="Density (kg/m³)", default=2400.0, min=0.1)
    thickness: bpy.props.FloatProperty(name="Shell Thickness (m)", default=0.30, min=0.0001)
    # Mesh TOPOLOGY target handed to build_fe_model — it decides whether the mesh
    # is quadrangulated or triangulated before nodes/elements are created. It is
    # NOT the per-face element type: that is assigned in the Element Types panel
    # and applied after the model is built. Set from scn.perm_mesh_topology in
    # invoke(); the default matches the panel default.
    element_type: bpy.props.EnumProperty(name="Mesh topology", items=[
        ('QUAD4', "Quadrilateral (recommended)",
         "Quadrangulate the mesh — quads get the full bending shell "
         "(QM6 membrane + MITC4 bending + drilling)"),
        ('TRI3', "Triangular",
         "Triangulate the mesh — triangles get the CST + MITC3 shell; "
         "quads remain the better element"),
    ], default='QUAD4')
    support_type: bpy.props.EnumProperty(name="Support Type", items=[
        ('FIXED', "Fixed (clamped)", "Fix ux,uy,uz,θx,θy"),
        ('PINNED', "Pinned (simply-supported)", "Fix ux,uy,uz; rotations free"),
        ('ROLLER_X', "Roller X", "Free X"),
        ('ROLLER_Y', "Roller Y", "Free Y"),
        ('ROLLER_Z', "Roller Z", "Free Z")], default='PINNED')
    apply_gravity: bpy.props.BoolProperty(name="Apply Gravity", default=True)
    apply_point_load: bpy.props.BoolProperty(name="Apply Point Load", default=False)
    load_fx: bpy.props.FloatProperty(name="Force X (N)", default=0.0)
    load_fy: bpy.props.FloatProperty(name="Force Y (N)", default=0.0)
    load_fz: bpy.props.FloatProperty(name="Force Z (N)", default=-10000.0)
    viz_displacement: bpy.props.BoolProperty(name="Show Displacement", default=True)

    # ─── Stress visualization checkboxes (18 individual components) ───
    # Defaults: all ON for backward compatibility. Uncheck unwanted ones.
    # The descriptions are short to keep the dialog readable.

    # Von Mises (unsigned magnitude — uses jet colormap)
    viz_vm:       bpy.props.BoolProperty(name="Von Mises (max of top/bot)", default=True,
                  description="Von Mises stress, worse fiber. Unsigned (jet colormap).")
    viz_vm_top:   bpy.props.BoolProperty(name="Von Mises top fiber", default=True,
                  description="Von Mises stress at the top fiber.")
    viz_vm_bot:   bpy.props.BoolProperty(name="Von Mises bottom fiber", default=True,
                  description="Von Mises stress at the bottom fiber.")

    # Principal stresses at top fiber (signed — red=tension, blue=compression)
    viz_s1_top:   bpy.props.BoolProperty(name="σ1 top fiber (max principal)", default=True,
                  description="Max principal stress at top. Red=tension, blue=compression.")
    viz_s2_top:   bpy.props.BoolProperty(name="σ2 top fiber (min principal)", default=True,
                  description="Min principal stress at top.")

    # Principal stresses at bottom fiber
    viz_s1_bot:   bpy.props.BoolProperty(name="σ1 bottom fiber (max principal)", default=True,
                  description="Max principal stress at bottom. Most useful for RC — "
                              "shows tension where concrete will crack.")
    viz_s2_bot:   bpy.props.BoolProperty(name="σ2 bottom fiber (min principal)", default=True,
                  description="Min principal stress at bottom.")

    # Cartesian stresses at midplane (membrane only, no bending)
    viz_sx:       bpy.props.BoolProperty(name="σxx midplane", default=True,
                  description="Cartesian σ_xx at element midplane (membrane only).")
    viz_sy:       bpy.props.BoolProperty(name="σyy midplane", default=True,
                  description="Cartesian σ_yy at midplane.")
    viz_txy:      bpy.props.BoolProperty(name="τxy midplane", default=True,
                  description="Shear τ_xy at midplane.")

    # Cartesian stresses at top fiber
    viz_sx_top:   bpy.props.BoolProperty(name="σxx top fiber", default=True,
                  description="Cartesian σ_xx at top fiber (membrane + bending).")
    viz_sy_top:   bpy.props.BoolProperty(name="σyy top fiber", default=True,
                  description="Cartesian σ_yy at top fiber.")
    viz_txy_top:  bpy.props.BoolProperty(name="τxy top fiber", default=True,
                  description="Shear τ_xy at top fiber.")

    # Cartesian stresses at bottom fiber
    viz_sx_bot:   bpy.props.BoolProperty(name="σxx bottom fiber", default=True,
                  description="Cartesian σ_xx at bottom fiber.")
    viz_sy_bot:   bpy.props.BoolProperty(name="σyy bottom fiber", default=True,
                  description="Cartesian σ_yy at bottom fiber.")
    viz_txy_bot:  bpy.props.BoolProperty(name="τxy bottom fiber", default=True,
                  description="Shear τ_xy at bottom fiber.")

    # Bending moments (per unit length, N·m/m)
    viz_Mx:       bpy.props.BoolProperty(name="Bending moment Mxx", default=True,
                  description="Bending moment per unit length about y-axis.")
    viz_My:       bpy.props.BoolProperty(name="Bending moment Myy", default=True,
                  description="Bending moment per unit length about x-axis.")
    viz_Mxy:      bpy.props.BoolProperty(name="Twisting moment Mxy", default=True,
                  description="Twisting moment per unit length.")

    # Support reactions
    viz_reactions: bpy.props.BoolProperty(name="Support reactions (arrows + values)",
                  default=True,
                  description="Arrows for the reaction force components at every "
                              "supported node, with their values in kN")

    disp_scale: bpy.props.FloatProperty(name="Displacement Scale", default=100.0, min=0.1)

    def execute(self, context):
        # Always analyse the MODEL -- even when a result mesh, the camera or
        # nothing is active (the model itself is hidden after the first run).
        obj = _perm_resolve_source(context)
        if obj is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model) first"); return {'CANCELLED'}
        if _perm_need_scipy(self):
            return {'CANCELLED'}
        _perm_apply_scene_options(context.scene)       # connection details and report options of this scene
        _un = _perm_unit_note(context.scene)
        if _un:
            print("[PERM Solver] NOTE: " + _un)
        # Run from a script without these arguments, the panel decides (the dialog copies them in invoke).
        if not self.properties.is_property_set("apply_gravity"):
            self.apply_gravity = bool(getattr(context.scene, 'perm_apply_gravity', False))
        if not self.properties.is_property_set("element_type"):
            self.element_type = str(getattr(context.scene, 'perm_mesh_topology', 'QUAD4'))
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')     # flush edit-mode assignments
        context.scene["perm_last_source"] = obj.name
        for _k in ("perm_working_copy", "perm_source_name"):
            if _k in obj:
                del obj[_k]                     # this mesh is the model now
        if obj is not context.active_object:
            print(f"[PERM Solver] Analysing model '{obj.name}' "
                  f"(active object '{getattr(context.active_object, 'name', None)}' is not the model)")
        # Multi-group support assignment.
        # Backward compat: legacy "PERM_Supports" group exists → use the operator's
        # support_type setting. For named groups "PERM_Support_<name>", use the
        # group's own bc_type custom property.
        has_legacy_sup = "PERM_Supports" in obj.vertex_groups
        has_named_sup = any(vg.name.startswith("PERM_Support_") for vg in obj.vertex_groups)
        if not has_legacy_sup and not has_named_sup:
            self.report({'ERROR'}, "Assign at least one support group first"); return {'CANCELLED'}

        try:
            # face attribute with the mesh part of every face (only when the model has connections)
            part_attr = _perm_write_part_attr(obj)[0] if len(getattr(obj, 'perm_connections', []) or []) else None
            converter = PERMMeshConverter(obj)
            # bars on edges that belong to no face (a frame of beams) need their loose vertices kept
            converter.clean_mesh(remove_loose=not _perm_has_wire_bars(obj))
            etype_map = {'TRI3': ElementType.TRI3, 'QUAD4': ElementType.QUAD4}
            converter.build_fe_model(target_type=etype_map[self.element_type],
                                      thickness=self.thickness, scale=1.0)
            # ── node → vertex inverse map ────────────────────────────────────
            # The per-element material / section assignment below reads
            # `_node_to_vert` (FE node id → vertex index). Without it no thickness
            # or material from the scene Sections would reach the elements, and the
            # shells would keep build_fe_model's default thickness.
            if not hasattr(converter, '_node_to_vert') or not converter._node_to_vert:
                converter._node_to_vert = {nid: vid for vid, nid
                                           in converter._vert_to_node.items()}

            # Source-object vertex index -> FE node (survives clean_mesh merges).
            # Every vertex-group lookup below goes through these, never through
            # the working copy's own _vert_to_node / _node_to_vert.
            src_v2n, src_n2v, src_n2vs = _source_to_fe_maps(obj, converter)
            converter._src_vert_to_node = src_v2n

            # ── N-GON FACES PRODUCED NO ELEMENT ────────────────────
            # build_fe_model skips faces with more than four vertices. This is
            # NOT cosmetic: those faces contribute no element, so the structure
            # has holes there -- no stiffness, no mass, no load path across
            # them. convert_to_quads() merges triangles into quads but does NOT
            # dissolve n-gons, so it is reachable on the recommended QUAD4 path.
            # The assembly stage later stabilises the orphaned DOFs to keep the
            # matrix solvable, which hides the numerical symptom but not the
            # modelling gap -- so report it here, in the UI, where it is seen.
            skipped_ngons = list(getattr(converter, 'skipped_ngons', []) or [])
            if skipped_ngons:
                msg = (f"{len(skipped_ngons)} n-gon face(s) produced NO element - "
                       f"the model has holes there (no stiffness, no mass, no load "
                       f"path). Triangulate or quadrangulate them (Ctrl+T in Edit "
                       f"Mode), then re-run. See console for face indices.")
                print(f"[PERM Solver] WARNING: {msg}")
                self.report({'WARNING'}, msg)

            converter.materials[0] = MaterialProperty(id=0, name="User",
                youngs_modulus=self.youngs_modulus,
                poissons_ratio=self.poissons_ratio,
                density=self.density)

            # The SOLVER owns material assignment from here on. build_fe_model
            # may have set material_id from the object's Blender material slots,
            # and those ids index a materials dict this operator is about to
            # replace -- so an element could silently keep an id that now means
            # a different material, or none at all. Reset to the fallback and
            # assign explicitly, below, from PERM_Material_* groups only.
            for _elem in converter.elements.values():
                _elem.material_id = 0

            # ── COLLECT SUPPORTS (all named + legacy if present) ──
            support_groups_to_apply = []   # list of (node_set_name, BCType)
            _custom_dofs = {}              # node set of a CUSTOM support -> the DOFs it fixes
            if has_legacy_sup:
                vg_s = obj.vertex_groups["PERM_Supports"]
                sup_ids = set()
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == vg_s.index and v.index in src_v2n:
                            sup_ids.add(src_v2n[v.index]); break
                converter.node_sets['manual_supports'] = sup_ids
                support_groups_to_apply.append(('manual_supports', self.support_type))

            for vg in obj.vertex_groups:
                if not vg.name.startswith("PERM_Support_"):
                    continue
                suffix = vg.name[len("PERM_Support_"):]
                bc_type = obj.get(f"perm_support_{suffix}_type", "FIXED")
                ids = set()
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == vg.index and v.index in src_v2n:
                            ids.add(src_v2n[v.index]); break
                node_set_name = f"support_{suffix}"
                converter.node_sets[node_set_name] = ids
                support_groups_to_apply.append((node_set_name, bc_type))
                if bc_type == 'CUSTOM':
                    _custom_dofs[node_set_name] = perm_support_dofs(obj, suffix)

            # ── COLLECT LOADS (all named + legacy if present) ──
            # Each entry: dict with keys node_set_name, fx, fy, fz, and optional per_node
            load_groups_to_apply = []
            if "PERM_Loads" in obj.vertex_groups and self.apply_point_load:
                vg_l = obj.vertex_groups["PERM_Loads"]
                ids = set()
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == vg_l.index and v.index in src_v2n:
                            ids.add(src_v2n[v.index]); break
                converter.node_sets['manual_loads'] = ids
                load_groups_to_apply.append({
                    'node_set': 'manual_loads',
                    'fx': self.load_fx, 'fy': self.load_fy, 'fz': self.load_fz,
                    'per_node': None,
                })

            for vg in obj.vertex_groups:
                if not vg.name.startswith("PERM_Load_"):
                    continue
                suffix = vg.name[len("PERM_Load_"):]
                fx = float(obj.get(f"perm_load_{suffix}_fx", 0.0))
                fy = float(obj.get(f"perm_load_{suffix}_fy", 0.0))
                fz = float(obj.get(f"perm_load_{suffix}_fz", 0.0))
                per_node_json = get_per_node_forces(obj, suffix)  # may be empty
                if abs(fx) + abs(fy) + abs(fz) < 1e-15 and not per_node_json:
                    continue  # nothing to apply
                ids = set()
                for v in obj.data.vertices:
                    for g in v.groups:
                        if g.group == vg.index and v.index in src_v2n:
                            ids.add(src_v2n[v.index]); break
                node_set_name = f"load_{suffix}"
                converter.node_sets[node_set_name] = ids
                load_groups_to_apply.append({
                    'node_set': node_set_name,
                    'fx': fx, 'fy': fy, 'fz': fz,
                    'per_node': per_node_json if per_node_json else None,
                })

            # ── ADD TRUSS ELEMENTS ──
            # Convert vertex-group truss definitions into FEElement objects
            # appended to converter.elements with TRUSS_T2 type.
            # We support two storage formats (for backward compat):
            #   1. PERM_Truss_<n>          — old "Add Truss" workflow
            #   2. PERM_Element_<typename>_t<n> — new Element Types workflow
            truss_groups_old = find_perm_truss_groups(obj)
            truss_groups_new = find_perm_element_type_truss_groups(obj)
            # Combine: new format uses (type_name, group_name, area, verts);
            # old format uses (suffix, group_name, area, verts). Treat them uniformly.
            truss_groups = []
            for suffix, gname, area, verts in truss_groups_old:
                truss_groups.append((f"OLD_{suffix}", gname, area, verts))
            for type_name, gname, area, verts in truss_groups_new:
                truss_groups.append((type_name, gname, area, verts))
            # Edge trusses: one bar per tagged edge of the SOURCE mesh, mapped to
            # FE nodes by position (src_v2n), so merges in clean_mesh() and the
            # quadrangulation of the working copy cannot drop or move them.
            edge_trusses = []      # (label, n1, n2, type, material, section)
            _bar_src = {}          # bar label -> its two source vertices
            _etags = _read_edge_truss_tags(obj.data)
            if _etags is not None and _etags.any():
                _combos = _truss_combos(obj)
                for e in obj.data.edges:
                    cid = int(_etags[e.index])
                    if cid <= 0:
                        continue
                    t_name, m_name, s_name = _combos.get(cid, ("?", "", ""))
                    v1, v2 = e.vertices
                    edge_trusses.append((f"edge{e.index}", src_v2n.get(v1), src_v2n.get(v2),
                                         t_name, m_name, s_name))
                    _bar_src[f"edge{e.index}"] = (int(v1), int(v2))

            if truss_groups or edge_trusses:
                scn_sec = {pg.name: pg for pg in context.scene.perm_sections}
                if not hasattr(converter, 'sections') or converter.sections is None:
                    converter.sections = {}

                # Legacy vertex-group bars -> same tuple form
                all_bars = []
                for suffix, group_name, area, vert_indices in truss_groups:
                    rest = group_name[len("PERM_Element_"):] if group_name.startswith("PERM_Element_") else None
                    m_t = _TRUSS_GROUP_RE.match(rest) if rest else None
                    if m_t:
                        key = f"perm_truss_{m_t.group(1)}_t{m_t.group(2)}"
                        t_name = m_t.group(1)
                    else:                                   # oldest PERM_Truss_<n>
                        key = f"perm_truss_{group_name[len('PERM_Truss_'):]}"
                        t_name = "PERM_Truss"
                    _bar_src[group_name] = (int(vert_indices[0]), int(vert_indices[1]))
                    all_bars.append((group_name,
                                     src_v2n.get(vert_indices[0]), src_v2n.get(vert_indices[1]),
                                     t_name, str(obj.get(f"{key}_material", "")),
                                     str(obj.get(f"{key}_section", "")), float(area)))
                for lbl, n1, n2, t_name, m_name, s_name in edge_trusses:
                    all_bars.append((lbl, n1, n2, t_name, m_name, s_name, None))

                next_eid = max(converter.elements.keys(), default=0) + 1
                truss_count = n_unmapped = n_dupe = 0
                n_beams = 0
                _bar_kind = {et.name: et.kind for et in context.scene.perm_element_types}
                no_section, stale_area, no_material = [], [], []
                seen_pairs = set()
                for lbl, n1, n2, t_name, m_name, s_name, stored_area in all_bars:
                    if n1 is None or n2 is None or n1 == n2:
                        n_unmapped += 1
                        continue
                    pair = (min(n1, n2), max(n1, n2))
                    if pair in seen_pairs:          # coincident edges of separate parts
                        n_dupe += 1
                        continue
                    seen_pairs.add(pair)
                    # Area from the section as it is now, not as it was when the bar was assigned
                    if s_name in scn_sec:
                        area = float(scn_sec[s_name].area)
                        if stored_area is not None and abs(area - stored_area) > 1e-12 * max(1.0, area):
                            stale_area.append(lbl)
                    elif stored_area is not None:
                        area = stored_area           # legacy bar without section name
                    else:
                        no_section.append(lbl)
                        continue
                    if not m_name:
                        no_material.append(lbl)
                    _is_beam = _bar_kind.get(t_name) == 'BEAM_B2'
                    if s_name:
                        section_id = f"{'beam' if _is_beam else 'truss'}_sec_{s_name}"
                    else:
                        section_id = f"truss_{lbl}"
                    _sd = {'area': area, 'name': s_name or lbl}
                    if _is_beam:
                        if s_name not in scn_sec:
                            no_section.append(lbl)
                            continue
                        _sd['beam'] = _perm_beam_props(scn_sec[s_name])
                        _sd['area'] = _sd['beam']['A']
                    converter.sections[section_id] = _sd
                    converter.elements[next_eid] = FEElement(
                        id=next_eid,
                        element_type=ElementType.BEAM_B2 if _is_beam else ElementType.TRUSS_T2,
                        node_ids=[n1, n2],
                        material_id=0,               # resolved from material_name below
                        section_id=section_id,
                        material_name=m_name or None,
                        src_verts=_bar_src.get(lbl),
                    )
                    next_eid += 1
                    truss_count += 1
                    n_beams += int(_is_beam)
                # later FE-only elements (post base plates) must not reuse the bars' ids
                converter._next_elem_id = max(int(getattr(converter, '_next_elem_id', 1)), next_eid)
                print(f"[PERM Solver] Added {truss_count - n_beams} truss and {n_beams} beam elements "
                      f"({len(edge_trusses)} edge bar(s), {len(truss_groups)} legacy bar(s))")
                if n_unmapped:
                    print(f"[PERM Solver] WARNING: {n_unmapped} truss bar(s) skipped - "
                          f"end vertices not found or merged into one node")
                if n_dupe:
                    print(f"[PERM Solver] NOTE: {n_dupe} duplicate truss bar(s) on the same "
                          f"node pair ignored (coincident edges of separate parts)")
                if stale_area:
                    print(f"[PERM Solver] NOTE: {len(stale_area)} legacy bar(s) use the "
                          f"section's CURRENT area (changed since the bar was created)")
                if no_section:
                    msg = (f"{len(no_section)} truss bar(s) reference a section that is no "
                           f"longer in the Sections list - skipped (no area)")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)
                if truss_count and not context.scene.perm_materials:
                    msg = (f"{truss_count} truss bar(s) but no scene Materials: every bar "
                           f"uses the Run dialog's E/ν/ρ. Add materials (e.g. Steel).")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)
                if no_material:
                    msg = (f"{len(no_material)} legacy truss bar(s) were created before trusses "
                           f"stored a material and will use the FIRST material in the list. "
                           f"Re-assign them to set their material.")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)


            # ── APPLY SCENE MATERIALS to converter ──
            # Build scene materials → converter.materials dict (MaterialProperty objects)
            scn = context.scene
            if scn.perm_materials:
                # Replace converter.materials with scene-defined ones.
                # converter.materials currently has a default MaterialProperty at idx 0;
                # we keep that as the fallback at idx 0 and append scene materials at idx 1+.
                # But for the per-element id system to work cleanly, we just OVERWRITE
                # the materials dict with scene ones (indexing from 0).
                new_materials = {}
                name_to_id = {}
                for i, pg in enumerate(scn.perm_materials):
                    new_materials[i] = MaterialProperty(
                        id=i,
                        name=pg.name,
                        youngs_modulus=float(pg.youngs_modulus),
                        poissons_ratio=float(pg.poissons_ratio),
                        density=float(pg.density),
                    )
                    name_to_id[pg.name] = i
                converter.materials = new_materials
                print(f"[PERM Solver] Loaded {len(new_materials)} scene materials")

                # Index 0 is now whatever material happens to be FIRST in the
                # scene list -- and every element without a PERM_Material_*
                # group falls back to it. That is silent, and reordering the
                # list changes results, so name it and count it below.
                fallback_mat = new_materials[0].name if 0 in new_materials else "(none)"
                for _elem in converter.elements.values():
                    _elem.material_id = 0
                
                # Elements that received a material on purpose (vertex group or
                # face tag). Material #0 can be a deliberate choice, so the
                # fallback count below must not be judged by material_id alone.
                _explicit_mat = set()

                # Assign per-element material_id from PERM_Material_<name> vertex
                # groups -- legacy path, used only when the mesh has no face tags.
                material_groups = ([] if _has_face_tags(converter, 'material')
                                   else find_perm_material_groups(obj))
                if material_groups:
                    # Build vert → material_id map (last group wins)
                    vert_to_matid = {}
                    for suffix, group_name in material_groups:
                        if suffix not in name_to_id:
                            print(f"[PERM Solver] ⚠ Vertex group '{group_name}' references "
                                  f"material '{suffix}' which is not in the scene Materials list")
                            continue
                        mat_id = name_to_id[suffix]
                        vg = obj.vertex_groups[group_name]
                        for v in obj.data.vertices:
                            for g in v.groups:
                                if g.group == vg.index:
                                    vert_to_matid[v.index] = mat_id
                                    break
                    # Apply to each shell element: if all its verts share a material_id, use it
                    n_assigned = 0
                    for eid, elem in converter.elements.items():
                        if _perm_is_line(elem):
                            continue
                        # Map FE node ids back to vertex indices
                        vert_indices = []
                        for nid in elem.node_ids:
                            vidx = src_n2v.get(nid)
                            if vidx is not None:
                                vert_indices.append(vidx)
                        if not vert_indices:
                            continue
                        mat_ids = {vert_to_matid.get(vi) for vi in vert_indices}
                        mat_ids.discard(None)
                        if len(mat_ids) == 1:
                            elem.material_id = mat_ids.pop()
                            _explicit_mat.add(eid)
                            n_assigned += 1
                    print(f"[PERM Solver] Assigned per-element material_id to "
                          f"{n_assigned} shell elements")

                # Truss bars: each carries its own material NAME (never a face's).
                _tr_missing = Counter()
                _tr_count = Counter()
                for eid, elem in converter.elements.items():
                    if not _perm_is_line(elem):
                        continue
                    mname = getattr(elem, 'material_name', None)
                    if mname and mname in name_to_id:
                        elem.material_id = name_to_id[mname]
                        _tr_count[mname] += 1
                    elif mname:
                        _tr_missing[mname] += 1
                if _tr_count:
                    print("[PERM Solver] Truss materials: " +
                          ", ".join(f"{k}={v}" for k, v in _tr_count.items()))
                for mname, n in _tr_missing.items():
                    msg = (f"{n} truss bar(s) use material '{mname}', which is not in the "
                           f"Materials list (renamed or deleted) - they fall back to "
                           f"'{fallback_mat}'")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)

                # Per-face tags are authoritative where present. They do not
                # depend on vertex membership, so elements on the line where two
                # parts meet get the material of the face they came from.
                face_mat = _element_face_tag_names(converter, 'material')
                if face_mat:
                    n_tag = n_missing = 0
                    for eid, mname in face_mat.items():
                        elem = converter.elements.get(eid)
                        if elem is None or _perm_is_line(elem):
                            continue
                        if mname in name_to_id:
                            elem.material_id = name_to_id[mname]
                            _explicit_mat.add(eid)
                            n_tag += 1
                        else:
                            n_missing += 1
                    print(f"[PERM Solver] Face tags: material set on {n_tag} shell elements")
                    if n_missing:
                        msg = (f"{n_missing} face(s) are tagged with a material that is "
                               f"no longer in the Materials list (renamed or deleted)")
                        print(f"[PERM Solver] WARNING: {msg}")
                        self.report({'WARNING'}, msg)

                n_fallback = sum(
                    1 for e in converter.elements.values()
                    if not _perm_is_line(e)
                    and not getattr(e, 'viz_only', False)
                    and e.id not in _explicit_mat)
                if not n_fallback:
                    print(f"[PERM Solver] Every shell element has an explicit material "
                          f"(fallback '{fallback_mat}' unused)")
                elif len(new_materials) == 1:
                    # One material in the list: the fallback is unambiguous and
                    # there is nothing for the user to decide. Console only.
                    print(f"[PERM Solver] {n_fallback} shell element(s) use the only "
                          f"scene material, '{fallback_mat}'")
                else:
                    # Two or more: which material an unassigned face gets depends
                    # on LIST ORDER, which is not a modelling decision anyone made.
                    msg = (f"{n_fallback} shell element(s) have no PERM_Material_* "
                           f"assignment and fall back to '{fallback_mat}' - the first "
                           f"of {len(new_materials)} materials. Reordering the "
                           f"Materials panel would change their stiffness and mass.")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)
            
            # ── APPLY SCENE SECTIONS (thickness) to shell elements ──
            _sec_assigned = set()   # shell element ids that received a Section
            if scn.perm_sections:
                # Legacy vertex groups only when the mesh has no section face tags.
                section_groups = ([] if _has_face_tags(converter, 'section')
                                  else find_perm_section_groups(obj))
                face_sec = _element_face_tag_names(converter, 'section')
                if section_groups or face_sec:
                    # Build name → section data
                    section_data = {pg.name: pg for pg in scn.perm_sections}
                    vert_to_section = {}
                    for suffix, group_name in section_groups:
                        if suffix not in section_data:
                            print(f"[PERM Solver] ⚠ Vertex group '{group_name}' references "
                                  f"section '{suffix}' not in scene Sections list")
                            continue
                        vg = obj.vertex_groups[group_name]
                        for v in obj.data.vertices:
                            for g in v.groups:
                                if g.group == vg.index:
                                    vert_to_section[v.index] = suffix
                                    break
                    n_assigned = 0
                    applied = {}   # section name -> thickness it applied
                    for eid, elem in converter.elements.items():
                        if _perm_is_line(elem):
                            continue
                        vert_indices = []
                        for nid in elem.node_ids:
                            vidx = src_n2v.get(nid)
                            if vidx is not None:
                                vert_indices.append(vidx)
                        tagged = face_sec.get(eid)
                        if tagged is not None and tagged in section_data:
                            sec_names = {tagged}
                        elif not vert_indices:
                            continue
                        else:
                            sec_names = {vert_to_section.get(vi) for vi in vert_indices}
                            sec_names.discard(None)
                        if len(sec_names) == 1:
                            sec_name = sec_names.pop()
                            sec_pg = section_data[sec_name]
                            sec_t = float(sec_pg.thickness)
                            elem.thickness = sec_t
                            # Smeared reinforcement (linear transformed section).
                            # 0 ρ / 0 E_s ⇒ plain concrete, no stiffness change.
                            elem.rebar_ratio   = float(getattr(sec_pg, 'rebar_ratio', 0.0))
                            elem.rebar_cover   = float(getattr(sec_pg, 'rebar_cover', 0.0))
                            elem.steel_modulus = float(getattr(sec_pg, 'steel_modulus', 0.0))
                            applied[sec_name] = sec_t
                            _sec_assigned.add(eid)
                            n_assigned += 1
                    if applied:
                        detail = ", ".join(f"{s}={t:.6g} m" for s, t in applied.items())
                        print(f"[PERM Solver] Applied per-element thickness from "
                              f"scene Section(s) to {n_assigned} shell elements: "
                              f"{detail}")
                        # Report any reinforced-concrete sections that are active.
                        rc_secs = {s: section_data[s] for s in applied
                                   if float(getattr(section_data[s], 'rebar_ratio', 0.0)) > 0.0
                                   and float(getattr(section_data[s], 'steel_modulus', 0.0)) > 0.0}
                        for s, pg in rc_secs.items():
                            _mi = min(max(scn.perm_material_index, 0), len(scn.perm_materials) - 1)
                            E_c = float(scn.perm_materials[_mi].youngs_modulus) \
                                  if len(scn.perm_materials) else 30e9
                            n_mod = float(pg.steel_modulus) / E_c if E_c > 0 else 0.0
                            rho = float(pg.rebar_ratio)
                            f_m = 1.0 + (n_mod - 1.0) * rho
                            lr = 1.0 - 2.0 * min(float(pg.rebar_cover), 0.5 * float(pg.thickness)) / float(pg.thickness)
                            f_b = 1.0 + 3.0 * (n_mod - 1.0) * rho * lr * lr
                            print(f"[PERM Solver] RC section '{s}': ρ={rho:.4f}, "
                                  f"c (face→bar centre)={float(pg.rebar_cover):.3f} m, n=E_s/E_c≈{n_mod:.1f} "
                                  f"→ transformed stiffness ×{f_m:.3f} membrane, "
                                  f"×{f_b:.3f} bending.")
                        if any(abs(t - self.thickness) > 1e-12 for t in applied.values()):
                            self.report({'INFO'},
                                f"Shell thickness taken from scene Section(s): {detail} "
                                f"(default thickness {self.thickness:.4g} m not used for those faces)")
                    elif section_groups:
                        msg = (f"Section vertex group(s) present but no element "
                               f"thickness was assigned; shells keep default "
                               f"{self.thickness} m. Check section names/assignments.")
                        print(f"[PERM Solver] WARNING: {msg}")
                        self.report({'WARNING'}, msg)

            # ── SHELL OFFSETS: mid-surface moved off the scanned surface ──
            # After the sections, because the report compares each offset with
            # the element's final thickness.
            _perm_apply_shell_offsets(converter)

            # ── FACES WITHOUT A CROSS-SECTION ──
            # Their thickness is the Run operator's default (self.thickness), NOT
            # the thickness typed in "Convert to Articulated Data" -- that one is
            # used only by Convert's own model and its exports. Count them and
            # warn; the solve continues.
            _shells = [eid for eid, e in converter.elements.items()
                       if not _perm_is_line(e)
                       and not getattr(e, 'viz_only', False)]
            _n_no_sec = sum(1 for eid in _shells if eid not in _sec_assigned)
            try:
                _nsec_src, _nmat_src = _perm_unassigned_faces(obj)
                context.scene["perm_unassigned_counts"] = [int(_nsec_src.sum()), int(_nmat_src.sum())]
            except Exception:
                pass
            if _n_no_sec:
                msg = (f"{_n_no_sec} of {len(_shells)} shell face(s) have no "
                       f"Cross-Section assigned and use the default thickness "
                       f"{self.thickness:g} m")
                print(f"[PERM Solver] WARNING: {msg}")
                self.report({'WARNING'}, msg)
            else:
                print(f"[PERM Solver] All {len(_shells)} shell faces have a Cross-Section")

            # ── APPLY SCENE ELEMENT TYPES to shell elements ──
            # Faces assigned in the Element Types panel via PERM_Element_<name>
            # vertex groups. Only the shell kinds are handled here; TRUSS_T2
            # instances were already converted above from the _t<n> groups.
            #
            # Kind → effect:
            #   QUAD4_MITC4    full bending shell (QM6 membrane + MITC4 + drilling)
            #   QUAD4_MEMBRANE membrane-only quad — hinge out of plane (deliberate)
            #   QUAD4_PLATE    MITC4 + drilling, no membrane — transverse load only
            #   TRI3           shell triangle: CST + MITC3 + drilling
            #
            # A face is assigned only when ALL of its vertices belong to the group,
            # matching how the assignment operator writes them. Unassigned faces
            # keep the default: full bending shell (MITC4 quads, MITC3 triangles).
            # Face tags are authoritative once the mesh has them; the vertex-
            # group rule ("all corners in the group") is the legacy fallback. It
            # is ambiguous on boundaries, where corner vertices sit in two groups.
            use_et_tags = _has_face_tags(converter, 'elemtype')
            shell_type_groups = {} if use_et_tags else find_perm_element_type_shell_groups(obj)
            face_et = _element_face_tag_names(converter, 'elemtype') if use_et_tags else {}

            # (eid, type_name) pairs to apply, from whichever source is active
            et_pairs = list(face_et.items())
            for type_name, vert_set in shell_type_groups.items():
                for eid, elem in converter.elements.items():
                    if (_perm_is_line(elem)
                            or getattr(elem, 'viz_only', False)):
                        continue
                    copies = [src_n2vs.get(nid) for nid in elem.node_ids]
                    if any(not c for c in copies):
                        continue
                    if all(any(v in vert_set for v in c) for c in copies):
                        et_pairs.append((eid, type_name))

            if et_pairs:
                kind_of = {et.name: et.kind for et in scn.perm_element_types}
                n_bending = n_membrane = n_plate = 0
                n_topology_mismatch = 0
                unknown_types = set()
                for eid, type_name in et_pairs:
                    if type_name not in kind_of:
                        unknown_types.add(type_name)
                        continue
                    kind = kind_of[type_name]
                    if kind in _PERM_LINE_TYPES:
                        continue
                    elem = converter.elements.get(eid)
                    if elem is None or getattr(elem, 'viz_only', False):
                        continue
                    etype_name = getattr(elem.element_type, 'name', '')
                    # Guard: a quad kind on a triangular face (or the reverse) is a
                    # user error. Report it rather than apply a mismatched setting.
                    wants_quad = kind in ('QUAD4_MITC4', 'QUAD4_MEMBRANE', 'QUAD4_PLATE')
                    if wants_quad and etype_name != 'QUAD4':
                        n_topology_mismatch += 1
                        continue
                    if kind == 'TRI3' and etype_name != 'TRI3':
                        n_topology_mismatch += 1
                        continue
                    # The three quad kinds are mutually exclusive: set both flags
                    # every time so a re-assignment cannot leave a stale marker.
                    if kind == 'QUAD4_MEMBRANE':
                        elem.membrane_only, elem.bending_only = True, False
                        n_membrane += 1
                    elif kind == 'QUAD4_PLATE':
                        elem.membrane_only, elem.bending_only = False, True
                        n_plate += 1
                    elif kind == 'QUAD4_MITC4':
                        elem.membrane_only, elem.bending_only = False, False
                        n_bending += 1
                if n_bending or n_membrane or n_plate:
                    print(f"[PERM Solver] Element types from panel "
                          f"({'face tags' if use_et_tags else 'vertex groups'}): "
                          f"{n_bending} bending shell, {n_membrane} membrane-only, "
                          f"{n_plate} plate (MITC4-only).")
                if n_plate:
                    msg = (f"{n_plate} face(s) assigned Plate (MITC4 only): no "
                           f"in-plane stiffness. Correct for a flat plate under "
                           f"transverse load; wrong for folded structures, where "
                           f"bending is carried by membrane action of the flanges.")
                    print(f"[PERM Solver] NOTE: {msg}")
                    self.report({'WARNING'}, msg)
                for t in sorted(unknown_types):
                    msg = (f"Element type '{t}' is assigned to faces but is not in the "
                           f"scene Element Types list")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)
                if n_topology_mismatch:
                    msg = (f"{n_topology_mismatch} face(s) have an element type whose "
                           f"topology does not match (quad type on a triangle or vice "
                           f"versa) — those faces kept their default")
                    print(f"[PERM Solver] WARNING: {msg}")
                    self.report({'WARNING'}, msg)

            # Coverage report: how many shell faces carry an explicit assignment.
            # Printed rather than drawn in the panel, because computing it means
            # walking every vertex group and would make the panel redraw slow on
            # a large captured mesh.
            n_shell_total = sum(
                1 for e in converter.elements.values()
                if not _perm_is_line(e)
                and not getattr(e, 'viz_only', False))
            n_mat_groups = len(find_perm_material_groups(obj))
            n_sec_groups = len(find_perm_section_groups(obj))
            _n_face_mat = len(_element_face_tag_names(converter, 'material'))
            _n_face_sec = len(_element_face_tag_names(converter, 'section'))
            if _n_face_mat or _n_face_sec:
                print(f"[PERM Solver] Face-tag coverage: material {_n_face_mat}/"
                      f"{n_shell_total}, section {_n_face_sec}/{n_shell_total} "
                      f"shell elements")
            print(f"[PERM Solver] Model coverage: {n_shell_total} shell elements | "
                  f"{n_mat_groups} material group(s), {n_sec_groups} section group(s), "
                  f"{len(shell_type_groups)} element-type group(s)")

            try:
                _perm_build_connections(converter, obj, src_v2n, part_attr)   # joints between mesh parts
            except Exception as exc:
                import traceback; traceback.print_exc()
                raise RuntimeError(f"PERM connections failed: {exc}")
            try:
                _perm_build_post_bases(converter)                     # post base plates (before the solver numbers the nodes)
            except Exception as exc:
                import traceback; traceback.print_exc()
                raise RuntimeError(f"PERM post base plates failed: {exc}")
            solver = PERMSolver(converter)
            try:
                _perm_model_inventory(solver, context.scene)
            except Exception as e:
                print(f"[PERM Solver] Inventory failed: {e}")
            bc_map = {'FIXED': BCType.FIXED, 'PINNED': BCType.PINNED,
                      'ROLLER_X': BCType.ROLLER_X, 'ROLLER_Y': BCType.ROLLER_Y,
                      'ROLLER_Z': BCType.ROLLER_Z}
            # Apply ALL support groups
            for node_set_name, bc_type_str in support_groups_to_apply:
                if bc_type_str == 'CUSTOM':                # any set of the six DOFs, all held at zero
                    _d = _custom_dofs.get(node_set_name) or []
                    if not _d:
                        self.report({'WARNING'}, f"Restraint '{node_set_name.replace('support_', '')}' is CUSTOM "
                                                 "with no degree of freedom ticked - it restrains nothing")
                    solver.add_boundary_condition(f"Support_{node_set_name}", BCType.CUSTOM, node_set_name,
                                                  prescribed_values={d: 0.0 for d in _d})
                    continue
                if bc_type_str not in bc_map:
                    raise ValueError(f"Restraint '{node_set_name.replace('support_', '')}' has the unknown type "
                                     f"'{bc_type_str}'. Assign it again in the panel")
                solver.add_boundary_condition(
                    f"Support_{node_set_name}", bc_map[bc_type_str], node_set_name)
            try:
                _perm_check_supports(solver, support_groups_to_apply)
            except Exception as e:
                print(f"[PERM Solver] Support check failed: {e}")

            # Gravity
            if self.apply_gravity:
                # Use scene-level gravity if set; otherwise use operator's default.
                # Once we phase out the operator's apply_gravity, this will be the
                # primary path.
                scn = context.scene
                gz_scene = float(getattr(scn, 'perm_gravity_z', -9.81))
                solver.add_gravity(gx=float(getattr(scn, 'perm_gravity_x', 0.0)),      # free direction
                                   gy=float(getattr(scn, 'perm_gravity_y', 0.0)), gz=gz_scene)

            # surface loads on faces (force per area)
            _n_sl = _perm_apply_surface_loads(solver)
            if _n_sl:
                print(f"[PERM Solver] Surface load on {_n_sl} elements")

            # Apply ALL load groups
            for lg in load_groups_to_apply:
                if lg['per_node']:
                    solver.add_nodal_force_per_node(
                        lg['node_set'], lg['per_node'],
                        name=f"PerNode_{lg['node_set']}")
                else:
                    solver.add_nodal_force(
                        lg['node_set'], fx=lg['fx'], fy=lg['fy'], fz=lg['fz'])
            # Where the point loads act (world coordinates, a/L, e) -> console
            # and Text Editor 'PERM_Loads.txt', for hand calculations.
            try:
                _perm_report_object_loads(obj)
            except Exception as e:
                print(f"[PERM Loads] coordinate report failed: {e}")

            results = solver.solve()
            try:
                _perm_support_resultants(solver, support_groups_to_apply)
            except Exception as e:
                print(f"[PERM Solver] Support resultants failed: {e}")
            _perm_report_peaks(solver, context.scene, obj)
            try:                                   # element numbers on the faces (Find Element)
                _perm_tag_element_ids(solver)
            except Exception as _e_tag:
                print(f"[PERM Solver] element numbers not stored on the faces: {_e_tag}")
            # everything needed for the hand-calculation comparison -> one report file
            try:
                _rp = _perm_write_results_report(solver, context.scene, obj, support_groups_to_apply)
                if _rp:
                    self.report({'INFO'}, f"Results report saved: {_rp}")
            except Exception as _e_rep:
                import traceback; traceback.print_exc()
                print(f"[PERM Results] report could not be written: {_e_rep}")

            # Stash the solver in __main__ so PERM_OT_RefreshVisualization
            # can find it later (to re-render at a different scale without re-solving)
            import __main__
            __main__._PERM_LAST_SOLVER = solver

            if math.isnan(results.max_displacement) or math.isnan(results.max_von_mises):
                self.report({'WARNING'}, "NaN — check supports/constraints")
            else:
                if self.viz_displacement:
                    # Use scene-level scale if set, otherwise the operator's own scale.
                    # The Analysis panel sets the scene scale; the operator dialog uses its own.
                    scn = bpy.context.scene
                    scale = float(getattr(scn, 'perm_deformation_scale', self.disp_scale))
                    if scale <= 0:
                        scale = self.disp_scale
                    solver.visualize_displacement('magnitude',
                                                   scale_factor=scale,
                                                   deformed=True)
                # Map each checkbox prop to its stress component name.
                # Skip a component if its checkbox is off.
                viz_map = [
                    ('viz_vm',      'von_mises'),
                    ('viz_vm_top',  'von_mises_top'),
                    ('viz_vm_bot',  'von_mises_bot'),
                    ('viz_s1_top',  'sigma1_top'),
                    ('viz_s2_top',  'sigma2_top'),
                    ('viz_s1_bot',  'sigma1_bot'),
                    ('viz_s2_bot',  'sigma2_bot'),
                    ('viz_sx',      'sx'),
                    ('viz_sy',      'sy'),
                    ('viz_txy',     'txy'),
                    ('viz_sx_top',  'sx_top'),
                    ('viz_sy_top',  'sy_top'),
                    ('viz_txy_top', 'txy_top'),
                    ('viz_sx_bot',  'sx_bot'),
                    ('viz_sy_bot',  'sy_bot'),
                    ('viz_txy_bot', 'txy_bot'),
                    ('viz_Mx',      'Mx'),
                    ('viz_My',      'My'),
                    ('viz_Mxy',     'Mxy'),
                ]
                n_created = 0
                for prop_name, comp_name in viz_map:
                    if getattr(self, prop_name, False):
                        try:
                            solver.visualize_stress(comp_name)
                            n_created += 1
                        except Exception as e:
                            print(f"[PERM Solver] Could not visualize '{comp_name}': {e}")
                if n_created > 0:
                    print(f"[PERM Solver] Created {n_created} stress visualizations.")
                if self.viz_reactions:
                    try:
                        solver.visualize_reactions(
                            size_scale=float(getattr(context.scene, 'perm_reaction_scale', 1.0)))
                    except Exception as e:
                        print(f"[PERM Solver] Could not visualize reactions: {e}")
                # The working copy only feeds the result meshes: keep it out of view
                try:
                    converter.working_object.hide_set(True)
                except Exception:
                    pass
                self.report({'INFO'},
                    f"PERM: '{obj.name}' solved - max disp={results.max_displacement*1000:.3f}mm, "
                    f"VM={results.max_von_mises/1e6:.3f}MPa")

        except Exception as e:
            self.report({'ERROR'}, f"PERM Error: {str(e)}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        # Preload settings from scene properties so the operator has the right
        # gravity / point-load state at execute time. The dialog itself shows
        # ONLY the visualization checkboxes — inputs (material, gravity, loads)
        # live in the panel's input sections.
        scn = context.scene
        # Self-weight (gravity) — read from scene
        self.apply_gravity = bool(getattr(scn, 'perm_apply_gravity', False))
        # Mesh topology target — read from the Element Types panel
        self.element_type = str(getattr(scn, 'perm_mesh_topology', 'QUAD4'))
        # Point load of the single PERM_Loads group (older files): off. Loads are
        # assigned per load group in the panel.
        self.apply_point_load = False
        self.load_fx = 0.0
        self.load_fy = 0.0
        self.load_fz = 0.0
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, layout):
        layout = self.layout
        layout.label(text="Confirm which results to visualize:",
                     icon='INFO')
        layout.label(text="(Materials, BCs, loads, gravity are taken from the panel)")
        
        box = layout.box(); box.label(text="Displacement", icon='COLOR')
        box.prop(self, "viz_displacement")

        box = layout.box(); box.label(text="Support reactions", icon='EMPTY_SINGLE_ARROW')
        box.prop(self, "viz_reactions")

        box = layout.box(); box.label(text="Stress Fields (check what you want):",
                                       icon='COLOR')

        # Von Mises group
        col = box.column(align=True)
        col.label(text="Von Mises (unsigned):", icon='IPO_LINEAR')
        col.prop(self, "viz_vm")
        col.prop(self, "viz_vm_top")
        col.prop(self, "viz_vm_bot")

        # Principal stress group
        col = box.column(align=True)
        col.label(text="Principal stresses (signed, red=tension):", icon='IPO_QUAD')
        col.prop(self, "viz_s1_top")
        col.prop(self, "viz_s2_top")
        col.prop(self, "viz_s1_bot")
        col.prop(self, "viz_s2_bot")

        # Cartesian midplane group
        col = box.column(align=True)
        col.label(text="Cartesian stresses at midplane:", icon='AXIS_TOP')
        col.prop(self, "viz_sx")
        col.prop(self, "viz_sy")
        col.prop(self, "viz_txy")

        # Cartesian top
        col = box.column(align=True)
        col.label(text="Cartesian stresses at top fiber:", icon='TRIA_UP')
        col.prop(self, "viz_sx_top")
        col.prop(self, "viz_sy_top")
        col.prop(self, "viz_txy_top")

        # Cartesian bottom
        col = box.column(align=True)
        col.label(text="Cartesian stresses at bottom fiber:", icon='TRIA_DOWN')
        col.prop(self, "viz_sx_bot")
        col.prop(self, "viz_sy_bot")
        col.prop(self, "viz_txy_bot")

        # Moments
        col = box.column(align=True)
        col.label(text="Bending moments (N·m/m):", icon='CON_ROTLIMIT')
        col.prop(self, "viz_Mx")
        col.prop(self, "viz_My")
        col.prop(self, "viz_Mxy")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DEFORMATION-SCALE  (slider drives the existing mesh in place)
# ─────────────────────────────────────────────────────────────────────────────
#
# A displacement-viz object stores, on its mesh, two per-vertex FLOAT_VECTOR
# attributes: the undeformed (rest) local coords and the unscaled local-frame
# displacement vector. The deformed shape is then simply  co = rest + scale·disp.
# This lets the Deformation Scale slider rescale the SAME mesh live — no re-solve
# and no new object created — instead of rebuilding the mesh on every change.

PERM_DISP_REST_ATTR = "perm_rest_co"   # POINT FLOAT_VECTOR — undeformed local coords
PERM_DISP_VEC_ATTR  = "perm_disp_vec"  # POINT FLOAT_VECTOR — unscaled displacement


def _store_disp_attributes(mesh, rest_co, disp_local):
    """Persist rest coords + per-vertex displacement on the mesh as POINT
    FLOAT_VECTOR attributes. rest_co, disp_local: (n_verts, 3) numpy arrays."""
    for name in (PERM_DISP_REST_ATTR, PERM_DISP_VEC_ATTR):
        if name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[name])
    a_rest = mesh.attributes.new(PERM_DISP_REST_ATTR, 'FLOAT_VECTOR', 'POINT')
    a_disp = mesh.attributes.new(PERM_DISP_VEC_ATTR, 'FLOAT_VECTOR', 'POINT')
    a_rest.data.foreach_set("vector", np.ascontiguousarray(rest_co, dtype=np.float64).ravel())
    a_disp.data.foreach_set("vector", np.ascontiguousarray(disp_local, dtype=np.float64).ravel())


def _apply_disp_scale_to_object(obj, scale):
    """Set deformed coords = rest + scale·disp from the stored attributes.
    Returns True if applied, False if the object carries no PERM displacement data."""
    if obj is None or obj.type != 'MESH':
        return False
    mesh = obj.data
    if (PERM_DISP_REST_ATTR not in mesh.attributes or
            PERM_DISP_VEC_ATTR not in mesh.attributes):
        return False
    n = len(mesh.vertices)
    rest = np.empty(n * 3, dtype=np.float64)
    disp = np.empty(n * 3, dtype=np.float64)
    mesh.attributes[PERM_DISP_REST_ATTR].data.foreach_get("vector", rest)
    mesh.attributes[PERM_DISP_VEC_ATTR].data.foreach_get("vector", disp)
    mesh.vertices.foreach_set("co", rest + scale * disp)
    mesh.update()
    return True


# ── RESULT OVERLAY OPACITY ─────────────────────────────────────
#
# In Solid shading with color_type='VERTEX' Blender draws ONLY the colour
# attribute: the object's texture is not rendered at all, so no alpha value
# can reveal it. Showing both means putting the result colour into a real
# MATERIAL and viewing in Material Preview.
#
# The material intercepts whatever the source material already fed to Material
# Output and mixes it with an Emission of the result colour:
#
#     [existing shader, texture and all] ─┐
#                                         ├─ Mix Shader (Fac = opacity) ─ Output
#     [Color Attribute] ─ Emission ──────┘
#
# Intercepting at the OUTPUT rather than at Base Color means the source
# material can be arbitrarily complex and still survives intact. Emission
# rather than Base Color for the overlay means that at opacity 1.0 the colours
# read exactly true (unlit, unshaded) instead of being darkened by scene
# lighting.

PERM_MIX_NODE = "PERM_ResultMix"
PERM_EMIT_NODE = "PERM_ResultEmission"
PERM_ATTR_NODE = "PERM_ResultColor"


def perm_build_result_material(obj, layer_name: str) -> bool:
    """Give a result object a material that mixes its texture with the colour map.

    Returns True if a material was built. The object's own material is COPIED,
    never modified: the result viz is a throwaway duplicate and must not alter
    the material on the mesh the user actually keeps.
    """
    try:
        mesh = obj.data
        base = mesh.materials[0] if len(mesh.materials) else None
        if base is not None:
            mat = base.copy()
            mat.name = f"PERM_Result_{layer_name}"
        else:
            mat = bpy.data.materials.new(f"PERM_Result_{layer_name}")
        mat.use_nodes = True
        mat["perm_result_material"] = True
        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links

        # A previous build may already be in here if the source material came
        # from an earlier viz. Strip ours out and reconnect what it wrapped.
        for name in (PERM_MIX_NODE, PERM_EMIT_NODE, PERM_ATTR_NODE):
            old = nodes.get(name)
            if old is not None:
                nodes.remove(old)

        out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'
                    and n.is_active_output), None)             or next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if out is None:
            out = nodes.new('ShaderNodeOutputMaterial')
            out.location = (600, 0)

        surf = out.inputs['Surface']
        if surf.is_linked:
            orig_socket = surf.links[0].from_socket
        else:
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (out.location.x - 400, out.location.y - 200)
            orig_socket = bsdf.outputs[0]

        attr = nodes.new('ShaderNodeVertexColor')
        attr.name = attr.label = PERM_ATTR_NODE
        attr.layer_name = layer_name
        attr.location = (out.location.x - 620, out.location.y + 260)

        emit = nodes.new('ShaderNodeEmission')
        emit.name = emit.label = PERM_EMIT_NODE
        emit.location = (out.location.x - 400, out.location.y + 240)
        links.new(attr.outputs['Color'], emit.inputs['Color'])

        mix = nodes.new('ShaderNodeMixShader')
        mix.name = mix.label = PERM_MIX_NODE
        mix.location = (out.location.x - 200, out.location.y + 60)
        # Fac = 0 -> the original textured material; Fac = 1 -> pure colour map.
        links.new(orig_socket, mix.inputs[1])
        links.new(emit.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs[0], surf)

        opacity = 1.0
        try:
            opacity = float(getattr(bpy.context.scene, 'perm_result_opacity', 1.0))
        except Exception:
            pass
        mix.inputs[0].default_value = opacity

        mesh.materials.clear()
        mesh.materials.append(mat)
        return True
    except Exception as e:
        print(f"[PERM Solver] Could not build result material for "
              f"'{getattr(obj, 'name', '?')}': {e}")
        return False


def perm_result_viz_objects():
    """Every PERM result object in the file — displacement and stress alike."""
    return [o for o in bpy.data.objects
            if o.get("perm_result_viz") or o.get("perm_disp_viz")]


def _update_result_opacity(self, context):
    """Live slider: set the Mix Shader factor on every PERM result material."""
    val = float(getattr(context.scene, 'perm_result_opacity', 1.0))
    n = 0
    for o in perm_result_viz_objects():
        for slot in o.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            mix = mat.node_tree.nodes.get(PERM_MIX_NODE)
            if mix is None:
                continue
            mix.inputs[0].default_value = val
            n += 1
    # Material Preview is the only viewport mode that can show both the texture
    # and the overlay; Solid+Vertex draws the attribute alone.
    if n and val < 1.0:
        _perm_set_material_preview()


def _perm_set_material_preview():
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.shading.type == 'SOLID':
                        space.shading.type = 'MATERIAL'
    except Exception:
        pass


def _update_deformation_scale(self, context):
    """Update callback for the perm_deformation_scale slider: live-rescale every
    PERM displacement-viz object in place. No re-solve, no new mesh."""
    scale = float(getattr(context.scene, 'perm_deformation_scale', 1.0))
    for obj in bpy.data.objects:
        if obj.get("perm_disp_viz"):
            _apply_disp_scale_to_object(obj, scale)



# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE RESULT INSPECTION  (hover a node → node number + values)
# ─────────────────────────────────────────────────────────────────────────────
# Every result mesh carries, per vertex, the FE node id, the solver's global
# displacement/rotation, the support reaction (if any) and the value its colours
# show (see PERMSolver._bake_hover_data). The inspector is a modal operator that
# projects those vertices to the screen on every mouse move, picks the nearest
# one that is not hidden behind a surface, and draws a tooltip in the viewport.
# All navigation keys pass straight through, so you can orbit/zoom while it runs.
# Values come from the mesh attributes, so it also works after save + reopen;
# "Shift = all fields" additionally needs the solver from this session.

PERM_HOVER_RADIUS_PX = 14
_PERM_LABEL_HANDLE = None


def _perm_hover_objects(scene):
    """Visible result meshes carrying hover data, plus visible reaction objects."""
    meshes, reac = [], []
    for o in scene.objects:
        try:
            vis = o.visible_get()
        except Exception:
            vis = not o.hide_get()
        if not vis:
            continue
        if o.get("perm_reaction_viz"):
            reac.append(o)
        elif (o.type == 'MESH' and o.mode != 'EDIT'          # edit-mode attribute data is empty
              and "perm_node_id" in o.data.attributes
              and len(o.data.attributes["perm_node_id"].data) == len(o.data.vertices)):
            meshes.append(o)
    return meshes, reac


def _perm_world_coords(obj) -> np.ndarray:
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    M = np.array(obj.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def _perm_project(world: np.ndarray, persp: np.ndarray, width: float, height: float):
    """World points → region pixels with the view's perspective matrix.
    Returns (sx, sy, in_front). Works for perspective AND orthographic views."""
    h = np.c_[world, np.ones(len(world))]
    clip = h @ np.asarray(persp, dtype=np.float64).T
    w = clip[:, 3]
    ok = w > 1e-12
    w_safe = np.where(ok, w, 1.0)
    sx = (clip[:, 0] / w_safe + 1.0) * 0.5 * width
    sy = (clip[:, 1] / w_safe + 1.0) * 0.5 * height
    return sx, sy, ok


def _perm_attr(mesh, name, vi, vector=True):
    a = mesh.attributes.get(name)
    if a is None:
        return None
    d = a.data[vi]
    return tuple(d.vector) if vector else d.value


def _perm_pick(meshes, reac_objs, persp, width, height, mx, my,
               ray_fn=None, active=None, radius=PERM_HOVER_RADIUS_PX):
    """Nearest visible node to the mouse.

    meshes     result objects with hover data
    reac_objs  reaction-arrow objects (their support nodes are pickable too)
    ray_fn     optional (obj, world_point, sx, sy) -> True if the point is hidden
               behind a surface; None skips occlusion (used by tests)
    Returns dict(obj, vi, nid, world, sx, sy) or None.
    """
    cands = []
    r2 = radius * radius
    for obj in meshes:
        wc = _perm_world_coords(obj)
        sx, sy, ok = _perm_project(wc, persp, width, height)
        d2 = (sx - mx) ** 2 + (sy - my) ** 2
        for vi in np.nonzero(ok & (d2 <= r2))[0]:
            cands.append((float(d2[vi]), 0 if obj is active else 1, obj, int(vi), wc[vi],
                          float(sx[vi]), float(sy[vi])))
    for obj in reac_objs:
        try:
            nodes = json.loads(obj.get("perm_reaction_nodes", "[]"))
        except Exception:
            nodes = []
        if not nodes:
            continue
        M = np.array(obj.matrix_world)
        wc = np.array([n[1:4] for n in nodes], dtype=np.float64) @ M[:3, :3].T + M[:3, 3]
        sx, sy, ok = _perm_project(wc, persp, width, height)
        d2 = (sx - mx) ** 2 + (sy - my) ** 2
        for k in np.nonzero(ok & (d2 <= r2))[0]:
            cands.append((float(d2[k]), 2, obj, int(k), wc[k], float(sx[k]), float(sy[k])))
    cands.sort(key=lambda c: (c[0], c[1]))
    for d2, _, obj, vi, w, sx_, sy_ in cands[:60]:
        if ray_fn is not None and obj.type == 'MESH' and not obj.get("perm_reaction_viz") \
                and ray_fn(obj, w, sx_, sy_):
            continue
        if obj.get("perm_reaction_viz"):
            nid = int(json.loads(obj["perm_reaction_nodes"])[vi][0])
        else:
            nid = int(_perm_attr(obj.data, "perm_node_id", vi, vector=False))
        if nid < 0:
            continue
        return {'obj': obj, 'vi': vi, 'nid': nid, 'world': w, 'sx': sx_, 'sy': sy_}
    return None


def _perm_fmt(v: float, zero_below: float = 0.0) -> str:
    """4 significant digits; |v| below `zero_below` is round-off noise -> 0."""
    if abs(v) < zero_below:
        v = 0.0
    return f"{v:.4g}"


def _perm_vec(v, zero_below: float) -> str:
    return ", ".join(_perm_fmt(float(c), zero_below) for c in v)


def _perm_hover_lines(hit, meshes, reac_objs, shift=False, solver=None):
    """Tooltip text for a picked node: list of (text, rgba)."""
    WHITE, GREY, HEAD = (1, 1, 1, 1), (0.75, 0.75, 0.75, 1), (1.0, 0.85, 0.35, 1)
    CYAN, ORANGE = (0.55, 0.85, 1.0, 1), (1.0, 0.6, 0.35, 1)
    nid = hit['nid']
    lines = [(f"Node {nid}", HEAD)]

    # A mesh vertex for this node (any visible result mesh) → coordinates + data
    src = None
    for o in ([hit['obj']] if hit['obj'] in meshes else []) + [m for m in meshes if m is not hit['obj']]:
        ids = o.data.attributes.get("perm_node_id")
        if ids is None:
            continue
        arr = np.empty(len(o.data.vertices), dtype=np.int32)
        ids.data.foreach_get("value", arr)
        where = np.nonzero(arr == nid)[0]
        if len(where):
            src = (o, int(where[0]))
            break
    if src is not None:
        o, vi = src
        me = o.data
        rest = _perm_attr(me, "perm_rest_co", vi)
        co = np.array(rest) if rest is not None else np.array(me.vertices[vi].co)
        M = np.array(o.matrix_world)
        xyz = co @ M[:3, :3].T + M[:3, 3]
        lines.append((f"x, y, z = {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} m", GREY))
        u = np.array(_perm_attr(me, "perm_u", vi) or (0, 0, 0)) * 1e3
        r = np.array(_perm_attr(me, "perm_rot", vi) or (0, 0, 0)) * 1e3
        # Noise floors: 1 nm, 1 nrad, 1 mN, 1 mN·m
        lines.append((f"u = ({_perm_vec(u, 1e-6)}) mm   "
                      f"|u| = {_perm_fmt(float(np.linalg.norm(u)), 1e-6)} mm", WHITE))
        lines.append((f"θ = ({_perm_vec(r, 1e-6)}) mrad", GREY))
        if _perm_attr(me, "perm_is_support", vi, vector=False):
            f = np.array(_perm_attr(me, "perm_reac_f", vi)) / 1e3
            m = np.array(_perm_attr(me, "perm_reac_m", vi)) / 1e3
            lines.append((f"Reaction F = ({_perm_vec(f, 1e-6)}) kN", ORANGE))
            if np.any(np.abs(m) > 1e-6):
                lines.append((f"Reaction M = ({_perm_vec(m, 1e-6)}) kN·m", ORANGE))
    else:
        # Only a reaction object is visible: read its stored node data
        for ro in reac_objs:
            for n in json.loads(ro.get("perm_reaction_nodes", "[]")):
                if int(n[0]) == nid:
                    lines.append((f"x, y, z = {n[1]:.3f}, {n[2]:.3f}, {n[3]:.3f} m", GREY))
                    f = np.array(n[4:7]) / 1e3
                    lines.append((f"Reaction F = ({_perm_vec(f, 1e-6)}) kN", ORANGE))
                    m = np.array(n[7:10]) / 1e3
                    if any(d >= 3 for d in n[10]):
                        lines.append((f"Reaction M = ({_perm_vec(m, 1e-6)}) kN·m", ORANGE))
                    break

    # The field shown by each visible result mesh at this node (picked one first).
    # With Shift the full list follows anyway, so only the picked one is shown;
    # without it, cap the list so the tooltip stays on screen.
    MAX_OTHER = 6
    ordered = ([hit['obj']] if hit['obj'] in meshes else []) + [m for m in meshes if m is not hit['obj']]
    if shift:
        ordered = ordered[:1]
    n_other_shown = n_other_hidden = 0
    for o in ordered:
        if o is not hit['obj']:
            if n_other_shown >= MAX_OTHER:
                if o.get("perm_field_component") is not None:
                    n_other_hidden += 1
                continue
        comp = o.get("perm_field_component")
        if comp is None:
            continue
        me = o.data
        ids = me.attributes.get("perm_node_id")
        arr = np.empty(len(me.vertices), dtype=np.int32)
        ids.data.foreach_get("value", arr)
        where = np.nonzero(arr == nid)[0]
        if not len(where):
            continue
        vi = int(where[0])
        label = o.get("perm_field_label", comp)
        unit = o.get("perm_field_unit", "")
        mult = float(o.get("perm_field_mult", 1.0))
        if not _perm_attr(me, "perm_field_valid", vi, vector=False):
            lines.append((f"{label}: no value (non-QUAD4 face)", GREY))
            continue
        val = float(_perm_attr(me, "perm_field", vi, vector=False)) * mult
        mark = "▸ " if o is hit['obj'] else "   "
        lines.append((f"{mark}{label}: {_perm_fmt(val, 1e-6 if unit == 'mm' else 0.0)} {unit}",
                      CYAN if o is hit['obj'] else WHITE))
        if o is not hit['obj']:
            n_other_shown += 1
    if n_other_hidden:
        lines.append((f"   … {n_other_hidden} more visible field(s) - hold Shift", GREY))

    if shift:
        if solver is not None:
            lines.append(("All fields (average of adjacent elements):", HEAD))
            for label, val, unit in solver.nodal_all_fields(nid):
                lines.append((f"   {label}: {_perm_fmt(val)} {unit}", WHITE))
        else:
            lines.append(("All fields: re-run the analysis in this session", GREY))
    else:
        lines.append(("Shift: all stress fields · Esc: exit", GREY))
    return lines


def _perm_session_solver(solve_id):
    """The solver from this session, if it produced the result being inspected."""
    try:
        import __main__
        sv = getattr(__main__, "_PERM_LAST_SOLVER", None)
    except Exception:
        return None
    if sv is not None and solve_id and getattr(sv, "_solve_id", None) == solve_id:
        return sv
    return None


def _perm_uniform_shader():
    import gpu
    for name in ('UNIFORM_COLOR', '2D_UNIFORM_COLOR'):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:
            continue
    return None


def _perm_draw_rect(shader, x0, y0, x1, y1, color):
    from gpu_extras.batch import batch_for_shader
    batch = batch_for_shader(shader, 'TRIS', {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]},
                             indices=[(0, 1, 2), (0, 2, 3)])
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _perm_blf_size(fid, px):
    import blf
    try:
        blf.size(fid, px)
    except TypeError:
        blf.size(fid, int(px), 72)


def _perm_hover_draw(op, context):
    """POST_PIXEL callback of the inspector: node marker + tooltip."""
    try:
        if context.area is None or context.area.as_pointer() != op._area_ptr:
            return
        hit, lines = op._hit, op._lines
        if not hit or not lines:
            return
        import gpu, blf
        ui = (context.preferences.system.ui_scale
              or context.preferences.view.ui_scale or 1.0)
        shader = _perm_uniform_shader()
        gpu.state.blend_set('ALPHA')
        sx, sy = hit['sx'], hit['sy']
        m = 4 * ui
        if shader is not None:
            _perm_draw_rect(shader, sx - m - 1, sy - m - 1, sx + m + 1, sy + m + 1, (0, 0, 0, 1))
            _perm_draw_rect(shader, sx - m, sy - m, sx + m, sy + m, (1.0, 0.85, 0.2, 1))
        fid = 0
        _perm_blf_size(fid, 12 * ui)
        line_h = 16 * ui
        pad = 7 * ui
        w = max(blf.dimensions(fid, t)[0] for t, _ in lines) + 2 * pad
        h = line_h * len(lines) + 2 * pad
        x0 = sx + 16 * ui
        y1 = sy - 10 * ui
        region = context.region
        if x0 + w > region.width:
            x0 = sx - 16 * ui - w
        y0 = y1 - h
        if y0 < 0:
            y0 = sy + 10 * ui
            y1 = y0 + h
        if shader is not None:
            _perm_draw_rect(shader, x0, y0, x0 + w, y1, (0.08, 0.08, 0.10, 0.88))
        for k, (text, col) in enumerate(lines):
            blf.color(fid, *col)
            blf.position(fid, x0 + pad, y1 - pad - (k + 1) * line_h + 4 * ui, 0)
            blf.draw(fid, text)
        gpu.state.blend_set('NONE')
    except Exception as e:  # never let a draw error spam every redraw
        if not getattr(op, "_draw_err", False):
            print(f"[PERM] Inspector draw error: {e}")
            op._draw_err = True


_PERM_ARROW_HANDLE = None
_PERM_ARROW_CACHE = {}


def _perm_draw_reaction_lines(objs, viewport_size, line_width=1.5):
    """Draw the reaction arrows of `objs` as thin lines (current view matrices)."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    try:
        sh = gpu.shader.from_builtin('POLYLINE_FLAT_COLOR')
        poly = True
    except Exception:
        sh = gpu.shader.from_builtin('FLAT_COLOR')
        poly = False
    for o in objs:
        key = (o.name, o.get("perm_build_id", ""),
               tuple(round(v, 6) for row in o.matrix_world for v in row))
        batch = _PERM_ARROW_CACHE.get(key)
        if batch is None:
            arr = np.asarray(json.loads(o.get("perm_reaction_lines", "[]")), dtype=np.float64)
            if not len(arr):
                continue
            M = np.array(o.matrix_world)
            p0 = arr[:, 0:3] @ M[:3, :3].T + M[:3, 3]
            p1 = arr[:, 3:6] @ M[:3, :3].T + M[:3, 3]
            pos = np.empty((2 * len(arr), 3), dtype=np.float32)
            pos[0::2], pos[1::2] = p0, p1
            col = np.ones((2 * len(arr), 4), dtype=np.float32)
            col[:, :3] = np.repeat(arr[:, 6:9], 2, axis=0)
            batch = batch_for_shader(sh, 'LINES', {"pos": pos.tolist(), "color": col.tolist()})
            if len(_PERM_ARROW_CACHE) > 16:
                _PERM_ARROW_CACHE.clear()
            _PERM_ARROW_CACHE[key] = batch
        sh.bind()
        if poly:
            sh.uniform_float("viewportSize", viewport_size)
            sh.uniform_float("lineWidth", line_width)
        batch.draw(sh)


def _perm_reaction_arrows_draw():
    """Persistent POST_VIEW handler: reaction arrows of visible reaction objects."""
    try:
        context = bpy.context
        region = context.region
        if region is None or context.region_data is None:
            return
        objs = [o for o in context.scene.objects
                if o.get("perm_reaction_viz") and o.get("perm_reaction_lines") and o.visible_get()]
        if not objs:
            return
        import gpu
        ui = (context.preferences.system.ui_scale or context.preferences.view.ui_scale or 1.0)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        _perm_draw_reaction_lines(objs, (region.width, region.height), 1.5 * ui)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


def _perm_draw_reaction_labels(objs, project, width, height, ui=1.0):
    """Draw node labels; `project(Vector) -> (x, y) | None`. Labels that would
    overlap an already drawn one are skipped (largest reactions first)."""
    import blf
    fid = 0
    _perm_blf_size(fid, 11 * ui)
    try:
        blf.enable(fid, blf.SHADOW)
        blf.shadow(fid, 3, 0.0, 0.0, 0.0, 0.9)
        blf.shadow_offset(fid, 1, -1)
    except Exception:
        pass
    cell = 13.0 * ui
    taken = set()
    for o in objs:
        M = o.matrix_world
        labs = json.loads(o.get("perm_reaction_labels", "[]"))
        labs.sort(key=lambda l: -(l[7] if len(l) > 7 else 0.0))
        for lab in labs:
            p2 = project(M @ Vector(lab[0:3]))
            if p2 is None:
                continue
            x, y = p2[0] + 5 * ui, p2[1] - 5 * ui
            w = blf.dimensions(fid, lab[3])[0]
            if x < -w or y < -cell or x > width or y > height:
                continue
            h = 11.0 * ui                      # label rectangle [x, x+w] x [y, y+h]
            cells = {(cx, cy)
                     for cx in range(int(x // cell), int((x + w) // cell) + 1)
                     for cy in range(int(y // cell), int((y + h) // cell) + 1)}
            if cells & taken:
                continue
            taken |= cells
            blf.color(fid, lab[4], lab[5], lab[6], 1.0)
            blf.position(fid, x, y, 0)
            blf.draw(fid, lab[3])
    try:
        blf.disable(fid, blf.SHADOW)
    except Exception:
        pass


def _perm_reaction_labels_draw():
    """Persistent POST_PIXEL handler: value labels of visible reaction objects."""
    try:
        context = bpy.context
        scn = context.scene
        if not getattr(scn, "perm_show_reaction_labels", True):
            return
        region, rv3d = context.region, context.region_data
        if region is None or rv3d is None:
            return
        objs = [o for o in scn.objects if o.get("perm_reaction_viz")]
        if not objs:
            return
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        ui = (context.preferences.system.ui_scale
              or context.preferences.view.ui_scale or 1.0)
        objs = [o for o in objs if o.visible_get()]
        _perm_draw_reaction_labels(
            objs, lambda v: location_3d_to_region_2d(region, rv3d, v),
            region.width, region.height, ui)
    except Exception:
        pass


class PERM_OT_InspectResults(bpy.types.Operator):
    """Hover a node of any visible result mesh to see its number, displacement,
    reaction and field values. Shift shows every stress field. Esc exits"""
    bl_idname = "perm.inspect_results"
    bl_label = "PERM: Inspect Results (hover)"

    _running = False
    _stop = False

    def invoke(self, context, event):
        cls = PERM_OT_InspectResults
        if cls._running:                     # second click = stop
            cls._stop = True
            return {'FINISHED'}
        area = context.area if (context.area and context.area.type == 'VIEW_3D') else None
        if area is None:
            for a in context.screen.areas:
                if a.type == 'VIEW_3D':
                    area = a
                    break
        if area is None:
            self.report({'ERROR'}, "No 3D Viewport to inspect")
            return {'CANCELLED'}
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            return {'CANCELLED'}
        meshes, reac = _perm_hover_objects(context.scene)
        if not meshes and not reac:
            self.report({'WARNING'}, "No result meshes with node data are visible - "
                                     "run the analysis first")
            return {'CANCELLED'}
        self._area, self._region = area, region
        self._area_ptr = area.as_pointer()
        self._hit, self._lines = None, []
        self._bvh = {}
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _perm_hover_draw, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        cls._running, cls._stop = True, False
        context.scene.perm_inspect_running = True
        try:
            area.header_text_set("PERM Inspect: hover a node · Shift = all fields · Esc = exit")
        except Exception:
            pass
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        cls = PERM_OT_InspectResults
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        except Exception:
            pass
        cls._running = cls._stop = False
        try:
            context.scene.perm_inspect_running = False
            self._area.header_text_set(None)
            self._area.tag_redraw()
        except Exception:
            pass

    def _occluded(self, obj, world_pt, sx, sy):
        """True if a surface of `obj` lies between the eye and the point."""
        from bpy_extras import view3d_utils
        from mathutils.bvhtree import BVHTree
        key = (obj.name, round(float(getattr(bpy.context.scene, 'perm_deformation_scale', 1.0)), 6),
               len(obj.data.vertices))
        bvh = self._bvh.get(key)
        if bvh is None:
            wc = _perm_world_coords(obj)
            bvh = BVHTree.FromPolygons([tuple(v) for v in wc],
                                       [tuple(p.vertices) for p in obj.data.polygons])
            self._bvh = {k: v for k, v in self._bvh.items() if k[0] != obj.name}
            self._bvh[key] = bvh
        rv3d = self._region.data
        origin = view3d_utils.region_2d_to_origin_3d(self._region, rv3d, (sx, sy))
        target = Vector(tuple(world_pt))
        direction = target - origin
        dist = direction.length
        if dist <= 0:
            return False
        hit = bvh.ray_cast(origin, direction.normalized(), dist)
        return hit[0] is not None and hit[3] < dist * (1.0 - 1e-3) - 1e-6

    def modal(self, context, event):
        cls = PERM_OT_InspectResults
        if cls._stop or (event.type == 'ESC' and event.value == 'PRESS'):
            self._finish(context)
            return {'FINISHED'}
        if event.type in {'MOUSEMOVE', 'LEFT_SHIFT', 'RIGHT_SHIFT', 'INBETWEEN_MOUSEMOVE'}:
            reg = self._region
            mx, my = event.mouse_x - reg.x, event.mouse_y - reg.y
            hit = None
            if 0 <= mx <= reg.width and 0 <= my <= reg.height and reg.data is not None:
                meshes, reac = _perm_hover_objects(context.scene)
                persp = np.array(reg.data.perspective_matrix)
                try:
                    hit = _perm_pick(meshes, reac, persp, reg.width, reg.height, mx, my,
                                     ray_fn=self._occluded, active=context.active_object)
                except Exception as e:
                    print(f"[PERM] Inspector pick error: {e}")
                    hit = None
                if hit is not None:
                    solver = None
                    if event.shift:
                        sid = hit['obj'].get("perm_solve_id", "")
                        solver = _perm_session_solver(sid)
                    try:
                        self._lines = _perm_hover_lines(hit, meshes, reac, shift=event.shift,
                                                        solver=solver)
                    except Exception as e:
                        print(f"[PERM] Inspector: {e}")
                        hit = None
            self._hit = hit
            self._area.tag_redraw()
        return {'PASS_THROUGH'}


def _update_reaction_scale(self, context):
    """Rebuild the reaction arrows at the new size (needs this session's solve)."""
    try:
        import __main__
        sv = getattr(__main__, "_PERM_LAST_SOLVER", None)
        if sv is not None and any(o.get("perm_reaction_viz") for o in bpy.data.objects):
            sv.visualize_reactions(size_scale=float(context.scene.perm_reaction_scale))
    except Exception as e:
        print(f"[PERM] Reaction rescale failed: {e}")



def _perm_check_supports(solver, groups) -> None:
    """Explain what the supports do to a girder, before solving.

    groups: [(node_set_name, bc_type_str)]. The bridge axis is the model's
    longest extent. Flags FIXED supports, supports spread along the span (a
    patch of pinned nodes resists rotation like a clamp) and the absence of
    any support free to slide along the bridge (horizontal thrust)."""
    conv = solver.converter
    P = solver.node_coords
    ext = P.max(axis=0) - P.min(axis=0)
    ax = int(np.argmax(ext)); axn = "XYZ"[ax]; span = float(ext[ax]) or 1.0
    frees_axis = {'ROLLER_X': 0, 'ROLLER_Y': 1, 'ROLLER_Z': 2}
    if not PERM_GIRDER_CHECKS:
        # any structure: list the supports, flag shared nodes; no girder advice
        print("[PERM Solver] Supports:")
        seen = {}
        for name, bc in groups:
            ids = [n for n in conv.node_sets.get(name, set()) if n in solver.node_id_to_index]
            if not ids:
                print(f"   ⚠ {name.replace('support_', '')}: no nodes of this group are in the model (vertices "
                      f"removed by the mesh clean-up?) - it restrains nothing.")
                continue
            Q = P[[solver.node_id_to_index[n] for n in ids]]
            label = name.replace("support_", "")
            lo, hi = Q.min(axis=0), Q.max(axis=0)
            print(f"   {label:<10} {bc:<9} {len(ids):5d} nodes | x {lo[0]:+.3f} … {hi[0]:+.3f}, "
                  f"y {lo[1]:+.3f} … {hi[1]:+.3f}, z {lo[2]:+.3f} … {hi[2]:+.3f} m")
            for n in ids:
                if n in seen and seen[n] != label:
                    print(f"   ⚠ {label} shares nodes with {seen[n]} - the stricter one wins there.")
                    break
            for n in ids:
                seen.setdefault(n, label)
        return
    print(f"[PERM Solver] Support check (bridge axis = {axn}, length {span:.2f} m):")
    any_slides = False
    seen = {}
    for name, bc in groups:
        ids = [n for n in conv.node_sets.get(name, set()) if n in solver.node_id_to_index]
        if not ids:
            continue
        Q = P[[solver.node_id_to_index[n] for n in ids]]
        lo, hi = float(Q[:, ax].min()), float(Q[:, ax].max())
        label = name.replace("support_", "")
        print(f"   {label:<10} {bc:<9} {len(ids):5d} nodes | along {axn}: {lo:+.3f} … {hi:+.3f} m "
              f"(length {hi - lo:.3f} m)")
        if frees_axis.get(bc) == ax:
            any_slides = True
        if bc == 'FIXED':
            print(f"   ⚠ {label}: FIXED clamps all rotations. Bridge bearings do not - "
                  f"use PINNED at one end and ROLLER_{axn} at the other.")
        zlo, zhi = float(Q[:, 2].min()), float(Q[:, 2].max())
        restrains_axis = bc in ('FIXED', 'PINNED') or (bc.startswith('ROLLER_') and frees_axis.get(bc) != ax)
        print(f"   {'':<10} {'':<9}       | height: {zlo:+.3f} … {zhi:+.3f} m (spread {zhi - zlo:.3f} m)")
        if restrains_axis and zhi - zlo > 0.05:
            print(f"   ⚠ {label}: {axn} is restrained at nodes spread {zhi - zlo:.2f} m in height. Their "
                  f"{axn}-reactions form a couple, so this end cannot rotate - it acts like a clamp. "
                  f"A bearing restrains only the bottom (soffit) nodes.")
        if hi - lo > 0.02 * span:
            print(f"   ⚠ {label}: restrained nodes spread {hi - lo:.2f} m along the span. "
                  f"Even PINNED nodes spread along the bridge resist rotation like a clamp. "
                  f"Restrain one transverse line of nodes at the bearing centreline.")
        for n in ids:
            if n in seen and seen[n] != label:
                print(f"   ⚠ {label} shares nodes with {seen[n]} - the stricter one wins there.")
                break
        for n in ids:
            seen.setdefault(n, label)
    if groups and not any_slides:
        print(f"   ⚠ No support is free to move along {axn}: the girder cannot lengthen at its "
              f"bottom fibre, so the supports push back (horizontal thrust, Rx arrows) and the "
              f"bridge acts stiffer than on bearings. Make one end ROLLER_{axn}.")


def _perm_model_inventory(solver, scene) -> None:
    """Before solving: list the materials as the solver will use them, the
    area and mass per (material, thickness), and flag implausible inputs."""
    conv = solver.converter
    print("[PERM Solver] Materials used:")
    for mid, m in sorted(conv.materials.items()):
        print(f"   {m.name:<14} E = {m.youngs_modulus:.3e} Pa | ν = {m.poissons_ratio:.2f} | "
              f"ρ = {m.density:.0f} kg/m³")
        if m.density > 7000 and m.youngs_modulus < 1.5e11:
            print(f"   ⚠ '{m.name}' has steel density but E = {m.youngs_modulus:.2e} Pa "
                  f"(steel ≈ 2.0e11). Type it as 2e11 or 200e9 - no space before 'e'.")
        if m.density < 3000 and m.youngs_modulus > 6e10:
            print(f"   ⚠ '{m.name}' has concrete density but E = {m.youngs_modulus:.2e} Pa.")
    groups = defaultdict(lambda: [0.0, 0.0, 0])          # (mat, section, t) -> area, mass, n
    sec_of = _element_face_tag_names(conv, 'section')
    for eid, e in conv.elements.items():
        if _perm_is_line(e) or getattr(e, 'viz_only', False):
            continue
        m = conv.materials.get(e.material_id, conv.materials[0])
        a = solver._compute_element_area(solver._get_element_coords(e), e.element_type)
        g = groups[(m.name, sec_of.get(eid, "(none)"), round(e.thickness, 6))]
        g[0] += a; g[1] += a * e.thickness * m.density; g[2] += 1
    total = sum(g[1] for g in groups.values())
    print(f"[PERM Solver] Area and mass by material + section (total {total:,.0f} kg):")
    mat_names = {mm.name.lower(): mm.name for mm in conv.materials.values()}
    for (mname, sname, t), (a, mass, n) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        note = "  ← dialog default (no Cross-Section)" if sname == "(none)" else ""
        print(f"   material {mname:<12} section {sname:<14} t = {t*1e3:6.1f} mm : "
              f"{n:7d} faces, {a:9.2f} m², {mass:10,.0f} kg{note}")
        # a section named like ANOTHER material almost always means the wrong
        # material was active when those faces were assigned
        other = mat_names.get(sname.lower())
        if other and other != mname and n > 0:
            print(f"   ⚠ {n} face(s) with section '{sname}' use material '{mname}', not "
                  f"'{other}'. Select them (section '{sname}' → Select), pick material "
                  f"'{other}' and Assign.")
        m = next((mm for mm in conv.materials.values() if mm.name == mname), None)
        if PERM_GIRDER_CHECKS and m is not None and m.density > 7000 and t > 0.02 and n > 0:
            print(f"   ⚠ {mname}: a {t*1e3:.0f} mm steel shell. Railing tube walls are 3-6 mm; "
                  f"a thick wall makes the railing ~{t/0.005:.0f}x too heavy and stiff.")
    for pg in scene.perm_sections:
        if pg.rebar_ratio > 0 and pg.rebar_cover > 0.5 * pg.thickness + 1e-9:
            print(f"   ⚠ Section '{pg.name}': cover {pg.rebar_cover:.3f} m > t/2 = "
                  f"{0.5*pg.thickness:.3f} m - cover is face→bar distance; it was clamped to t/2.")


def _perm_support_resultants(solver, groups) -> None:
    """Resultant force and moment of every support group about its own centroid.
    A real pin / roller line carries (almost) no bending moment about the
    transverse axis; a large one means the support is clamping the girder."""
    R = solver.results.reaction_forces
    if R is None:
        return
    P = solver.node_coords
    ext = P.max(axis=0) - P.min(axis=0)
    ax = int(np.argmax(ext))
    tr = 1 if ax == 0 else 0                          # transverse horizontal axis
    span = float(ext[ax]) or 1.0
    W = float(abs(solver.results.reaction_total[2])) if solver.results.reaction_total is not None else 0.0
    ref = W * span / 8.0                              # simple-span midspan moment, order of magnitude
    print("[PERM Solver] Support resultants (about each group's centroid):")
    for name, bc in groups:
        ids = [n for n in solver.converter.node_sets.get(name, set()) if n in solver.node_id_to_index]
        if not ids:
            continue
        ii = [solver.node_id_to_index[n] for n in ids]
        c = P[ii].mean(axis=0)
        F = R[ii, :3].sum(axis=0)
        M = R[ii, 3:6].sum(axis=0) + np.cross(P[ii] - c, R[ii, :3]).sum(axis=0)
        mb = M[tr]                                    # bending about the transverse axis
        label = name.replace("support_", "")
        print(f"   {label:<10} {bc:<9} F = ({F[0]/1e3:+.2f}, {F[1]/1e3:+.2f}, {F[2]/1e3:+.2f}) kN | "
              f"bending moment M{'XYZ'[tr]} = {mb/1e3:+.2f} kN·m")
        if PERM_GIRDER_CHECKS and ref > 0 and abs(mb) > 0.05 * ref and str(bc) != 'FIXED':   # a FIXED support carries moment by design
            print(f"   ⚠ {label}: carries {abs(mb)/ref*100:.0f}% of a simple-span moment (wL/8 ≈ "
                  f"{ref/1e3:.1f} kN·m). A pin/roller bearing should carry ~0 - this support clamps "
                  f"the girder.")


def _perm_report_peaks(solver, scene, src_obj, top: int = 10) -> None:
    """Print the elements with the highest von Mises stress, with what they are
    (thickness, material), where they are, and their shape quality -- and keep
    their centroids so 'Select Peak Faces' can find them in the model."""
    vm = solver.results.von_mises or {}
    if not vm:
        return
    conv = solver.converter
    ranked = sorted(vm.items(), key=lambda kv: -kv[1])[:top]
    adj = defaultdict(int)
    for e in conv.elements.values():
        for n in e.node_ids:
            adj[n] += 1
    print(f"[PERM Solver] Top {len(ranked)} von Mises elements "
          f"(element: VM | t | material | centroid x,y,z | scaled Jacobian | warpage | min/max edge):")
    cents = []
    for eid, val in ranked:
        e = conv.elements[eid]
        pts = [conv.nodes[n].position for n in e.node_ids[:4]]
        c = sum((p for p in pts), Vector((0, 0, 0))) / len(pts)
        jac = PERMMeshConverter._scaled_jacobian(pts)
        edges = [(pts[(k + 1) % len(pts)] - pts[k]).length for k in range(len(pts))]
        warp = 0.0
        if len(pts) == 4:
            nv = (pts[1] - pts[0]).cross(pts[2] - pts[0])
            if nv.length > 1e-15:
                warp = abs((pts[3] - pts[0]).dot(nv.normalized())) / max(sum(edges) / 4, 1e-15)
        mat = conv.materials.get(e.material_id)
        lone = sum(1 for n in e.node_ids if adj[n] == 1)
        flags = []
        if jac < 0: flags.append("FOLDED")
        if warp > 0.3: flags.append("WARPED")
        if abs(e.thickness - 0.30) < 1e-6: flags.append("NO-SECTION(0.30m)")
        if lone >= 2: flags.append("FREE-CORNERS")
        print(f"   #{eid}: {val/1e6:9.1f} MPa | t={e.thickness:.4g} m | {getattr(mat, 'name', '?')} | "
              f"({c.x:.3f}, {c.y:.3f}, {c.z:.3f}) | J={jac:+.2f} | warp={warp:.2f} | "
              f"edges {min(edges)*1e3:.0f}/{max(edges)*1e3:.0f} mm {' '.join(flags)}")
        cents.append([c.x, c.y, c.z])
    scene["perm_peak_centroids"] = json.dumps(cents)
    scene["perm_peak_source"] = src_obj.name


def _perm_face_problems(obj, warp_limit: float = 0.30):
    """Classify the model's faces. Returns dict category -> list of polygon indices.

    folded    scaled Jacobian < 0 (concave / bow-tie / folded quad): integrated
              with |det J|, so its stiffness is not meaningful
    ngon      more than 4 vertices: produces NO element (a hole)
    warped    out-of-plane distance / mean edge > warp_limit: the flat-shell
              formulation is inaccurate there
    no_section / no_material   no face tag: falls back to the 0.30 m default
              thickness and the first material
    """
    me = obj.data
    M = obj.matrix_world
    co = [M @ v.co for v in me.vertices]
    out = {k: [] for k in ("folded", "ngon", "warped", "no_section", "no_material")}
    for p in me.polygons:
        n = len(p.vertices)
        if n > 4:
            out["ngon"].append(p.index)
            continue
        pts = [co[i] for i in p.vertices]
        if PERMMeshConverter._scaled_jacobian(pts) < 0.0:
            out["folded"].append(p.index)
        if n == 4:
            nv = (pts[1] - pts[0]).cross(pts[2] - pts[0])
            if nv.length > 1e-15:
                nv.normalize()
                avg = sum((pts[(k + 1) % 4] - pts[k]).length for k in range(4)) / 4.0
                if abs((pts[3] - pts[0]).dot(nv)) / max(avg, 1e-15) > warp_limit:
                    out["warped"].append(p.index)
    scn = bpy.context.scene
    for kind, key, names in (("section", "no_section", {pg.name for pg in scn.perm_sections}),
                             ("material", "no_material", {pg.name for pg in scn.perm_materials})):
        tags = _read_face_tags(me, kind)
        lookup = _face_tag_names(obj, kind)
        for p in me.polygons:
            if len(p.vertices) > 4:
                continue
            t = int(tags[p.index]) if tags is not None else 0
            if lookup.get(t) not in names:
                out[key].append(p.index)
    return out


class PERM_OT_SelectProblemFaces(bpy.types.Operator):
    """Select the model's problem faces in Edit Mode: folded/inverted quads,
    n-gons, strongly warped quads and faces with no section or material"""
    bl_idname = "perm.select_problem_faces"
    bl_label = "PERM: Select Problem Faces"
    bl_options = {'REGISTER', 'UNDO'}

    folded: bpy.props.BoolProperty(name="Folded / inverted (Jacobian < 0)", default=True)
    ngon: bpy.props.BoolProperty(name="N-gons (> 4 vertices)", default=True)
    warped: bpy.props.BoolProperty(name="Strongly warped quads", default=False)
    warp_limit: bpy.props.FloatProperty(name="Warpage limit", default=0.30, min=0.01, max=1.0)
    unassigned: bpy.props.BoolProperty(name="No section or material", default=True)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def execute(self, context):
        src = _perm_resolve_source(context)
        if src is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model)"); return {'CANCELLED'}
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # make the model visible + active (results hidden), like Edit Model
        for o in context.scene.objects:
            if o is not src and o.get("perm_source_name") == src.name:
                o.hide_set(True)
        src.hide_set(False)
        for o in context.selected_objects:
            o.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src

        probs = _perm_face_problems(src, self.warp_limit)
        chosen = set()
        if self.folded: chosen |= set(probs["folded"])
        if self.ngon: chosen |= set(probs["ngon"])
        if self.warped: chosen |= set(probs["warped"])
        if self.unassigned: chosen |= set(probs["no_section"]) | set(probs["no_material"])

        me = src.data
        for v in me.vertices: v.select = False
        for e in me.edges: e.select = False
        for p in me.polygons: p.select = p.index in chosen
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')

        summary = (f"folded {len(probs['folded'])}, n-gons {len(probs['ngon'])}, "
                   f"warped>{self.warp_limit:g} {len(probs['warped'])}, "
                   f"no section {len(probs['no_section'])}, no material {len(probs['no_material'])}")
        print(f"[PERM] Problem faces in '{src.name}': {summary}")
        for k, idx in probs.items():
            if idx:
                print(f"[PERM]   {k}: {idx[:20]}{' ...' if len(idx) > 20 else ''}")
        self.report({'INFO'} if not chosen else {'WARNING'},
                    f"PERM: {len(chosen)} face(s) selected - {summary}. "
                    f"Press Numpad . to frame them.")
        return {'FINISHED'}


def _perm_unassigned_faces(obj):
    """(no_section, no_material) boolean arrays over the model's polygons.
    A face counts as assigned only if its tag names a section / material that
    is still in the scene lists. N-gons (no element at all) are left out here."""
    me = obj.data
    n = len(me.polygons)
    sizes = np.empty(n, dtype=np.int32)
    if n:
        me.polygons.foreach_get("loop_total", sizes)
    scn = bpy.context.scene
    out = []
    for kind, names in (("section", {pg.name for pg in scn.perm_sections}),
                        ("material", {pg.name for pg in scn.perm_materials})):
        tags = _read_face_tags(me, kind)
        if tags is None:
            missing = np.ones(n, dtype=bool)
        else:
            ok = [tid for tid, nm in _face_tag_names(obj, kind).items() if nm in names]
            missing = ~np.isin(tags, np.asarray(ok, dtype=np.int32))
        out.append(missing & (sizes <= 4))
    return out[0], out[1]


class PERM_OT_SelectUnassignedFaces(bpy.types.Operator):
    """Select the model's faces that have no Cross-Section and/or no Material
    (they would fall back to the 0.30 m default thickness / the first material)"""
    bl_idname = "perm.select_unassigned_faces"
    bl_label = "PERM: Select Unassigned Faces"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(
        name="Missing", default='ANY',
        items=[('ANY', "Section or material", "Faces missing either assignment"),
               ('SECTION', "Cross-Section", "Faces without a Cross-Section"),
               ('MATERIAL', "Material", "Faces without a Material")])

    def execute(self, context):
        src = _perm_resolve_source(context)
        if src is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model)"); return {'CANCELLED'}
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        no_sec, no_mat = _perm_unassigned_faces(src)
        mask = {'ANY': no_sec | no_mat, 'SECTION': no_sec, 'MATERIAL': no_mat}[self.kind]
        idx = np.nonzero(mask)[0]
        context.scene["perm_unassigned_counts"] = [int(no_sec.sum()), int(no_mat.sum())]
        # show + select the model, results hidden
        for o in context.scene.objects:
            if o is not src and o.get("perm_source_name") == src.name:
                o.hide_set(True)
        src.hide_set(False)
        for o in context.selected_objects:
            o.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(src.data)
        for v in bm.verts: v.select = False
        for e in bm.edges: e.select = False
        for f_ in bm.faces: f_.select = False
        bm.faces.ensure_lookup_table()
        for i in idx:
            bm.faces[int(i)].select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(src.data)
        what = {'ANY': "without a section or material", 'SECTION': "without a Cross-Section",
                'MATERIAL': "without a Material"}[self.kind]
        if len(idx):
            self.report({'WARNING'}, f"PERM: {len(idx)} face(s) {what} selected "
                                     f"(no section: {int(no_sec.sum())}, no material: "
                                     f"{int(no_mat.sum())}). Numpad . to frame them, then assign.")
        else:
            self.report({'INFO'}, "PERM: every face has a section and a material")
        print(f"[PERM] Unassigned faces in '{src.name}': no section {int(no_sec.sum())}, "
              f"no material {int(no_mat.sum())}")
        return {'FINISHED'}


class PERM_OT_SelectPeakFaces(bpy.types.Operator):
    """Select, in the model, the faces with the highest von Mises stress from
    the last analysis (listed in the console as 'Top 10 von Mises elements')"""
    bl_idname = "perm.select_peak_faces"
    bl_label = "PERM: Select Peak-Stress Faces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cents = json.loads(context.scene.get("perm_peak_centroids", "[]"))
        src_name = context.scene.get("perm_peak_source")
        if not cents or src_name not in bpy.data.objects:
            self.report({'WARNING'}, "Run the analysis first"); return {'CANCELLED'}
        src = bpy.data.objects[src_name]
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.scene.objects:
            if o is not src and o.get("perm_source_name") == src.name:
                o.hide_set(True)
        src.hide_set(False)
        for o in context.selected_objects:
            o.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        from mathutils import kdtree
        me = src.data
        M = src.matrix_world
        kd = kdtree.KDTree(len(me.polygons))
        for p in me.polygons:
            kd.insert(M @ p.center, p.index)
        kd.balance()
        hits = {kd.find(Vector(c))[1] for c in cents}
        for v in me.vertices: v.select = False
        for e in me.edges: e.select = False
        for p in me.polygons: p.select = p.index in hits
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        self.report({'INFO'}, f"PERM: {len(hits)} peak-stress face(s) selected - Numpad . to frame them")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# FIND ELEMENT BY NUMBER
# ─────────────────────────────────────────────────────────────────────────────
# After each solve every face of the working copy gets its element number (the
# '#' in the console and the report) and its stresses as face attributes. The
# result meshes are copies of the working copy, so they carry them too, and the
# numbers survive saving and reopening the .blend.
_PERM_EID_ATTR = "perm_element_id"                      # 0 = face produced no element
_PERM_EL_VALUES = ("perm_element_vm", "perm_element_s1", "perm_element_s2")   # Pa
_PERM_FIND_NS_KEY = "PERM_FIND_ELEMENT_HANDLES"         # draw handlers, survive a re-run of the script


def _perm_write_face_attr(mesh, name, kind, values) -> None:
    a = mesh.attributes.get(name)
    if a is not None and (a.domain != 'FACE' or a.data_type != kind):
        mesh.attributes.remove(a)
        a = None
    if a is None:
        a = mesh.attributes.new(name, kind, 'FACE')
    a.data.foreach_set("value", values)


def _perm_tag_element_ids(solver) -> int:
    """Write each element's number and its von Mises / σ1 / σ2 (worse of top and
    bottom) onto its face in the working copy and in result meshes copied from it."""
    conv = solver.converter
    wobj = getattr(conv, 'working_object', None)
    if wobj is None or wobj.type != 'MESH':
        return 0
    nf = len(wobj.data.polygons)
    ids = np.zeros(nf, dtype=np.int32)
    vals = np.zeros((3, nf), dtype=np.float32)
    res = solver.results
    vm = res.von_mises or {}
    pt = res.principal_stresses_top or {}
    pb = res.principal_stresses_bot or {}
    for eid, p in conv._element_to_poly.items():
        if 0 <= p < nf:
            ids[p] = eid
            vals[0, p] = vm.get(eid, 0.0)
            if eid in pt and eid in pb:
                vals[1, p] = max(pt[eid][0], pb[eid][0])
                vals[2, p] = min(pt[eid][1], pb[eid][1])
    hosts = [wobj] + [o for o in bpy.data.objects
                      if o is not wobj and o.type == 'MESH' and _perm_is_result(o)
                      and o.get("perm_source_name") == conv.source_name
                      and len(o.data.polygons) == nf]
    for o in hosts:
        _perm_write_face_attr(o.data, _PERM_EID_ATTR, 'INT', ids)
        for k, name in enumerate(_PERM_EL_VALUES):
            _perm_write_face_attr(o.data, name, 'FLOAT', vals[k])
    return int((ids > 0).sum())


def _perm_find_hosts(src):
    """Meshes of this model that carry element numbers (result meshes and the working copy)."""
    return [o for o in bpy.data.objects
            if o.type == 'MESH' and o is not src and o.get("perm_source_name") == src.name
            and _PERM_EID_ATTR in o.data.attributes]


def _perm_face_attr_array(mesh, name, dtype):
    a = mesh.attributes.get(name)
    if a is None or a.domain != 'FACE':
        return None
    arr = np.zeros(len(mesh.polygons), dtype=dtype)
    if len(arr):
        a.data.foreach_get("value", arr)
    return arr


def _perm_face_world(obj, f):
    M = obj.matrix_world
    me = obj.data
    return [M @ me.vertices[v].co for v in me.polygons[f].vertices]


def _perm_frame_point(context, center, dist) -> None:
    """Centre the 3D view(s) on `center` at `dist` metres, keeping the view direction."""
    scr = context.screen
    areas = ([context.area] if context.area is not None and context.area.type == 'VIEW_3D'
             else [a for a in (scr.areas if scr is not None else []) if a.type == 'VIEW_3D'])
    for a in areas:
        for sp in a.spaces:
            if sp.type == 'VIEW_3D' and sp.region_3d is not None:
                r3d = sp.region_3d
                if r3d.view_perspective == 'CAMERA':
                    r3d.view_perspective = 'PERSP'
                r3d.view_location = center
                r3d.view_distance = dist
        a.tag_redraw()


def _perm_find_marked():
    """Corner points (world) of the marked element on every visible mesh that shows it
    (each visible result mesh, or the model), and its number. None if nothing is marked."""
    scn = bpy.context.scene
    srcn, f = scn.get("perm_find_src"), scn.get("perm_find_face")
    if not srcn or f is None:
        return None
    f = int(f)
    if scn.get("perm_find_mode") == 'MODEL':
        objs = [bpy.data.objects.get(srcn)]
    else:
        objs = [o for o in scn.objects if o.type == 'MESH' and o.get("perm_source_name") == srcn
                and _perm_is_result(o) and _PERM_EID_ATTR in o.data.attributes]
    faces = [_perm_face_world(o, f) for o in objs
             if o is not None and o.type == 'MESH' and o.visible_get() and f < len(o.data.polygons)]
    return (faces, int(scn.get("perm_find_eid", 0))) if faces else None


def _perm_find_draw_view():
    """POST_VIEW: outline and diagonals of the marked element, drawn through the mesh."""
    try:
        context = bpy.context
        if context.region is None or context.region_data is None:
            return
        mk = _perm_find_marked()
        if mk is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        segs = []
        for pts in mk[0]:
            for k in range(len(pts)):
                segs += [pts[k][:], pts[(k + 1) % len(pts)][:]]
            if len(pts) == 4:
                segs += [pts[0][:], pts[2][:], pts[1][:], pts[3][:]]
        try:
            sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            poly = True
        except Exception:
            sh = gpu.shader.from_builtin('UNIFORM_COLOR')
            poly = False
        batch = batch_for_shader(sh, 'LINES', {"pos": segs})
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        sh.bind()
        if poly:
            ui = (context.preferences.system.ui_scale or 1.0)
            sh.uniform_float("viewportSize", (context.region.width, context.region.height))
            sh.uniform_float("lineWidth", 3.0 * ui)
        sh.uniform_float("color", (1.0, 0.1, 0.85, 1.0))
        batch.draw(sh)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


def _perm_find_draw_label():
    """POST_PIXEL: '#number' next to the marked element."""
    try:
        context = bpy.context
        region, rv3d = context.region, context.region_data
        if region is None or rv3d is None:
            return
        mk = _perm_find_marked()
        if mk is None:
            return
        faces, eid = mk
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        import blf
        ui = (context.preferences.system.ui_scale or 1.0)
        fid = 0
        _perm_blf_size(fid, 14 * ui)
        try:
            blf.enable(fid, blf.SHADOW)
            blf.shadow(fid, 3, 0.0, 0.0, 0.0, 0.9)
            blf.shadow_offset(fid, 1, -1)
        except Exception:
            pass
        blf.color(fid, 1.0, 0.35, 0.9, 1.0)
        done = []
        for pts in faces:
            p2 = location_3d_to_region_2d(region, rv3d, sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts))
            if p2 is None or any(abs(p2[0] - q[0]) < 4 and abs(p2[1] - q[1]) < 4 for q in done):
                continue
            done.append(p2)
            blf.position(fid, p2[0] + 12 * ui, p2[1] + 8 * ui, 0)
            blf.draw(fid, f"#{eid}")
        try:
            blf.disable(fid, blf.SHADOW)
        except Exception:
            pass
    except Exception:
        pass


def _perm_find_handlers_add() -> None:
    _perm_find_handlers_remove()
    try:
        hv = bpy.types.SpaceView3D.draw_handler_add(_perm_find_draw_view, (), 'WINDOW', 'POST_VIEW')
        hp = bpy.types.SpaceView3D.draw_handler_add(_perm_find_draw_label, (), 'WINDOW', 'POST_PIXEL')
        bpy.app.driver_namespace[_PERM_FIND_NS_KEY] = (hv, hp)
    except Exception as e:
        print(f"[PERM] Could not add the Find Element overlay: {e}")


def _perm_find_handlers_remove() -> None:
    hs = bpy.app.driver_namespace.pop(_PERM_FIND_NS_KEY, None)
    for h in (hs or ()):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass


class PERM_OT_FindElement(bpy.types.Operator):
    """Find an element by its number (the '#' in the console's Top 10 list and in the
    results report), mark it in magenta and zoom the view to it"""
    bl_idname = "perm.find_element"
    bl_label = "PERM: Find Element"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scn = context.scene
        eid = int(scn.perm_find_element)
        src = _perm_resolve_source(context)
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        hosts = _perm_find_hosts(src) if src is not None else []
        if not hosts:          # the active mesh is the working copy, or another mesh: use the model last analysed
            for nm in (getattr(src, "get", lambda k: None)("perm_source_name"), scn.get("perm_peak_source")):
                o = bpy.data.objects.get(nm) if nm else None
                if o is not None and o.type == 'MESH' and _perm_find_hosts(o):
                    src, hosts = o, _perm_find_hosts(o)
                    break
        if src is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model) first"); return {'CANCELLED'}
        if not hosts:          # the result meshes carry no element numbers yet: tag them from the last analysis
            try:
                import __main__
                sv = getattr(__main__, '_PERM_LAST_SOLVER', None)
                if sv is not None and getattr(sv.converter, 'source_name', None) == src.name:
                    _perm_tag_element_ids(sv)
                    hosts = _perm_find_hosts(src)
            except Exception as e:
                print(f"[PERM Find] could not number the elements: {e}")
        if not hosts:
            self.report({'WARNING'}, "No element numbers for this model yet - run the analysis first")
            return {'CANCELLED'}
        # undeformed geometry: the working copy if it is there
        base = next((o for o in hosts if not _perm_is_result(o)), hosts[0])
        ids = _perm_face_attr_array(base.data, _PERM_EID_ATTR, np.int32)
        hit = np.nonzero(ids == eid)[0] if ids is not None else []
        if not len(hit):
            emax = int(ids.max()) if ids is not None and len(ids) else 0
            self.report({'WARNING'}, f"Element {eid} does not exist - this model has elements 1 to {emax}")
            return {'CANCELLED'}
        f = int(hit[0])
        pts = _perm_face_world(base, f)
        c0 = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)

        # where to show it
        target, tface, note = None, None, ""
        if scn.perm_find_show == 'RESULT':
            res = [o for o in hosts if _perm_is_result(o) and o.visible_get()
                   and len(o.data.polygons) == len(base.data.polygons)]
            if res:            # marked on every visible result mesh; the view centres on the active one
                act = context.active_object
                target = act if act in res else next((o for o in res if o.get("perm_result_viz")), res[0])
                tface = f
                scn["perm_find_mode"] = 'RESULT'
            else:
                note = " (no result mesh visible - shown in the model)"
        if target is None:     # in the model, Edit Mode, the face selected
            me = src.data
            sf = None
            if len(me.polygons) > f:
                cs = sum(_perm_face_world(src, f), Vector((0.0, 0.0, 0.0))) / len(me.polygons[f].vertices)
                if (cs - c0).length < 1e-6:
                    sf = f
            if sf is None:
                from mathutils import kdtree
                M = src.matrix_world
                kd = kdtree.KDTree(len(me.polygons))
                for p in me.polygons:
                    kd.insert(M @ p.center, p.index)
                kd.balance()
                sf = kd.find(c0)[1]
            for o in context.scene.objects:
                if o is not src and o.get("perm_source_name") == src.name and _perm_is_result(o):
                    o.hide_set(True)
            src.hide_set(False)
            for o in context.selected_objects:
                o.select_set(False)
            src.select_set(True)
            context.view_layer.objects.active = src
            nv, ne, npl = len(me.vertices), len(me.edges), len(me.polygons)
            me.vertices.foreach_set("select", np.zeros(nv, dtype=bool))
            me.edges.foreach_set("select", np.zeros(ne, dtype=bool))
            me.polygons.foreach_set("select", np.zeros(npl, dtype=bool))
            poly = me.polygons[sf]
            poly.select = True
            for vi in poly.vertices:
                me.vertices[vi].select = True
            for li in poly.loop_indices:
                me.edges[me.loops[li].edge_index].select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')
            target, tface = src, sf
            scn["perm_find_mode"] = 'MODEL'

        scn["perm_find_src"] = src.name
        scn["perm_find_face"] = int(tface)
        scn["perm_find_eid"] = eid
        tp = _perm_face_world(target, tface)
        _perm_frame_point(context, sum(tp, Vector((0.0, 0.0, 0.0))) / len(tp),
                          float(scn.perm_find_distance))

        # what it is
        lines = []
        mat = sec = "?"
        tm = _read_face_tags(base.data, 'material')
        ts = _read_face_tags(base.data, 'section')
        if tm is not None and tm[f]:
            mat = _face_tag_names(base, 'material').get(int(tm[f]), "?")
        if ts is not None and ts[f]:
            sec = _face_tag_names(base, 'section').get(int(ts[f]), "?")
        thk = next((x.thickness for x in scn.perm_sections if x.name == sec), None)
        lines.append(f"#{eid}: {mat} | section {sec}" + (f" (t = {thk * 1000:.0f} mm)" if thk else ""))
        v = [_perm_face_attr_array(base.data, n, np.float32) for n in _PERM_EL_VALUES]
        if all(a is not None for a in v):
            lines.append(f"von Mises {v[0][f] / 1e6:.1f} | σ1 {v[1][f] / 1e6:+.1f} | "
                         f"σ2 {v[2][f] / 1e6:+.1f} MPa (worse of top/bottom)")
        lines.append(f"at x {c0.x:.3f}, y {c0.y:.3f}, z {c0.z:.3f} m (undeformed)")
        edges = [(pts[(k + 1) % len(pts)] - pts[k]).length for k in range(len(pts))]
        q = f"{len(pts)} corners, edges {min(edges) * 1e3:.0f}/{max(edges) * 1e3:.0f} mm"
        if len(pts) in (3, 4):
            try:
                q += f", J {PERMMeshConverter._scaled_jacobian(pts):+.2f}"
            except Exception:
                pass
        lines.append(q)
        scn["perm_find_info"] = "\n".join(lines)
        print("[PERM Find] " + "\n[PERM Find]   ".join(lines) +
              f"\n[PERM Find]   shown on '{target.name}', face {tface}")
        self.report({'INFO'}, f"PERM: element #{eid} on '{target.name}'{note}")
        return {'FINISHED'}


class PERM_OT_ClearFindElement(bpy.types.Operator):
    """Remove the Find Element mark"""
    bl_idname = "perm.clear_find_element"
    bl_label = "PERM: Clear Element Mark"
    bl_options = {'REGISTER'}

    def execute(self, context):
        for k in ("perm_find_src", "perm_find_mode", "perm_find_face", "perm_find_eid", "perm_find_info"):
            if k in context.scene:
                del context.scene[k]
        for a in (context.screen.areas if context.screen is not None else []):
            if a.type == 'VIEW_3D':
                a.tag_redraw()
        return {'FINISHED'}


class PERM_OT_EditModel(bpy.types.Operator):
    """Show the analysis model, hide the result meshes and select the model,
    ready for material / section / support / load assignments"""
    bl_idname = "perm.edit_model"
    bl_label = "PERM: Edit Model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        src = _perm_resolve_source(context)
        if src is None:
            self.report({'ERROR'}, "No model found - select the mesh to analyse"); return {'CANCELLED'}
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.scene.objects:
            if o is not src and o.get("perm_source_name") == src.name:
                o.hide_set(True)
        src.hide_set(False)
        for o in context.selected_objects:
            o.select_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        context.scene["perm_last_source"] = src.name
        return {'FINISHED'}


class PERM_OT_ShowResults(bpy.types.Operator):
    """Hide the model and show its result meshes again"""
    bl_idname = "perm.show_results"
    bl_label = "PERM: Show Results"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        src = _perm_resolve_source(context)
        if src is None:
            return {'CANCELLED'}
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        shown = 0
        for o in context.scene.objects:
            if o.get("perm_source_name") == src.name and _perm_is_result(o):
                o.hide_set(False); shown += 1
        if shown:
            src.hide_set(True)
        else:
            self.report({'WARNING'}, "No results for this model yet - run the analysis")
        return {'FINISHED'}


class PERM_OT_ClearResults(bpy.types.Operator):
    """Delete the result meshes and this model's hidden working copy, and show
    the model again. Your own meshes are never deleted"""
    bl_idname = "perm.clear_results"
    bl_label = "PERM: Clear Results"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        src = _perm_resolve_source(context)
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # Never by name: a user's model can itself be called "..._FEA_Base".
        doomed = [o for o in bpy.data.objects
                  if o is not src and (
                      _perm_is_result(o)
                      or (src is not None and o.get("perm_working_copy") and not o.select_get()
                          and o.get("perm_source_name") == src.name and not o.visible_get()))]
        for o in doomed:
            _perm_remove_object(o)
        if src is not None:
            src.hide_set(False)
            src.select_set(True)
            context.view_layer.objects.active = src
        self.report({'INFO'}, f"PERM: removed {len(doomed)} result/working object(s)")
        return {'FINISHED'}


class PERM_OT_RefreshVisualization(bpy.types.Operator):
    """PERM: Update / rebuild the deformation visualization.

    If a deformation-viz mesh already exists, this just re-applies the current
    Deformation Scale to it in place (the slider already does this live; the
    button is a manual fallback). If none exists, it rebuilds one from the last
    solve. Either way it never stacks duplicate meshes.
    """
    bl_idname = "perm.refresh_visualization"
    bl_label = "PERM: Refresh Visualization"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scn = context.scene
        scale = float(getattr(scn, 'perm_deformation_scale', 100.0))

        # Fast path: a deformation-viz object already exists → rescale in place.
        existing = [o for o in bpy.data.objects if o.get("perm_disp_viz")]
        if existing:
            for o in existing:
                _apply_disp_scale_to_object(o, scale)
            self.report({'INFO'},
                f"PERM: Rescaled deformation viz in place (×{scale:.1f})")
            return {'FINISHED'}

        # Nothing to rescale yet → rebuild from the most recent solve.
        import __main__
        solver = getattr(__main__, '_PERM_LAST_SOLVER', None)
        if solver is None:
            self.report({'ERROR'},
                "No deformation viz to update, and no solve to rebuild from. "
                "Click RUN ANALYSIS first.")
            return {'CANCELLED'}
        try:
            _src = solver.converter.working_object or solver.converter.source_object
            _ = _src.name                              # raises if the mesh of that analysis was deleted
        except (ReferenceError, AttributeError):
            self.report({'ERROR'}, "The last analysis no longer matches the scene (its mesh was removed). "
                                   "Click RUN ANALYSIS again.")
            return {'CANCELLED'}
        try:
            solver.visualize_displacement('magnitude', scale_factor=scale, deformed=True)
            self.report({'INFO'},
                f"PERM: Built deformation viz at scale ×{scale:.1f}")
        except Exception as e:
            self.report({'ERROR'}, f"Refresh failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        return {'FINISHED'}


def _perm_last_solver_for(obj):
    """The solver of this session's last analysis, if it was run on obj and its mesh still exists; else None."""
    import __main__
    sv = getattr(__main__, '_PERM_LAST_SOLVER', None)
    if sv is None:
        return None
    try:
        src = sv.converter.source_object
        if src is None or src.name != obj.name:
            return None
        _ = (sv.converter.working_object or src).name
    except (ReferenceError, AttributeError):
        return None
    return sv


def _perm_model_dict(solver, obj, include_results: bool) -> dict:
    """The model as analysed, as plain Python data (JSON-ready). SI units."""
    conv = solver.converter
    res = solver.results
    D = solver.DOFS_PER_NODE
    i2n = solver.index_to_node_id
    sec_of = _element_face_tag_names(conv, 'section')
    sec_of.update(getattr(conv, '_plate_sections', None) or {})
    data = {
        'format': 'PERM_FEA_v2',
        'generator': f"PERM {PERM_VERSION} (Blender {bpy.app.version_string})",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_object': obj.name,
        'units': {'length': 'm', 'force': 'N', 'moment': 'N·m', 'stress': 'Pa', 'density': 'kg/m3',
                  'rotation': 'rad'},
        'dofs_per_node': ['ux', 'uy', 'uz', 'rx', 'ry', 'rz'],
        'options': _perm_options_dict(),
        'materials': [{'id': int(mid), 'name': m.name, 'E': float(m.youngs_modulus), 'nu': float(m.poissons_ratio),
                       'rho': float(m.density)} for mid, m in sorted(conv.materials.items())],
        'nodes': [{'id': int(i2n[k]), 'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2])}
                  for k, q in enumerate(solver.node_coords)],
        'elements': [], 'supports': [], 'loads': [],
    }
    for eid, e in sorted(conv.elements.items()):
        if getattr(e, 'viz_only', False):
            continue
        row = {'id': int(eid), 'type': e.element_type.name, 'nodes': [int(n) for n in e.node_ids],
               'material_id': int(e.material_id)}
        if _perm_is_line(e):
            sec = conv.sections.get(e.section_id) if e.section_id else None
            row['section'] = str(e.section_id) if e.section_id else None
            if sec:
                row['area'] = float(sec['beam']['A'] if sec.get('beam') else sec.get('area', 0.0))
                if sec.get('beam'):
                    row['beam'] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else
                                       (np.asarray(v, float).tolist() if isinstance(v, (list, tuple, np.ndarray)) else str(v)))
                                   for k, v in sec['beam'].items()}
        else:
            row['thickness'] = float(e.thickness)
            row['section'] = sec_of.get(eid)
            if abs(getattr(e, 'offset', 0.0) or 0.0) > 0.0:
                row['offset'] = float(e.offset)
            if getattr(e, 'membrane_only', False):
                row['membrane_only'] = True
            if getattr(e, 'bending_only', False):
                row['bending_only'] = True
            if getattr(e, 'rebar_ratio', 0.0) and getattr(e, 'steel_modulus', 0.0):
                row['rebar'] = {'ratio': float(e.rebar_ratio), 'cover': float(e.rebar_cover),
                                'E_steel': float(e.steel_modulus)}
        data['elements'].append(row)
    for bc in solver.boundary_conditions:
        dofs = solver._get_constrained_local_dofs(bc.bc_type, bc.prescribed_values)
        ids = sorted(int(n) for n in conv.node_sets.get(bc.node_set, set()) if n in solver.node_id_to_index)
        data['supports'].append({'name': bc.node_set.replace("support_", ""), 'type': bc.bc_type.name,
                                 'fixed_dofs': [solver.DOF_NAMES[d] for d in dofs], 'nodes': ids})
    pads = getattr(solver, '_bearing_pads', None) or []
    if pads:
        data['bearing_pads'] = [{'group': str(pd['group']).replace("support_", ""), 'support_node': int(pd['node']),
                                 'nodes': [int(i2n[int(k)]) for k in pd['nodes']],
                                 'weights': [float(w) for w in pd['weights']]} for pd in pads]
        data['bearing_pads_note'] = ("each pad restrains the weighted mean displacement of its nodes in the "
                                     "support's fixed directions (not every node)")
    # every load case as the nodal forces it puts into the force vector (patch spreading included)
    import io, contextlib
    F_saved = solver._F_global
    feq_saved = dict(getattr(solver, '_beam_feq', {}) or {})
    try:
        for lc in solver.load_cases:
            solver._F_global = np.zeros(solver.n_dofs)
            with contextlib.redirect_stdout(io.StringIO()):
                if lc.load_type == LoadType.NODAL_FORCE:
                    solver._apply_nodal_force(lc)
                elif lc.load_type == LoadType.GRAVITY:
                    solver._apply_gravity(lc)
                elif lc.load_type == LoadType.PRESSURE:
                    solver._apply_pressure(lc)
                elif lc.load_type == LoadType.DISTRIBUTED:
                    solver._apply_surface_load(lc)
            F = solver._F_global.reshape(-1, D)
            nz = np.nonzero(np.abs(F).sum(1) > 0.0)[0]
            item = {'name': lc.name, 'type': {'NODAL_FORCE': 'point', 'GRAVITY': 'self_weight', 'PRESSURE': 'pressure',
                                              'DISTRIBUTED': 'surface'}.get(lc.load_type.name, lc.load_type.name),
                    'total_force': F[:, :3].sum(0).tolist(),
                    'nodal_forces': [[int(i2n[int(k)])] + F[k].tolist() for k in nz],
                    'nodal_forces_columns': ['node', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']}
            if lc.load_type == LoadType.GRAVITY:
                item['acceleration'] = [float(c) for c in lc.gravity_vector]
            if lc.load_type == LoadType.DISTRIBUTED:
                item['element_tractions'] = [[int(k)] + np.asarray(v, float).tolist()
                                             for k, v in sorted(getattr(lc, 'element_loads', {}).items())]
                item['element_tractions_columns'] = ['element', 'qx', 'qy', 'qz']
            data['loads'].append(item)
    finally:
        solver._F_global = F_saved
        solver._beam_feq = feq_saved
    cf = getattr(res, 'connection_forces', None) or {}
    if cf:
        data['connections'] = [{'name': str(nm), 'type': str(c.get('ctype')), 'method': str(c.get('method')),
                                'centre': np.asarray(c.get('centre'), float).tolist()} for nm, c in cf.items()]
    detail_info = getattr(solver, '_detail_info', None) or {}
    if detail_info.get('ties') or detail_info.get('plates') or getattr(conv, '_plate_ties', None):
        data['internal_ties_note'] = ("the analysed model also holds rigid ties (base plates, layer ties) that are "
                                      "not listed node by node in this file: see 'options' and the results report")
    if include_results and res is not None and res.displacements is not None:
        def per_elem(dct, scalar=False):
            if not dct:
                return None
            return {str(k): (float(v) if scalar else np.asarray(v, float).tolist()) for k, v in dct.items()}
        R = res.reaction_forces
        rz = np.nonzero(np.abs(R).sum(1) > 0.0)[0] if R is not None else []
        out = {
            'displacements_columns': ['node', 'ux', 'uy', 'uz', 'rx', 'ry', 'rz'],
            'displacements': [[int(i2n[k])] + np.asarray(u, float).tolist() for k, u in enumerate(res.displacements)],
            'reactions_columns': ['node', 'Rx', 'Ry', 'Rz', 'Mx', 'My', 'Mz'],
            'reactions': [[int(i2n[int(k)])] + np.asarray(R[k], float).tolist() for k in rz],
            'max_displacement': float(res.max_displacement), 'max_uz': float(res.max_uz),
            'max_von_mises': float(res.max_von_mises),
            'max_tension': float(res.max_principal_tension),
            'min_compression': float(res.min_principal_compression),
            'strain_energy': float(getattr(res, 'total_strain_energy', 0.0)),
            'element_results_note': "per element id; stresses [sxx, syy, txy] in the element's LOCAL axes at its "
                                    "centre, top = +z face; principal [s1, s2]; moments [Mx, My, Mxy] per unit width",
        }
        for key, dct, sc in (('stresses_mid', res.stresses, False), ('stresses_top', res.stresses_top, False),
                             ('stresses_bot', res.stresses_bot, False), ('moments', res.moments, False),
                             ('von_mises', res.von_mises, True), ('principal_top', res.principal_stresses_top, False),
                             ('principal_bot', res.principal_stresses_bot, False),
                             ('truss_axial_force', res.truss_axial_forces, True)):
            v = per_elem(dct, sc)
            if v:
                out[key] = v
        data['results'] = out
    return data


class PERM_OT_ExportJSON(bpy.types.Operator):
    """Write the model AS ANALYSED to a JSON file: nodes, elements with their material, thickness and offset, supports, every load case as nodal forces and, on request, the results. For checking the model in another program. SI units"""
    bl_idname = "perm.export_json"
    bl_label = "PERM: Export FEA JSON"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(
        name="Output File",
        description="Path of the JSON file ('//' = next to the .blend file)",
        subtype='FILE_PATH',
        default="//perm_model.json"
    )
    solve_first: bpy.props.BoolProperty(
        name="Run the analysis first",
        description="On: the model is analysed now with the settings in the panel (no result meshes are made), so "
                    "the file matches the panel. Off: the last analysis of this session is written as it was run",
        default=True)
    include_results: bpy.props.BoolProperty(
        name="Include results",
        description="Also write displacements, reactions and the stresses of every element (a large file on a "
                    "large mesh)",
        default=False)

    def execute(self, context):
        obj = _perm_resolve_source(context)
        if obj is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model) first")
            return {'CANCELLED'}
        if _perm_need_scipy(self):
            return {'CANCELLED'}
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if not find_perm_support_groups(obj):
            self.report({'ERROR'}, "Assign at least one restraint first (4. Restraints)")
            return {'CANCELLED'}
        solver = None if self.solve_first else _perm_last_solver_for(obj)
        if solver is None:
            if not self.solve_first:
                print("[PERM Export] no analysis of this model in this session: running it now")
            scn = context.scene
            kw = {k: False for k in PERM_OT_RunSolver.__annotations__ if k.startswith("viz_")}
            try:
                for o in context.view_layer.objects:
                    o.select_set(False)
                obj.hide_set(False); context.view_layer.objects.active = obj; obj.select_set(True)
                ret = bpy.ops.perm.run_solver('EXEC_DEFAULT', apply_gravity=bool(scn.perm_apply_gravity),
                                              element_type=str(scn.perm_mesh_topology), **kw)
            except RuntimeError as e:
                self.report({'ERROR'}, f"PERM Export: the analysis did not run - {str(e).strip().splitlines()[-1]}")
                return {'CANCELLED'}
            if 'FINISHED' not in ret:
                self.report({'ERROR'}, "PERM Export: the analysis did not run (see the message above)")
                return {'CANCELLED'}
            solver = _perm_last_solver_for(obj)
            if solver is None:
                self.report({'ERROR'}, "PERM Export: no analysis result to write")
                return {'CANCELLED'}
        try:
            data = _perm_model_dict(solver, obj, bool(self.include_results))
            fp = (self.filepath or "").strip() or "//perm_model.json"
            if fp.startswith("//") and not bpy.data.filepath:        # unsaved .blend: next to the reports
                fp = os.path.join(_perm_results_dir(context.scene), fp[2:] or "perm_model.json")
            filepath = bpy.path.abspath(fp)
            if not filepath.lower().endswith(".json"):
                filepath += ".json"
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
            n_with_results = " (with results)" if 'results' in data else ""
            self.report({'INFO'}, f"PERM: exported{n_with_results} → {filepath}")
            print(f"[PERM] Exported FEA JSON: {filepath}")
            print(f"       {len(data['nodes'])} nodes, {len(data['elements'])} elements, "
                  f"{len(data['supports'])} supports, {len(data['loads'])} load cases")
        except Exception as e:
            self.report({'ERROR'}, f"PERM Export Error: {str(e)}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        box = layout.box(); box.label(text="Output File", icon='FILE_FOLDER')
        box.prop(self, "filepath")
        box = layout.box(); box.label(text="What is written", icon='OUTPUT')
        box.prop(self, "solve_first")
        box.prop(self, "include_results")
        box.label(text="Materials, sections, supports and loads: as in the panel", icon='INFO')


# ─────────────────────────────────────────────────────────────────────────────
# SHELL OFFSET
# ─────────────────────────────────────────────────────────────────────────────
# A reality-captured mesh is the OUTER surface of the concrete. A shell element
# puts half of its thickness on each side of its nodes, so a deck slab that was
# scanned only on its top carries half of its concrete ABOVE the deck, and the
# slab's lever arm about the neutral axis is too long. A shell offset moves the
# element's mid-surface off the nodes, along the face normal, without moving
# any captured vertex.
#
# Stored per face as the FLOAT_VECTOR attribute 'perm_offset': the offset as a
# vector in the object's LOCAL coordinates. A vector, not a signed number,
# because Clean (normals_make_consistent) may flip face normals in the working
# copy; the vector does not care. The solver projects it on each element's own
# normal (_perm_apply_shell_offsets) and links nodes to the mid-surface with a
# rigid offset (PERMSolver._offset_matrix).
#
# Self-weight stays at the nodes: it is vertical, and on a deck the offset is
# vertical too, so moving it would change nothing.
PERM_OFFSET_ATTR = "perm_offset"
_PERM_PREVIEW_COLL = "PERM Thickness Preview"
_PERM_PREVIEW_NEW = "PERM preview - with offset (what PERM solves)"
_PERM_PREVIEW_OLD = "PERM preview - without offset (centred on the mesh)"


def _perm_read_face_offsets(mesh):
    """OBJECT MODE: (n_faces, 3) offset vectors (object local), or None."""
    attr = mesh.attributes.get(PERM_OFFSET_ATTR)
    if attr is None or attr.domain != 'FACE' or attr.data_type != 'FLOAT_VECTOR':
        return None
    v = np.zeros(3 * len(mesh.polygons))
    if len(v):
        attr.data.foreach_get('vector', v)
    return v.reshape(-1, 3)


def _perm_section_thickness_lut(obj) -> Dict[int, float]:
    """Face section tag id -> thickness (m) of that scene Cross-Section."""
    names = _face_tag_names(obj, 'section')
    secs = {s.name: float(s.thickness) for s in bpy.context.scene.perm_sections}
    return {i: secs[n] for i, n in names.items() if n in secs}


def _perm_face_thickness(obj) -> np.ndarray:
    """OBJECT MODE: per-face section thickness (m); NaN where no section."""
    me = obj.data
    th = np.full(len(me.polygons), np.nan)
    tags = _read_face_tags(me, 'section')
    if tags is None:
        return th
    for i, t in _perm_section_thickness_lut(obj).items():
        th[tags == i] = t
    return th


def _perm_world_normals(obj) -> np.ndarray:
    me = obj.data
    n = np.zeros(3 * len(me.polygons))
    me.polygons.foreach_get('normal', n)
    R = np.array(obj.matrix_world.to_3x3())
    nw = n.reshape(-1, 3) @ np.linalg.inv(R)          # rows of R^-T n
    nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-30)
    return nw


def _perm_count_offsets(obj) -> int:
    if obj.mode == 'EDIT':                           # mesh attributes are empty in Edit Mode
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bm.faces.layers.float_vector.get(PERM_OFFSET_ATTR)
        return 0 if lay is None else sum(1 for f in bm.faces if Vector(f[lay]).length > 0.0)
    D = _perm_read_face_offsets(obj.data)
    return 0 if D is None else int(np.count_nonzero(np.abs(D).sum(1) > 0.0))


def _perm_apply_shell_offsets(converter) -> int:
    """Face attribute 'perm_offset' -> FEElement.offset (signed, along the
    element normal, model units). Returns the number of offset elements."""
    for e in converter.elements.values():
        e.offset = 0.0
    wobj = getattr(converter, 'working_object', None)
    if wobj is None:
        return 0
    D = _perm_read_face_offsets(wobj.data)
    if D is None or not np.any(D):
        return 0
    R = np.array(wobj.matrix_world.to_3x3()) * float(getattr(converter, 'scale_factor', 1.0) or 1.0)
    nodes = converter.nodes
    n_off = n_over = n_tilt = 0
    hist = Counter()
    for eid, elem in converter.elements.items():
        if (_perm_is_line(elem)
                or getattr(elem, 'viz_only', False)):
            continue
        poly = converter._element_to_poly.get(eid)
        if poly is None or poly >= len(D):
            continue
        d_loc = D[poly]
        if not d_loc.any():
            continue
        dw = R @ d_loc
        p0, p1, p2 = (np.array([nodes[n].x, nodes[n].y, nodes[n].z]) for n in elem.node_ids[:3])
        ez = np.cross(p1 - p0, p2 - p0)            # same normal as _compute_local_system
        if len(elem.node_ids) >= 4 and not getattr(elem.element_type, 'name', '').startswith('TRI'):
            p3 = np.array([nodes[elem.node_ids[3]].x, nodes[elem.node_ids[3]].y, nodes[elem.node_ids[3]].z])
            nd = np.cross(p2 - p0, p3 - p1)        # quads use the mean plane
            if np.linalg.norm(nd) > 1e-15:
                ez = nd
        nz = np.linalg.norm(ez)
        if nz < 1e-15:
            continue
        ez /= nz
        e = float(dw @ ez)
        if abs(e) < 0.9 * float(np.linalg.norm(dw)):
            n_tilt += 1                             # face re-shaped after the offset was set
        elem.offset = e
        n_off += 1
        if abs(e) > 0.5 * float(elem.thickness) + 1e-6:
            n_over += 1
        hist[(round(float(elem.thickness), 4), round(abs(e) * 1000.0))] += 1
    if n_off:
        rows = ", ".join(f"{c} faces t={t:g} m off by {o:.0f} mm"
                         for (t, o), c in sorted(hist.items(), key=lambda kv: -kv[1])[:6])
        print(f"[PERM Solver] Shell offset on {n_off} elements (concrete moved off the scanned "
              f"surface, mesh unchanged): {rows}")
        if n_over:
            print(f"[PERM Solver]   NOTE {n_over} element(s) are offset by more than half their "
                  f"thickness: their concrete does not touch the scanned surface at all.")
        if n_tilt:
            print(f"[PERM Solver]   NOTE {n_tilt} element(s) carry an offset that is no longer "
                  f"normal to the face (face edited after Set Offset?): only the normal part is used.")
    return n_off


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE LOADS
# ─────────────────────────────────────────────────────────────────────────────
# A force per unit area on faces: a live load on a floor, wind or water pressure on a wall. Stored per face
# as the FLOAT_VECTOR attribute 'perm_surface_load': the traction (N/m2) in WORLD axes, per unit of the
# face's own area. A vector, as for the shell offset, so that it does not depend on the face normals of the
# working copy (the mesh clean-up may flip them). 'Along the face normal' is turned into that vector when the
# load is set, with the normals the faces have then: a pressure p > 0 pushes AGAINST the normal (onto the side
# the normal points out of).
PERM_SURFLOAD_ATTR = "perm_surface_load"


def _perm_read_face_vectors(mesh, name):
    """OBJECT MODE: (n_faces, 3) values of a FLOAT_VECTOR face attribute, or None."""
    attr = mesh.attributes.get(name)
    if attr is None or attr.domain != 'FACE' or attr.data_type != 'FLOAT_VECTOR':
        return None
    v = np.zeros(3 * len(mesh.polygons))
    if len(v):
        attr.data.foreach_get('vector', v)
    return v.reshape(-1, 3)


def _perm_surface_load_info(obj):
    """OBJECT MODE: (number of loaded faces, resultant force in N, world axes) of the model's surface loads."""
    me = obj.data
    Q = _perm_read_face_vectors(me, PERM_SURFLOAD_ATTR)
    if Q is None or not np.any(Q):
        return 0, np.zeros(3)
    M = np.array(obj.matrix_world)
    co = np.zeros(3 * len(me.vertices)); me.vertices.foreach_get('co', co)
    W = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tv = np.zeros(3 * nt, dtype=np.int32); tp = np.zeros(nt, dtype=np.int32)
    me.loop_triangles.foreach_get('vertices', tv); me.loop_triangles.foreach_get('polygon_index', tp)
    T = W[tv.reshape(-1, 3)]
    ta = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    area = np.bincount(tp, weights=ta, minlength=len(me.polygons))
    loaded = np.abs(Q).sum(1) > 0.0
    return int(loaded.sum()), (Q * area[:, None]).sum(0)


def _perm_store_surface_load_info(obj) -> None:
    n, F = _perm_surface_load_info(obj)
    obj["perm_surface_load_info"] = [float(n), float(F[0]), float(F[1]), float(F[2])]


def _perm_apply_surface_loads(solver) -> int:
    """Face attribute 'perm_surface_load' of the working copy -> one DISTRIBUTED load case. Returns the number
    of loaded elements."""
    conv = solver.converter
    wobj = getattr(conv, 'working_object', None)
    if wobj is None:
        return 0
    Q = _perm_read_face_vectors(wobj.data, PERM_SURFLOAD_ATTR)
    if Q is None or not np.any(Q):
        return 0
    loads = {}
    for eid, elem in conv.elements.items():
        if _perm_is_line(elem) or getattr(elem, 'viz_only', False):
            continue
        poly = conv._element_to_poly.get(eid)
        if poly is None or poly >= len(Q) or not Q[poly].any():
            continue
        loads[eid] = Q[poly].copy()
    if loads:
        solver.add_surface_load(loads)
    return len(loads)


class PERM_OT_SetSurfaceLoad(bpy.types.Operator):
    """Put a force per unit area on the selected faces (5b. Surface Loads): in a global direction, or as a pressure along the face normals. The force on a face = value x the face's own area"""
    bl_idname = "perm.set_surface_load"
    bl_label = "Set Surface Load"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'ERROR'}, "Edit Mode on the model, select the faces first")
            return {'CANCELLED'}
        if _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        scn = context.scene
        mode = scn.perm_sl_mode
        q_glob = np.array([_perm_r6(c) for c in scn.perm_sl_vector])
        pres = _perm_r6(scn.perm_sl_pressure)
        add = bool(scn.perm_sl_add)
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        lay = (bm.faces.layers.float_vector.get(PERM_SURFLOAD_ATTR)
               or bm.faces.layers.float_vector.new(PERM_SURFLOAD_ATTR))
        Rinv = np.linalg.inv(np.array(obj.matrix_world.to_3x3()))
        n_set = n_bad = 0
        for f in bm.faces:
            if not f.select:
                continue
            if mode == 'NORMAL':
                nw = Rinv.T @ np.array(f.normal[:])          # world normal of the face
                ln = np.linalg.norm(nw)
                if ln < 1e-12:
                    n_bad += 1
                    continue
                q = -pres * nw / ln                          # pressure pushes against the normal
            else:
                q = q_glob
            if add:
                q = q + np.array(f[lay][:])
            f[lay] = Vector(q.tolist())
            n_set += 1
        bmesh.update_edit_mesh(me)
        obj.update_from_editmode()
        if n_set == 0:
            self.report({'ERROR'}, "No faces selected (Edit Mode, face select)")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        _perm_store_surface_load_info(obj)
        bpy.ops.object.mode_set(mode='EDIT')
        n, Fx, Fy, Fz = obj["perm_surface_load_info"]
        what = (f"pressure {pres:g} N/m² against the face normals" if mode == 'NORMAL'
                else f"({q_glob[0]:g}, {q_glob[1]:g}, {q_glob[2]:g}) N/m²")
        msg = (f"Surface load {'added to' if add else 'set on'} {n_set} faces: {what}. Model total: {int(n)} faces, "
               f"F = ({Fx / 1e3:+.3f}, {Fy / 1e3:+.3f}, {Fz / 1e3:+.3f}) kN")
        if n_bad:
            msg += f"; {n_bad} faces without a normal skipped"
        print("[PERM Surface load] " + msg)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class PERM_OT_ClearSurfaceLoad(bpy.types.Operator):
    """Remove the surface load from the selected faces (Edit Mode), or from all faces"""
    bl_idname = "perm.clear_surface_load"
    bl_label = "Clear Surface Load"
    bl_options = {'REGISTER', 'UNDO'}
    all_faces: bpy.props.BoolProperty(name="All faces", default=False)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            if not self.all_faces:
                self.report({'ERROR'}, "Edit Mode: select the faces to clear")
                return {'CANCELLED'}
            a = obj.data.attributes.get(PERM_SURFLOAD_ATTR)
            if a is not None:
                obj.data.attributes.remove(a)
            obj["perm_surface_load_info"] = [0.0, 0.0, 0.0, 0.0]
            self.report({'INFO'}, "All surface loads removed")
            return {'FINISHED'}
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bm.faces.layers.float_vector.get(PERM_SURFLOAD_ATTR)
        n = 0
        if lay is not None:
            for f in bm.faces:
                if (self.all_faces or f.select) and Vector(f[lay]).length > 0.0:
                    f[lay] = Vector((0.0, 0.0, 0.0)); n += 1
        bmesh.update_edit_mesh(obj.data)
        obj.update_from_editmode()
        bpy.ops.object.mode_set(mode='OBJECT')
        _perm_store_surface_load_info(obj)
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Surface load cleared on {n} faces")
        return {'FINISHED'}


class PERM_OT_SelectSurfaceLoadFaces(bpy.types.Operator):
    """Select every face that carries a surface load"""
    bl_idname = "perm.select_surface_load_faces"
    bl_label = "Select Loaded Faces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _perm_resolve_source(context)
        if obj is None:
            self.report({'ERROR'}, "Select the mesh to analyse (the model) first")
            return {'CANCELLED'}
        try:
            obj.hide_set(False)
        except Exception:
            pass
        context.view_layer.objects.active = obj
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bm.faces.layers.float_vector.get(PERM_SURFLOAD_ATTR)
        for seq in (bm.verts, bm.edges, bm.faces):
            for x in seq:
                x.select = False
        n = 0
        if lay is not None:
            for f in bm.faces:
                if Vector(f[lay]).length > 0.0:
                    f.select_set(True); n += 1
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"{n} faces carry a surface load")
        return {'FINISHED'}


def _perm_register_surface_load_props() -> None:
    S = bpy.types.Scene
    S.perm_sl_mode = bpy.props.EnumProperty(
        name="Direction", default='GLOBAL',
        items=[('GLOBAL', "Global direction", "The same force per area on every selected face, given by its "
                                              "global components (e.g. a floor load: 0, 0, -2000 N/m²)"),
               ('NORMAL', "Along the face normal", "A pressure: it pushes against each face's normal "
                                                   "(onto the side the normal points out of); negative = suction")])
    S.perm_sl_vector = bpy.props.FloatVectorProperty(
        name="q (N/m²)", size=3, default=(0.0, 0.0, -1000.0), subtype='XYZ', precision=1,
        description="Force per unit area in the global X, Y, Z directions, per unit of the face's own area "
                    "(1 kN/m² = 1000 N/m²)")
    S.perm_sl_pressure = bpy.props.FloatProperty(
        name="p (N/m²)", default=1000.0, precision=1,
        description="Pressure in N/m² (Pa). Positive pushes against the face normal; negative pulls (suction)")
    S.perm_sl_add = bpy.props.BoolProperty(
        name="Add to the load on the faces", default=False,
        description="On: the new value is added to what the selected faces already carry. Off: it replaces it")


def _perm_draw_surface_loads(layout, scn, obj) -> None:
    """Panel section 5b."""
    box = layout.box()
    box.label(text="5b. Surface Loads (force per area)", icon='MOD_FLUIDSIM')
    col = box.column(align=True)
    col.prop(scn, "perm_sl_mode", text="")
    if scn.perm_sl_mode == 'NORMAL':
        col.prop(scn, "perm_sl_pressure")
        _perm_label_wrapped(box, "p > 0 pushes against the face normal", icon='INFO')
    else:
        col.prop(scn, "perm_sl_vector", text="")
        _perm_label_wrapped(box, "N/m² of face area, along global X, Y, Z", icon='INFO')
    box.prop(scn, "perm_sl_add")
    box.label(text="Edit Mode → select faces → set:")
    row = box.row(align=True)
    row.operator("perm.set_surface_load", text="Set", icon='CHECKMARK')
    row.operator("perm.select_surface_load_faces", text="Select", icon='RESTRICT_SELECT_OFF')
    row.operator("perm.clear_surface_load", text="Clear", icon='X')
    info = obj.get("perm_surface_load_info")
    if info is not None and len(info) == 4 and info[0] > 0:
        col = box.column(align=True)
        col.label(text=f"{int(info[0])} faces loaded", icon='CHECKMARK')
        col.label(text=f"ΣF = ({info[1] / 1e3:+.2f}, {info[2] / 1e3:+.2f}, {info[3] / 1e3:+.2f}) kN")
    else:
        box.label(text="No surface loads yet", icon='INFO')


def _perm_find_layer_collection(lc, name):
    if lc.collection.name == name:
        return lc
    for ch in lc.children:
        r = _perm_find_layer_collection(ch, name)
        if r is not None:
            return r
    return None


def _perm_clear_thickness_preview() -> int:
    n = 0
    root = bpy.data.collections.get(_PERM_PREVIEW_COLL)
    if root is not None:
        for c in list(root.children) + [root]:
            for o in list(c.objects):
                me = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                n += 1
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
        for c in list(root.children):
            bpy.data.collections.remove(c)
        bpy.data.collections.remove(root)
    for o in bpy.data.objects:                       # give the model its look back
        if "perm_prev_display" in o.keys():
            try:
                o.display_type = o["perm_prev_display"]
            except Exception:
                pass
            del o["perm_prev_display"]
    return n


def _perm_preview_material(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m


def perm_build_thickness_preview(src, only_selected: bool = False, compare: bool = True) -> dict:
    """Solid preview of the concrete PERM puts on every face with a Cross-Section:
    thickness from the section, position from the shell offset. One object per
    (thickness, offset) with a Solidify modifier. The mesh is not changed."""
    if src.mode == 'EDIT':
        src.update_from_editmode()
    me = src.data
    nf = len(me.polygons)
    th = _perm_face_thickness(src)
    nw = _perm_world_normals(src)
    R = np.array(src.matrix_world.to_3x3())
    D = _perm_read_face_offsets(me)
    e = np.zeros(nf) if D is None else np.einsum('ij,ij->i', D @ R.T, nw)
    use = np.isfinite(th) & (th > 0)
    if only_selected:
        sel = np.zeros(nf, bool)
        me.polygons.foreach_get('select', sel)
        use &= sel
    if not use.any():
        raise RuntimeError("No faces with a Cross-Section to preview"
                           + (" in the selection" if only_selected else ""))
    _perm_clear_thickness_preview()
    co = np.zeros(3 * len(me.vertices)); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    ls = np.zeros(nf, np.int64); lt = np.zeros(nf, np.int64)
    me.polygons.foreach_get('loop_start', ls); me.polygons.foreach_get('loop_total', lt)
    lv = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
    # Solidify puts the concrete's centre at offset·t/2 along the face normal.
    # Faces are grouped by thickness, offset size AND the world direction the
    # concrete moves (dominant axis of the offset vector), and faces whose
    # concrete lies on their normal side are reversed: inside a group every
    # face then has its normal pointing away from its concrete. (One group
    # holding a slab moved up next to one moved down would average their
    # normals to zero where they meet.)
    o_rel = 2.0 * e / np.where(use, th, 1.0)
    flip = o_rel > 1e-6
    o_rel = -np.abs(o_rel)
    dvec = e[:, None] * nw
    dax = np.argmax(np.abs(dvec), axis=1)
    vert = np.abs(dvec[:, 2]) >= 0.5 * np.linalg.norm(dvec, axis=1)   # mostly up/down -> z group
    dax = np.where(vert, 2, dax)
    dsg = np.sign(dvec[np.arange(nf), dax])
    dkey = np.where(np.abs(e) > 1e-9, (dax + 1) * dsg, 0).astype(int)   # ±1 x, ±2 y, ±3 z, 0 centred
    scn = bpy.context.scene
    root = bpy.data.collections.new(_PERM_PREVIEW_COLL); scn.collection.children.link(root)
    c_new = bpy.data.collections.new(_PERM_PREVIEW_NEW); root.children.link(c_new)
    c_old = None
    if compare and D is not None and np.any(np.abs(e[use]) > 1e-9):
        c_old = bpy.data.collections.new(_PERM_PREVIEW_OLD); root.children.link(c_old)
    m_off = _perm_preview_material("PERM preview: offset", (0.17, 0.42, 0.69, 1.0))
    m_cen = _perm_preview_material("PERM preview: centred", (0.60, 0.60, 0.57, 1.0))
    m_old = _perm_preview_material("PERM preview: before", (0.80, 0.33, 0.20, 1.0))

    def make(coll, name, idx, t, o, mat, use_flip):
        faces = []
        for i in idx:
            f = lv[ls[i]:ls[i] + lt[i]]
            faces.append(f[::-1] if (use_flip and flip[i]) else f)
        allv = np.unique(np.concatenate(faces))
        remap = np.full(len(co), -1, np.int64); remap[allv] = np.arange(len(allv))
        pm = bpy.data.meshes.new(name)
        pm.from_pydata(co[allv].tolist(), [], [remap[f].tolist() for f in faces])
        if abs(o) < 1e-6:                            # centred: winding does not matter, make it consistent
            bm = bmesh.new(); bm.from_mesh(pm)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(pm); bm.free()
        o_s = max(-1.0, min(1.0, float(o)))          # Solidify reaches at most t/2 off the mesh ...
        excess = (float(o) - o_s) * float(t) / 2.0
        if abs(excess) > 1e-9:                       # ... move the preview surface for the rest
            pm.update()
            vn = np.zeros(3 * len(pm.vertices)); pm.vertices.foreach_get('normal', vn)
            vc = np.zeros(3 * len(pm.vertices)); pm.vertices.foreach_get('co', vc)
            Rl = np.linalg.inv(np.array(src.matrix_world.to_3x3()))   # world distance -> local
            vn = vn.reshape(-1, 3); vn_w = vn @ Rl                    # local normal -> world (rows of R^-T n)
            vn_w /= np.maximum(np.linalg.norm(vn_w, axis=1, keepdims=True), 1e-30)
            vc = vc.reshape(-1, 3) + (excess * vn_w) @ Rl.T
            pm.vertices.foreach_set('co', vc.ravel()); pm.update()
        pm.materials.append(mat)
        ob = bpy.data.objects.new(name, pm)
        ob.matrix_world = src.matrix_world.copy()
        coll.objects.link(ob)
        md = ob.modifiers.new("PERM thickness", 'SOLIDIFY')
        md.thickness = float(t); md.offset = o_s
        md.use_even_offset = False; md.use_quality_normals = True   # even offset spikes at sharp scan folds
        ob.hide_select = True
        ob["perm_preview"] = True
        return ob

    groups = defaultdict(list)
    for i in np.nonzero(use)[0]:
        groups[(round(float(th[i]), 4), round(float(o_rel[i]), 3), int(dkey[i]))].append(int(i))
    info = {}
    _dname = {1: "+x", -1: "-x", 2: "+y", -2: "-y", 3: "up", -3: "down"}
    for (t, o, dk), idx in sorted(groups.items()):
        off_mm = abs(o) * t / 2 * 1000
        name = f"t{t*1000:.0f}mm " + ("centred" if off_mm < 0.05
                                     else f"offset {off_mm:.0f}mm {_dname.get(dk, '')}")
        make(c_new, name, idx, t, o, m_cen if off_mm < 0.05 else m_off, True)
        info[name] = len(idx)
    if c_old is not None:                            # same face groups, concrete centred on the mesh
        for (t, o, dk), idx in sorted(groups.items()):
            off_mm = abs(o) * t / 2 * 1000
            name = f"t{t*1000:.0f}mm " + ("centred" if off_mm < 0.05 else f"{_dname.get(dk, '')}") + " before"
            make(c_old, name, idx, t, 0.0, m_old, True)
        lc = _perm_find_layer_collection(bpy.context.view_layer.layer_collection, c_old.name)
        if lc is not None:
            lc.hide_viewport = True                  # eye icon off: toggle it to compare
    if "perm_prev_display" not in src.keys():
        src["perm_prev_display"] = src.display_type
    src.display_type = 'WIRE'                        # scanned lines stay visible over the solids
    return info


class PERM_OT_SetShellOffset(bpy.types.Operator):
    """Move the thickness of the selected faces to one side of the mesh surface (shell offset). The mesh is not moved"""
    bl_idname = "perm.set_shell_offset"
    bl_label = "Set Shell Offset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'ERROR'}, "Edit Mode on the model, select the faces first")
            return {'CANCELLED'}
        if _perm_reject_copy(self, obj):             # a result copy never reaches the solver
            return {'CANCELLED'}
        scn = context.scene
        how, half, dist = scn.perm_offset_direction, scn.perm_offset_half, scn.perm_offset_distance
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        lay = (bm.faces.layers.float_vector.get(PERM_OFFSET_ATTR)
               or bm.faces.layers.float_vector.new(PERM_OFFSET_ATTR))
        slay = bm.faces.layers.int.get(PERM_FACE_ATTR['section'])
        lut = _perm_section_thickness_lut(obj) if half else {}
        R = np.array(obj.matrix_world.to_3x3()); Rinv = np.linalg.inv(R)
        n_set = n_nosec = n_vert = 0
        hs = Counter()
        for f in bm.faces:
            if not f.select:
                continue
            if half:
                t = lut.get(f[slay]) if slay is not None else None
                if not t:
                    n_nosec += 1
                    continue
                h = 0.5 * t
            else:
                h = float(dist)
            nw = Rinv.T @ np.array(f.normal[:])
            ln = np.linalg.norm(nw)
            if ln < 1e-12:
                continue
            nw /= ln
            if how in ('DOWN', 'UP'):
                if abs(nw[2]) < 0.3:
                    n_vert += 1
                    continue
                s = -np.sign(nw[2]) if how == 'DOWN' else np.sign(nw[2])
            else:
                s = 1.0 if how == 'ALONG' else -1.0
            f[lay] = Vector((Rinv @ (s * h * nw)).tolist())
            n_set += 1
            hs[round(h * 1000, 1)] += 1
        bmesh.update_edit_mesh(me)
        obj.update_from_editmode()
        obj["perm_offset_count"] = _perm_count_offsets(obj)
        det = ", ".join(f"{c} by {mm:g} mm" for mm, c in sorted(hs.items()))
        msg = f"Shell offset set on {n_set} faces ({det or 'none'})"
        if n_nosec:
            msg += f"; {n_nosec} skipped: no Cross-Section (assign one, or untick 'Half the section thickness')"
        if n_vert:
            msg += f"; {n_vert} skipped: vertical faces have no up/down (use Along/Against normal)"
        print("[PERM Offset] " + msg)
        self.report({'WARNING'} if (n_nosec or n_vert) else {'INFO'}, msg)
        return {'FINISHED'}


class PERM_OT_ClearShellOffset(bpy.types.Operator):
    """Centre the thickness of the selected faces on the mesh again (offset 0)"""
    bl_idname = "perm.clear_shell_offset"
    bl_label = "Clear Shell Offset"
    bl_options = {'REGISTER', 'UNDO'}
    all_faces: bpy.props.BoolProperty(name="All faces", default=False)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or _perm_reject_copy(self, obj):
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            if not self.all_faces:
                self.report({'ERROR'}, "Edit Mode: select the faces to clear")
                return {'CANCELLED'}
            a = obj.data.attributes.get(PERM_OFFSET_ATTR)
            if a is not None:
                obj.data.attributes.remove(a)
            obj["perm_offset_count"] = 0
            self.report({'INFO'}, "All shell offsets removed")
            return {'FINISHED'}
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bm.faces.layers.float_vector.get(PERM_OFFSET_ATTR)
        n = 0
        if lay is not None:
            for f in bm.faces:
                if (self.all_faces or f.select) and Vector(f[lay]).length > 0.0:
                    f[lay] = Vector((0.0, 0.0, 0.0)); n += 1
        bmesh.update_edit_mesh(obj.data)
        obj.update_from_editmode()
        obj["perm_offset_count"] = _perm_count_offsets(obj)
        self.report({'INFO'}, f"Shell offset cleared on {n} faces")
        return {'FINISHED'}


class PERM_OT_SelectOffsetFaces(bpy.types.Operator):
    """Select every face that carries a shell offset"""
    bl_idname = "perm.select_offset_faces"
    bl_label = "Select Offset Faces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _perm_resolve_source(context)
        if obj is None:
            return {'CANCELLED'}
        context.view_layer.objects.active = obj
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bm.faces.layers.float_vector.get(PERM_OFFSET_ATTR)
        for seq in (bm.verts, bm.edges, bm.faces):
            for x in seq:
                x.select = False
        n = 0
        if lay is not None:
            for f in bm.faces:
                if Vector(f[lay]).length > 0.0:
                    f.select_set(True); n += 1
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"{n} faces carry a shell offset")
        return {'FINISHED'}


class PERM_OT_PreviewThickness(bpy.types.Operator):
    """Show the thickness PERM will analyse as solids: section thickness, placed by the shell offset. The mesh is shown as wireframe"""
    bl_idname = "perm.preview_thickness"
    bl_label = "Preview Thickness"
    bl_options = {'REGISTER', 'UNDO'}
    only_selected: bpy.props.BoolProperty(name="Selected faces only", default=False)
    compare: bpy.props.BoolProperty(name="Also build 'without offset'", default=True)

    def execute(self, context):
        src = _perm_resolve_source(context)
        if src is None:
            self.report({'ERROR'}, "Select the model")
            return {'CANCELLED'}
        try:
            info = perm_build_thickness_preview(src, self.only_selected, self.compare)
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        det = ", ".join(f"{k}: {v}" for k, v in info.items())
        print(f"[PERM Preview] {det}")
        msg = f"Thickness preview built ({sum(info.values())} faces). "
        if bpy.data.collections.get(_PERM_PREVIEW_OLD):
            msg += "Outliner: turn on the eye of 'without offset' to compare. "
        msg += "Alt+B in the viewport cuts a cross-section."
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class PERM_OT_ClearThicknessPreview(bpy.types.Operator):
    """Delete the thickness preview objects and show the model normally again"""
    bl_idname = "perm.clear_thickness_preview"
    bl_label = "Clear Thickness Preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        n = _perm_clear_thickness_preview()
        self.report({'INFO'}, f"Removed {n} preview object(s)")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCALES, LEGEND, RESULTS REPORT
# ─────────────────────────────────────────────────────────────────────────────
_PERM_LEGEND_HANDLE = None
PERM_VERSION = "v32"

# Switches of the connection details. They are module values that every analysis sets from the scene options
# (_perm_apply_scene_options); what is written here is the state of a new model: everything off.
#
# Results report, third stress line: elements closer than PERM_LOCAL_ZONE_R to a support node, a point-load
# node or a railing-post base (steel-concrete joint) are left out. Stresses there are local artefacts of the
# idealised connection and grow as the mesh is refined.
PERM_LOCAL_ZONE_R = 0.0           # m; 0 switches the third line off (0.25 in the bridge preset)

PERM_PATCH_LOADS = False          # spread each point load (a load group of 1-3 nodes) over a loading patch
PERM_PATCH_SIZE = (0.25, 0.25)    # m: patch length along the span, width across
PERM_BEARING_PADS = False         # every node of a PINNED / ROLLER support group becomes a bearing pad
PERM_PAD_LENGTH = 0.20            # m along the span, from the support node inward (towards the other support)
PERM_PAD_WIDTH = 0.10             # m across (the web thickness): soffit nodes within half of it (+1 cm) of the node
PERM_POST_PLATES = False          # railing posts (steel faces meeting concrete faces) on base plates
PERM_PLATE_MODEL = "shell"        # "shell": steel plate bonded on the deck, tube welded to it; "rigid": slab nodes tied rigidly to the post
PERM_PLATE_SIZE = 0.15            # m, square, centred on the post (enlarged where the scanned base is wider)
PERM_PLATE_THICKNESS = 0.015      # m, steel ("shell": the plate's thickness, 2.6 kg each; "rigid": for the record only)
PERM_PLATE_E = 200.0e9            # Pa, plate steel
PERM_PLATE_NU = 0.3
PERM_PLATE_DENSITY = 7850.0       # kg/m3
PERM_TIE_OVERHANG = False         # tie each node of an offset lower layer to the upper-layer node above it
PERM_TIE_MAX_PLAN = 0.03          # m: largest plan distance of a tie (the two scanned surfaces are meshed independently)
PERM_TIE_MAX_RISE = 0.25          # m: largest vertical distance of a tie
PERM_RIGID_PENALTY = 1.0e4        # rigid ties: x the largest diagonal stiffness of the same DOF kind
PERM_PAD_PENALTY = 1.0e7          # bearing pads: x the largest diagonal stiffness (as the point supports)
# joints between mesh parts
PERM_CONN_PENALTY = 1.0e4         # joint links: x the largest diagonal stiffness of the same DOF kind (as the rigid ties)
PERM_CONN_LINE_TOL = 0.02         # a hinge joint is a LINE when every node lies within this share of its length (min 2 mm)

PERM_CONN_ROT_WARN = 0.02         # rad: a pinned / hinged joint that turns more than this is reported as a possible mechanism


# ── detail options of the scene ────────────────────────────────────────────────────────────────
PERM_GIRDER_CHECKS = False        # girder-bridge advice in the console; span, girder-line and midspan parts of the report

# scene property -> value in 'Bridge preset (paper settings)'
PERM_BRIDGE_PRESET = {
    "perm_opt_patch_loads": True, "perm_opt_patch_length": 0.25, "perm_opt_patch_width": 0.25,
    "perm_opt_bearing_pads": True, "perm_opt_pad_length": 0.20, "perm_opt_pad_width": 0.10,
    "perm_opt_post_plates": True, "perm_opt_plate_model": 'SHELL', "perm_opt_plate_size": 0.15,
    "perm_opt_plate_thickness": 0.015, "perm_opt_plate_E": 200.0e9, "perm_opt_plate_nu": 0.3,
    "perm_opt_plate_density": 7850.0,
    "perm_opt_tie_layers": True, "perm_opt_tie_max_plan": 0.03, "perm_opt_tie_max_rise": 0.25,
    "perm_opt_local_zone": 0.25, "perm_opt_girder_checks": True,
}
PERM_OPTION_SWITCHES = (("perm_opt_patch_loads", "patch loads"), ("perm_opt_bearing_pads", "bearing pads"),
                        ("perm_opt_post_plates", "post base plates"), ("perm_opt_tie_layers", "layer ties"),
                        ("perm_opt_girder_checks", "girder checks"))


def _perm_r6(v) -> float:
    """Scene properties are single precision (0.2 is stored as 0.20000000298): back to the typed value."""
    return float(f"{float(v):.6g}")


def _perm_apply_scene_options(scn) -> Dict[str, object]:
    """Copy the scene's detail options into the module switches the solver reads; called at every analysis.
    Returns what was set (for the report and the JSON export)."""
    g = globals()

    def val(key):
        return getattr(scn, key, PERM_BRIDGE_PRESET[key] if not isinstance(PERM_BRIDGE_PRESET[key], bool) else False)

    g['PERM_PATCH_LOADS'] = bool(val("perm_opt_patch_loads"))
    g['PERM_PATCH_SIZE'] = (_perm_r6(val("perm_opt_patch_length")), _perm_r6(val("perm_opt_patch_width")))
    g['PERM_BEARING_PADS'] = bool(val("perm_opt_bearing_pads"))
    g['PERM_PAD_LENGTH'] = _perm_r6(val("perm_opt_pad_length"))
    g['PERM_PAD_WIDTH'] = _perm_r6(val("perm_opt_pad_width"))
    g['PERM_POST_PLATES'] = bool(val("perm_opt_post_plates"))
    g['PERM_PLATE_MODEL'] = "rigid" if str(val("perm_opt_plate_model")).upper() == 'RIGID' else "shell"
    g['PERM_PLATE_SIZE'] = _perm_r6(val("perm_opt_plate_size"))
    g['PERM_PLATE_THICKNESS'] = _perm_r6(val("perm_opt_plate_thickness"))
    g['PERM_PLATE_E'] = _perm_r6(val("perm_opt_plate_E"))
    g['PERM_PLATE_NU'] = _perm_r6(val("perm_opt_plate_nu"))
    g['PERM_PLATE_DENSITY'] = _perm_r6(val("perm_opt_plate_density"))
    g['PERM_TIE_OVERHANG'] = bool(val("perm_opt_tie_layers"))
    g['PERM_TIE_MAX_PLAN'] = _perm_r6(val("perm_opt_tie_max_plan"))
    g['PERM_TIE_MAX_RISE'] = _perm_r6(val("perm_opt_tie_max_rise"))
    g['PERM_LOCAL_ZONE_R'] = max(0.0, _perm_r6(getattr(scn, "perm_opt_local_zone", 0.0)))
    g['PERM_GIRDER_CHECKS'] = bool(val("perm_opt_girder_checks"))
    return _perm_options_dict()


def _perm_options_dict() -> Dict[str, object]:
    """The detail options as the solver has them now."""
    return dict(patch_loads=PERM_PATCH_LOADS, patch_size_m=list(PERM_PATCH_SIZE),
                bearing_pads=PERM_BEARING_PADS, pad_length_m=PERM_PAD_LENGTH, pad_width_m=PERM_PAD_WIDTH,
                post_plates=PERM_POST_PLATES, plate_model=PERM_PLATE_MODEL, plate_size_m=PERM_PLATE_SIZE,
                plate_thickness_m=PERM_PLATE_THICKNESS, plate_E_Pa=PERM_PLATE_E, plate_nu=PERM_PLATE_NU,
                plate_density_kg_m3=PERM_PLATE_DENSITY,
                layer_ties=PERM_TIE_OVERHANG, tie_max_plan_m=PERM_TIE_MAX_PLAN, tie_max_rise_m=PERM_TIE_MAX_RISE,
                local_zone_m=PERM_LOCAL_ZONE_R, girder_checks=PERM_GIRDER_CHECKS)


def _perm_options_summary(scn) -> str:
    """One line for the panel: which detail options are on."""
    on = [lab for key, lab in PERM_OPTION_SWITCHES if getattr(scn, key, False)]
    if float(getattr(scn, "perm_opt_local_zone", 0.0) or 0.0) > 0.0:
        on.append("local zones")
    return ", ".join(on) if on else "all off (plain shell model)"


def _perm_is_bridge_preset(scn) -> bool:
    for k, v in PERM_BRIDGE_PRESET.items():
        cur = getattr(scn, k, None)
        if isinstance(v, (bool, str)):
            if cur != v:
                return False
        elif cur is None or abs(_perm_r6(cur) - v) > 1e-9 * max(1.0, abs(v)):
            return False
    return True


def _perm_label_wrapped(layout, text, icon='NONE', width=33) -> None:
    """Label broken into lines that fit the sidebar at its default width (Blender clips long labels)."""
    import textwrap
    col = layout.column(align=True)
    for i, ln in enumerate(textwrap.wrap(str(text), width=width) or [""]):
        if i == 0 and icon != 'NONE':
            col.label(text=ln, icon=icon)
        else:
            col.label(text=ln)


def _perm_unit_note(scn) -> str:
    """Text for the panel and the console when the scene's unit settings differ from 'metres, scale 1'."""
    us = getattr(scn, "unit_settings", None)
    if us is None:
        return ""
    sl = float(getattr(us, "scale_length", 1.0) or 1.0)
    if abs(sl - 1.0) > 1e-9:
        return (f"Scene Unit Scale is {sl:g}. PERM reads one Blender unit as 1 m: lengths typed in the panel and "
                f"the mesh are {1.0 / sl:g} times larger for PERM than Blender displays. Set Unit Scale to 1 and "
                "scale the object instead.")
    if str(getattr(us, "system", "METRIC")) == 'IMPERIAL':
        return ("Scene units are Imperial. PERM works in SI (m, N, Pa, kg): one Blender unit is 1 m; forces, "
                "moduli and densities are typed in N, Pa and kg/m³.")
    return ""


def _perm_draw_unit_notice(layout, scn) -> None:
    txt = _perm_unit_note(scn)
    if txt:
        box = layout.box()
        box.alert = True
        _perm_label_wrapped(box, txt, icon='ERROR')


def _perm_draw_scipy_notice(layout) -> None:
    """Top of the panel while SciPy is missing."""
    if HAS_SCIPY:
        return
    box = layout.box()
    box.alert = True
    box.label(text="SciPy is not installed", icon='ERROR')
    _perm_label_wrapped(box, "The solver needs SciPy and Blender does not ship it. The button downloads it "
                             "(about 40 MB, needs internet access).")
    box.operator("perm.install_scipy", icon='IMPORT')


class PERM_OT_InstallSciPy(bpy.types.Operator):
    """Download SciPy with pip into Blender's user scripts folder (about 40 MB; needs internet access; Blender waits until it is done, usually under a minute). NumPy is pinned to the version Blender ships, which is not replaced. The pip command is printed in the system console"""
    bl_idname = "perm.install_scipy"
    bl_label = "Install SciPy"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import subprocess, shutil, glob
        import sys as _sys
        if _perm_try_import_scipy():
            self.report({'INFO'}, "PERM: SciPy is already available")
            return {'FINISHED'}
        try:
            target = _perm_scipy_target()
        except Exception as e:
            self.report({'ERROR'}, f"PERM: no folder to install into ({e})")
            return {'CANCELLED'}
        cmd = _perm_scipy_command(target)
        print("[PERM] SciPy install command:\n   " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not getattr(bpy.app, "online_access", True):
            self.report({'ERROR'}, "PERM: Blender's online access is off. Allow it (Preferences > System > Network) "
                                   "and click again, or run the pip command printed in the system console")
            return {'CANCELLED'}
        had_numpy = bool(glob.glob(os.path.join(target, "numpy")))
        env = dict(os.environ); env["PYTHONNOUSERSITE"] = "1"
        try:
            if subprocess.run([_sys.executable, "-m", "pip", "--version"], capture_output=True, env=env).returncode != 0:
                subprocess.run([_sys.executable, "-m", "ensurepip", "--default-pip"], capture_output=True, env=env)
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        except Exception as e:
            self.report({'ERROR'}, f"PERM: pip could not be started ({e.__class__.__name__}: {e})")
            return {'CANCELLED'}
        print((r.stdout or "")[-4000:]); print((r.stderr or "")[-4000:])
        if r.returncode != 0:
            last = [ln for ln in (r.stderr or r.stdout or "").strip().splitlines() if ln.strip()][-1:] or ["no message"]
            self.report({'ERROR'}, f"PERM: pip failed - {last[0][:200]} (full output in the system console)")
            return {'CANCELLED'}
        if not had_numpy:
            # pip put a second copy of Blender's own NumPy version next to SciPy: remove it, Blender's is used
            for pat in ("numpy", "numpy.libs", "numpy-*.dist-info"):
                for d in glob.glob(os.path.join(target, pat)):
                    shutil.rmtree(d, ignore_errors=True)
        if target not in _sys.path:
            _sys.path.append(target)
        if not _perm_try_import_scipy():
            self.report({'ERROR'}, f"PERM: SciPy was installed in {target} but does not load ({PERM_SCIPY_ERROR}). "
                                   "Restart Blender; if it still fails, see the README")
            return {'CANCELLED'}
        import scipy as _scipy
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                area.tag_redraw()
        self.report({'INFO'}, f"PERM: SciPy {_scipy.__version__} installed in {target}")
        return {'FINISHED'}


class PERM_OT_BridgePreset(bpy.types.Operator):
    """Switch on the connection details of the girder-bridge study with the paper's sizes: point loads on 0.25 m patches, supports on 0.20 x 0.10 m bearing pads, steel posts on 0.15 m x 15 mm base plates, offset layers tied through the thickness, 0.25 m local zones and the girder checks"""
    bl_idname = "perm.bridge_preset"
    bl_label = "Bridge preset (paper settings)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        for k, v in PERM_BRIDGE_PRESET.items():
            setattr(scn, k, v)
        scn.perm_opt_show = True
        self.report({'INFO'}, "PERM: bridge preset set (connection details on, paper sizes)")
        return {'FINISHED'}


class PERM_OT_ResetDetails(bpy.types.Operator):
    """Switch every connection detail off: loads on their nodes, supports at their nodes, no base plates, no layer ties, no local zones, no girder checks (the plain shell model). The sizes typed in the panel are kept"""
    bl_idname = "perm.reset_details"
    bl_label = "All details off"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        for k, _lab in PERM_OPTION_SWITCHES:
            setattr(scn, k, False)
        scn.perm_opt_local_zone = 0.0
        self.report({'INFO'}, "PERM: all connection details off")
        return {'FINISHED'}


def _perm_draw_detail_options(layout, scn) -> None:
    """Panel section '8b. Connection details': the connection details as options."""
    box = layout.box()
    row = box.row(align=True)
    show = bool(getattr(scn, "perm_opt_show", False))
    row.prop(scn, "perm_opt_show", text="", icon='TRIA_DOWN' if show else 'TRIA_RIGHT', emboss=False)
    row.label(text="8b. Connection details", icon='MODIFIER')
    _perm_label_wrapped(box, ("Now: " + _perm_options_summary(scn))
                        + (" = bridge preset" if _perm_is_bridge_preset(scn) else ""), icon='INFO')
    if not show:
        return
    col = box.column(align=True)
    col.operator("perm.bridge_preset", icon='PRESET')
    col.operator("perm.reset_details", icon='LOOP_BACK')
    col = box.column(align=True)
    col.prop(scn, "perm_opt_patch_loads")
    if scn.perm_opt_patch_loads:
        col.prop(scn, "perm_opt_patch_length"); col.prop(scn, "perm_opt_patch_width")
    col = box.column(align=True)
    col.prop(scn, "perm_opt_bearing_pads")
    if scn.perm_opt_bearing_pads:
        col.prop(scn, "perm_opt_pad_length"); col.prop(scn, "perm_opt_pad_width")
    col = box.column(align=True)
    col.prop(scn, "perm_opt_post_plates")
    if scn.perm_opt_post_plates:
        col.prop(scn, "perm_opt_plate_model", text="")
        col.prop(scn, "perm_opt_plate_size"); col.prop(scn, "perm_opt_plate_thickness")
        if scn.perm_opt_plate_model == 'SHELL':
            col.prop(scn, "perm_opt_plate_E")
            col.prop(scn, "perm_opt_plate_nu"); col.prop(scn, "perm_opt_plate_density")
    col = box.column(align=True)
    col.prop(scn, "perm_opt_tie_layers")
    if scn.perm_opt_tie_layers:
        col.prop(scn, "perm_opt_tie_max_plan"); col.prop(scn, "perm_opt_tie_max_rise")
    col = box.column(align=True)
    col.prop(scn, "perm_opt_local_zone")
    col.prop(scn, "perm_opt_girder_checks")
    _perm_label_wrapped(box, "Span axis = the line between the two farthest support groups", icon='EMPTY_AXIS')


def _perm_register_detail_options() -> None:
    S = bpy.types.Scene
    FP, BP = bpy.props.FloatProperty, bpy.props.BoolProperty
    S.perm_opt_show = BP(name="Show the connection details", default=False)
    S.perm_opt_patch_loads = BP(
        name="Point loads on a patch", default=False,
        description="Spread each point load (a load group of 1 to 3 nodes) over the faces around it instead of "
                    "putting it on its node. Loaded: faces whose centre lies in the patch, within 3 cm in height "
                    "of the load, less than 45 degrees from horizontal, of a material with E < 100 GPa and mass. "
                    "A load with no such face stays on its node")
    S.perm_opt_patch_length = FP(name="Length", default=0.25, min=0.001, soft_max=2.0, precision=3, unit='LENGTH',
                                 description="Patch size along the span axis (the line between the two farthest support groups)")
    S.perm_opt_patch_width = FP(name="Width", default=0.25, min=0.001, soft_max=2.0, precision=3, unit='LENGTH',
                                description="Patch size across the span axis")
    S.perm_opt_bearing_pads = BP(
        name="Supports on bearing pads", default=False,
        description="Every node of a PINNED or ROLLER support group (8 nodes or fewer, two support groups or more) "
                    "becomes a pad: the nodes at its height (within 8 mm) from the node towards the other support. "
                    "The pad is held as a whole with uniform pressure, so it stays free to rotate")
    S.perm_opt_pad_length = FP(name="Length", default=0.20, min=0.001, soft_max=2.0, precision=3, unit='LENGTH',
                               description="Pad length along the span axis, from the support node towards the other support")
    S.perm_opt_pad_width = FP(name="Width", default=0.10, min=0.001, soft_max=2.0, precision=3, unit='LENGTH',
                              description="Pad width across the span axis (nodes within half of it, plus 1 cm, of the support node)")
    S.perm_opt_post_plates = BP(
        name="Posts on base plates", default=False,
        description="Where faces of a stiff material (E >= 100 GPa: a steel post) meet faces of a softer one "
                    "(a concrete slab), the post stands on a base plate instead of on its few shared nodes")
    S.perm_opt_plate_model = bpy.props.EnumProperty(
        name="Plate", default='SHELL',
        items=[('SHELL', "Steel plate (shell elements)",
                "The hole in the slab under the post is closed, a plate of shell elements is bonded on the slab "
                "and the tube walls are carried down to it"),
               ('RIGID', "Rigid plate", "The slab nodes under the plate are tied rigidly to the post")])
    S.perm_opt_plate_size = FP(name="Size", default=0.15, min=0.01, soft_max=1.0, precision=3, unit='LENGTH',
                               description="Side of the square base plate, centred on the post")
    S.perm_opt_plate_thickness = FP(name="Thickness", default=0.015, min=0.0005, soft_max=0.1, precision=4, unit='LENGTH',
                                    description="Thickness of the base plate")
    S.perm_opt_plate_E = FP(name="E (Pa)", default=200.0e9, min=1.0e6, description="Young's modulus of the base plate")
    S.perm_opt_plate_nu = FP(name="ν", default=0.3, min=0.0, max=0.499, description="Poisson's ratio of the base plate")
    S.perm_opt_plate_density = FP(name="ρ (kg/m³)", default=7850.0, min=0.0, description="Density of the base plate")
    S.perm_opt_tie_layers = BP(
        name="Tie offset layers together", default=False,
        description="A slab modelled by two scanned surfaces (top and underside, each with a shell offset): every "
                    "node of the lower layer is tied rigidly to the nearest node of the upper layer above it")
    S.perm_opt_tie_max_plan = FP(name="Max plan distance", default=0.03, min=0.0, soft_max=0.5, precision=3, unit='LENGTH',
                                 description="Largest horizontal distance between the two nodes of a tie")
    S.perm_opt_tie_max_rise = FP(name="Max rise", default=0.25, min=0.0, soft_max=2.0, precision=3, unit='LENGTH',
                                 description="Largest vertical distance between the two nodes of a tie")
    S.perm_opt_local_zone = FP(
        name="Local zone radius", default=0.0, min=0.0, soft_max=2.0, precision=3, unit='LENGTH',
        description="Results report: a further line of stress extremes that leaves out the elements closer than "
                    "this to a support node, a point-load node or a stiff-to-soft material joint, where the "
                    "stresses are local artefacts of the idealised connection. 0 = no such line")
    S.perm_opt_girder_checks = BP(
        name="Girder checks and span report", default=False,
        description="For a girder on two supports: advice on the supports in the console (clamping, thrust), and in "
                    "the report the positions along the span, the deflection per girder line, the stress extremes "
                    "within 0.1-0.9 L and the midspan section (moment, neutral axis, effective I)")


def _perm_unregister_detail_options() -> None:
    for k in ["perm_opt_show", "perm_opt_local_zone"] + list(PERM_BRIDGE_PRESET):
        if hasattr(bpy.types.Scene, k):
            try:
                delattr(bpy.types.Scene, k)
            except Exception:
                pass


def _perm_principal_rainbow() -> bool:
    try:
        return getattr(bpy.context.scene, 'perm_principal_colormap', 'RAINBOW') == 'RAINBOW'
    except Exception:
        return True


def perm_cmap_jet(t: float):
    """Rainbow (RGB): blue -> cyan -> green -> yellow -> red, t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:   s = t / 0.25;          return (0.0, s, 1.0, 1.0)
    elif t < 0.5:  s = (t - 0.25) / 0.25; return (0.0, 1.0, 1.0 - s, 1.0)
    elif t < 0.75: s = (t - 0.5) / 0.25;  return (s, 1.0, 0.0, 1.0)
    s = (t - 0.75) / 0.25;                return (1.0, 1.0 - s, 0.0, 1.0)


def perm_cmap_signed(t: float):
    """Blue -> white -> red, t in [0, 1] (0.5 = zero)."""
    u = 2.0 * max(0.0, min(1.0, t)) - 1.0
    if u < 0.0:
        s = 1.0 + u; return (s, s, 1.0, 1.0)
    s = 1.0 - u;     return (1.0, s, s, 1.0)


def _perm_store_legend(obj, scale, title, unit, factor, note=""):
    """Keep the colour scale of a result mesh on the object, for the viewport legend.
    clip < 0 means a manual colour range."""
    if obj is None or not scale:
        return
    cmap, vmin, vmax, rmin, rmax, clip = scale
    obj["perm_legend"] = json.dumps(dict(title=str(title), unit=unit, cmap=cmap,
                                         vmin=vmin * factor, vmax=vmax * factor,
                                         rmin=rmin * factor, rmax=rmax * factor, clip=clip,
                                         note=str(note or "")))


def _perm_legend_color(info, t):
    """Colour of the legend bar at fraction t (0 = bottom, 1 = top)."""
    vmin, vmax = float(info['vmin']), float(info['vmax'])
    if info.get('cmap') != 'signed':
        return perm_cmap_jet(t)
    v = vmin + (vmax - vmin) * t
    neg = abs(vmin) if vmin < 0 else max(abs(vmax), 1e-30)
    pos = vmax if vmax > 0 else max(abs(vmin), 1e-30)
    u = max(-1.0, min(1.0, v / neg if v < 0 else v / pos))
    return perm_cmap_signed(0.5 * (u + 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR FILTER (all / concrete / steel) AND COLOUR RANGE
# ─────────────────────────────────────────────────────────────────────────────
# Every stress / moment result mesh keeps its face values ("perm_face_value"),
# so it can be repainted at any time:
#   filter  ALL       every face (an overview: steel and concrete share one scale)
#           CONCRETE  faces whose material has E < 100 GPa and a density > 0
#           STEEL     faces whose material has E >= 100 GPa
#   Massless faces (deck edges, web connectors: modelling devices, their stress
#   is not physical) are shown only with ALL. Faces outside the filter are grey
#   and do not take part in the colour scale, and the node values averaged for
#   the colours (and the hover inspector) use only the faces that are shown.
#   range   AUTO      from the shown faces, the top/bottom few % saturated
#           MANUAL    min / max typed in the panel, kept per map AND per filter
_PERM_FILTER_CLASS = {'CONCRETE': 1, 'STEEL': 2}
_PERM_FILTER_NOTE = {'CONCRETE': "concrete only", 'STEEL': "steel only"}
_PERM_GREY = (0.5, 0.5, 0.5, 1.0)


def _perm_material_class(E, rho) -> int:
    """1 concrete, 2 steel (E >= 100 GPa), 3 massless link face (density 0), 0 unknown."""
    if E is None:
        return 0
    if E >= 1.0e11:
        return 2
    if (rho or 0.0) <= 0.0:
        return 3
    return 1


def _perm_material_classes(scn) -> Dict[str, int]:
    return {m.name.strip(): _perm_material_class(float(m.youngs_modulus), float(m.density))
            for m in scn.perm_materials}


def _perm_face_classes(obj, scn) -> np.ndarray:
    """Per-face class (see _perm_material_class) from the face's PERM material."""
    me = obj.data
    out = np.zeros(len(me.polygons), dtype=np.int8)
    tags = _read_face_tags(me, 'material')
    names = _face_tag_names(obj, 'material')
    if tags is None or not names or not len(tags):
        return out
    mc = _perm_material_classes(scn)
    lut = np.zeros(max(max(names), int(tags.max())) + 1, dtype=np.int8)
    for tid, nm in names.items():
        if 0 < int(tid) < len(lut):
            lut[int(tid)] = mc.get(str(nm).strip(), 0)
    ok = tags > 0
    out[ok] = lut[tags[ok]]
    return out


def _perm_stored_range(scn, component, flt):
    d = scn.get("perm_ranges")
    v = d.get(f"{component}|{flt}") if d is not None else None
    return (float(v[0]), float(v[1])) if v is not None and len(v) == 2 else None


def _perm_jet_np(t):
    t = np.clip(t, 0.0, 1.0)
    return np.stack([np.clip(4.0 * t - 2.0, 0.0, 1.0),
                     np.clip(np.minimum(4.0 * t, 4.0 - 4.0 * t), 0.0, 1.0),
                     np.clip(2.0 - 4.0 * t, 0.0, 1.0)], axis=1)


def _perm_recolor_result(obj, scn):
    """(Re)paint one stress / moment result mesh from its stored face values with the
    scene's filter and colour range. Returns (node values, node has a value) as used
    for the colours, or None for a result mesh without stored face values."""
    me = obj.data
    fv = _perm_face_attr_array(me, "perm_face_value", np.float32)
    if fv is None:
        return None
    fv = fv.astype(np.float64)
    nf, nv, nl = len(me.polygons), len(me.vertices), len(me.loops)
    comp = str(obj.get("perm_component", ""))
    flt = getattr(scn, "perm_viz_filter", 'ALL')
    show = np.isfinite(fv)
    links = np.zeros(nf, dtype=bool)
    if flt in _PERM_FILTER_CLASS:
        _cls = _perm_face_classes(obj, scn)
        if flt == 'CONCRETE' and getattr(scn, "perm_viz_links", False):
            links = show & (_cls == 3)      # painted on the concrete scale, kept out of it
        show &= _cls == _PERM_FILTER_CLASS[flt]
    ls = np.zeros(nf, dtype=np.int64); lt = np.zeros(nf, dtype=np.int64)
    me.polygons.foreach_get("loop_start", ls); me.polygons.foreach_get("loop_total", lt)
    lv = np.zeros(nl, dtype=np.int64); me.loops.foreach_get("vertex_index", lv)
    lf = np.zeros(nl, dtype=np.int64)
    n_l = int(lt.sum())
    lf[np.repeat(ls, lt) + np.arange(n_l) - np.repeat(np.cumsum(lt) - lt, lt)] = np.repeat(np.arange(nf), lt)
    sel = show[lf]
    vcnt = np.bincount(lv[sel], minlength=nv).astype(np.float64)
    vsum = np.bincount(lv[sel], weights=fv[lf[sel]], minlength=nv)
    vok = vcnt > 0
    vval = np.zeros(nv)
    vval[vok] = vsum[vok] / vcnt[vok]
    # massless link faces: their own node averages, so they never change the concrete colours
    lsel = links[lf]
    lcnt = np.bincount(lv[lsel], minlength=nv).astype(np.float64)
    lsum = np.bincount(lv[lsel], weights=fv[lf[lsel]], minlength=nv)
    lok = lcnt > 0
    lval = np.zeros(nv)
    lval[lok] = lsum[lok] / lcnt[lok]

    factor = float(obj.get("perm_factor", 1e-6))
    signed = bool(obj.get("perm_signed", False))
    if signed and comp.startswith('sigma') and _perm_principal_rainbow():
        signed = False
    sv = vval[vok]
    rmin, rmax = (float(sv.min()), float(sv.max())) if len(sv) else (0.0, 0.0)
    rng = (_perm_stored_range(scn, comp, flt)
           if getattr(scn, "perm_color_range", 'AUTO') == 'MANUAL' else None)
    clip = float(getattr(scn, "perm_clip_percent", 2.0))
    if rng is not None:
        vmin, vmax = rng[0] / factor, rng[1] / factor
        if vmax <= vmin:
            vmax = vmin + max(abs(vmin), 1.0 / factor) * 1e-6
        clip_tag = -1.0
    else:
        clip_tag = clip
        if len(sv):
            lo = float(np.percentile(sv, clip)) if clip > 0 else rmin
            hi = float(np.percentile(sv, 100.0 - clip)) if clip > 0 else rmax
        else:
            lo, hi = 0.0, 1.0
        if signed:
            lim = max(abs(lo), abs(hi), 1e-30)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = lo, (hi if hi > lo else lo + 1e-30)
    val_l = np.where(lsel, lval[lv], vval[lv])
    if signed:
        neg = abs(vmin) if vmin < 0 else max(abs(vmax), 1e-30)
        pos = vmax if vmax > 0 else max(abs(vmin), 1e-30)
        u = np.clip(np.where(val_l < 0.0, val_l / neg, val_l / pos), -1.0, 1.0)
        a = 1.0 - np.abs(u)
        rgb = np.where((u < 0.0)[:, None], np.stack([a, a, np.ones_like(a)], axis=1),
                       np.stack([np.ones_like(a), a, a], axis=1))
    else:
        rgb = _perm_jet_np((val_l - vmin) / (vmax - vmin))
    cols = np.ones((nl, 4), dtype=np.float32)
    cols[:, :3] = rgb
    cols[~(sel | lsel)] = _PERM_GREY
    layer = str(obj.get("perm_result_layer", "STRESS"))
    ca = me.color_attributes.get(layer)
    if ca is None or ca.domain != 'CORNER':
        if ca is not None:
            me.color_attributes.remove(ca)
        ca = me.color_attributes.new(name=layer, type='BYTE_COLOR', domain='CORNER')
    me.color_attributes.active_color = ca
    ca.data.foreach_set("color", cols.ravel())
    hv = np.where(vok, vval, lval)
    for name, data in (("perm_field", hv.astype(np.float32)), ("perm_field_valid", vok | lok)):
        at = me.attributes.get(name)            # what the hover inspector shows
        if at is not None and at.domain == 'POINT' and len(at.data) == nv:
            at.data.foreach_set("value", data)
    _perm_store_legend(obj, ('signed' if signed else 'jet', vmin, vmax, rmin, rmax, clip_tag),
                       obj.get("perm_title", comp), obj.get("perm_unit", "MPa"), factor,
                       note=_PERM_FILTER_NOTE.get(flt, "") + (" + massless links" if links.any() else ""))
    me.update()
    unit = obj.get("perm_unit", "MPa")
    how = "manual range" if clip_tag < 0 else (f"{clip:g}-{100 - clip:g}% range" if clip else "min-max")
    print(f"[PERM Colours] '{obj.name}' {_PERM_FILTER_NOTE.get(flt, 'all faces')}: "
          f"{int(show.sum())} faces" + (f" + {int(links.sum())} link faces (not in the range)" if links.any() else "") +
          f"; values {rmin * factor:+.3f} .. {rmax * factor:+.3f} {unit}; "
          f"colours {vmin * factor:+.3f} .. {vmax * factor:+.3f} {unit} ({how})")
    return hv, vok | lok


def _perm_recolor_all(context) -> Tuple[int, int]:
    """Repaint every stress / moment result mesh in the scene. Returns (repainted, too old)."""
    n = old = 0
    for o in list(context.scene.objects):
        if o.type != 'MESH' or not o.get("perm_result_viz") or o.mode == 'EDIT':
            continue
        try:
            if _perm_recolor_result(o, context.scene) is None:
                old += 1
            else:
                n += 1
        except Exception as e:
            print(f"[PERM Colours] '{o.name}' not repainted: {e}")
    for a in (context.screen.areas if context.screen is not None else []):
        if a.type == 'VIEW_3D':
            a.tag_redraw()
    return n, old


_PERM_NO_REPAINT = [False]      # set while a range field switches the mode itself


def _perm_set_range_mode(scn, mode) -> None:
    _PERM_NO_REPAINT[0] = True
    try:
        if scn.perm_color_range != mode:
            scn.perm_color_range = mode
    finally:
        _PERM_NO_REPAINT[0] = False


def _perm_update_colors(self, context):
    if _PERM_NO_REPAINT[0]:
        return
    try:
        _perm_recolor_all(context)
    except Exception as e:
        print(f"[PERM Colours] repaint failed: {e}")


def _perm_range_target(context):
    """The result map the colour range fields refer to: the active one, or the one the legend shows."""
    objs = [o for o in context.scene.objects
            if o.type == 'MESH' and o.get("perm_result_viz") and o.get("perm_component")
            and o.visible_get()]
    if not objs:
        return None
    a = context.active_object
    return a if a in objs else objs[-1]


def _perm_range_get(idx):
    def _get(self):
        try:
            o = _perm_range_target(bpy.context)
            if o is None:
                return 0.0
            r = _perm_stored_range(self, o.get("perm_component", ""), self.perm_viz_filter)
            if r is not None and self.perm_color_range == 'MANUAL':
                return r[idx]
            info = json.loads(o.get("perm_legend", "{}"))
            return float(info.get('vmin' if idx == 0 else 'vmax', 0.0))
        except Exception:
            return 0.0
    return _get


def _perm_range_set(idx):
    def _set(self, value):
        o = _perm_range_target(bpy.context)
        if o is None:
            return
        comp, flt = o.get("perm_component", ""), self.perm_viz_filter
        cur = [_perm_range_get(0)(self), _perm_range_get(1)(self)]
        cur[idx] = float(value)
        rp = self.get("perm_ranges")
        d = rp.to_dict() if rp is not None else {}
        d[f"{comp}|{flt}"] = cur
        self["perm_ranges"] = d
        _perm_set_range_mode(self, 'MANUAL')     # without repainting every map
        _perm_recolor_result(o, self)
    return _set


class PERM_OT_RecolorResults(bpy.types.Operator):
    """Repaint all result maps with the current filter and colour range"""
    bl_idname = "perm.recolor_results"
    bl_label = "PERM: Repaint Result Maps"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if context.active_object is not None and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        n, old = _perm_recolor_all(context)
        if old:
            self.report({'WARNING'}, f"PERM: {n} map(s) repainted; {old} map(s) are from an older "
                                     f"version - run the analysis once more to filter them")
        else:
            self.report({'INFO'}, f"PERM: {n} result map(s) repainted")
        return {'FINISHED'}


class PERM_OT_AutoColorRange(bpy.types.Operator):
    """Forget the typed colour range of this map (for the current filter) and go back to automatic"""
    bl_idname = "perm.auto_color_range"
    bl_label = "PERM: Automatic Colour Range"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scn = context.scene
        o = _perm_range_target(context)
        rp = scn.get("perm_ranges")
        if o is not None and rp is not None:
            d = rp.to_dict()
            d.pop(f"{o.get('perm_component', '')}|{scn.perm_viz_filter}", None)
            scn["perm_ranges"] = d
        if not any(k.endswith("|" + scn.perm_viz_filter) for k in (scn.get("perm_ranges") or {}).keys()):
            _perm_set_range_mode(scn, 'AUTO')
        if o is not None:
            _perm_recolor_result(o, scn)
        return {'FINISHED'}


def _perm_fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 100:   return f"{v:.0f}"
    if a >= 10:    return f"{v:.1f}"
    if a >= 1:     return f"{v:.2f}"
    if a >= 0.01:  return f"{v:.3f}"
    if a == 0:     return "0"
    return f"{v:.2e}"


def _perm_draw_color_legend(info, width, height, ui=1.0, n_more=0):
    """Vertical colour bar with values, bottom-right of the 3D view (POST_PIXEL)."""
    import gpu, blf
    sh = _perm_uniform_shader()
    if sh is None:
        return
    fid = 0
    vmin, vmax = float(info['vmin']), float(info['vmax'])
    bar_w = 18 * ui
    bar_h = max(80 * ui, min(250 * ui, height * 0.55))
    x1 = width - 22 * ui; x0 = x1 - bar_w
    y0 = 46 * ui; y1 = y0 + bar_h
    ticks = [vmin + (vmax - vmin) * k / 5.0 for k in range(6)]
    labels = [_perm_fmt_num(v) for v in ticks]
    _perm_blf_size(fid, 12 * ui)
    lw = max(blf.dimensions(fid, s)[0] for s in labels)
    title = f"{info.get('title', '')}  [{info.get('unit', '')}]"
    if info.get('note'):
        title += f"  - {info['note']}"
    sub1 = f"raw min {_perm_fmt_num(float(info['rmin']))} / max {_perm_fmt_num(float(info['rmax']))}"
    _clip = float(info.get('clip', 0) or 0)
    sub2 = ("manual colour range" if _clip < 0 else
            f"colours span the {_clip:g}-{100 - _clip:g}% range" if _clip else "colours span min-max")
    _perm_blf_size(fid, 11 * ui)
    tw = max(blf.dimensions(fid, s)[0] for s in (title, sub1, sub2))
    px0 = min(x0 - 10 * ui - lw - 8 * ui, x1 - tw - 8 * ui)
    gpu.state.blend_set('ALPHA')
    _perm_draw_rect(sh, px0 - 8 * ui, y0 - 40 * ui, width - 6 * ui, y1 + 30 * ui, (0.0, 0.0, 0.0, 0.55))
    n = 64
    for k in range(n):
        ya = y0 + bar_h * k / n; yb = y0 + bar_h * (k + 1) / n
        _perm_draw_rect(sh, x0, ya, x1, yb + 0.5, _perm_legend_color(info, (k + 0.5) / n))
    for k in range(6):
        yt = y0 + bar_h * k / 5.0
        _perm_draw_rect(sh, x0 - 5 * ui, yt - 0.5 * ui, x0, yt + 0.5 * ui, (1, 1, 1, 0.9))
    z0 = None
    if vmin < 0.0 < vmax:                                       # where tension turns into compression
        z0 = y0 + bar_h * (0.0 - vmin) / (vmax - vmin)
        _perm_draw_rect(sh, x0 - 3 * ui, z0 - 1.0 * ui, x1 + 3 * ui, z0 + 1.0 * ui, (1, 1, 1, 1))
    _perm_blf_size(fid, 12 * ui)
    blf.color(fid, 1, 1, 1, 1)
    for k, s in enumerate(labels):
        yt = y0 + bar_h * k / 5.0
        w = blf.dimensions(fid, s)[0]
        blf.position(fid, x0 - 8 * ui - w, yt - 4 * ui, 0); blf.draw(fid, s)
    if z0 is not None:
        blf.position(fid, x1 + 5 * ui, z0 - 4 * ui, 0); blf.draw(fid, "0")
    _perm_blf_size(fid, 11 * ui)
    blf.color(fid, 1.0, 0.85, 0.35, 1)
    blf.position(fid, px0, y1 + 12 * ui, 0); blf.draw(fid, title)
    blf.color(fid, 0.8, 0.8, 0.8, 1)
    blf.position(fid, px0, y0 - 18 * ui, 0); blf.draw(fid, sub1)
    blf.position(fid, px0, y0 - 32 * ui, 0); blf.draw(fid, sub2)
    gpu.state.blend_set('NONE')


def _perm_legend_target(context):
    objs = [o for o in context.scene.objects if o.get("perm_legend") and o.visible_get()]
    if not objs:
        return None
    a = context.active_object
    o = a if a in objs else objs[-1]
    return json.loads(o["perm_legend"]), len(objs) - 1


def _perm_color_legend_draw():
    """Persistent POST_PIXEL handler: colour legend of the visible result mesh."""
    try:
        context = bpy.context
        if not getattr(context.scene, "perm_show_color_legend", True):
            return
        region = context.region
        if region is None:
            return
        got = _perm_legend_target(context)
        if got is None:
            return
        ui = context.preferences.system.ui_scale or 1.0
        _perm_draw_color_legend(got[0], region.width, region.height, ui, got[1])
    except Exception:
        pass


# ---------------------------------------------------------------- results report
def _perm_results_dir(scene) -> str:
    d = (getattr(scene, "perm_results_dir", "") or "//PERM_results/").strip()
    if d.startswith("//") and not bpy.data.filepath:
        d = os.path.join(os.path.expanduser("~"), "PERM_results")
    else:
        d = bpy.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _perm_poly_cut(Q, p0, ax):
    """Segment where the plane through p0 (normal ax) cuts the polygon Q (n x 3), or None."""
    d = (Q - p0) @ ax
    pts = []
    for k in range(len(Q)):
        a, b = d[k], d[(k + 1) % len(Q)]
        if (a < 0.0) != (b < 0.0):
            u = a / (a - b)
            pts.append(Q[k] + u * (Q[(k + 1) % len(Q)] - Q[k]))
    if len(pts) != 2:
        return None
    return pts[0], pts[1]


def _perm_write_results_report(solver, scene, src_obj, support_groups):
    """One text report per analysis with everything the hand calculations are compared
    with (V0-V4), one row per analysis in PERM_results_log.csv, and the report in the
    Text Editor as 'PERM_Results.txt'. Returns the report path."""
    import datetime, io, contextlib
    conv = solver.converter
    res = solver.results
    P = np.asarray(solver.node_coords, float)
    U = np.asarray(res.displacements, float)
    Rf = np.asarray(res.reaction_forces, float) if res.reaction_forces is not None else np.zeros_like(U)
    nid2i = solver.node_id_to_index
    now = datetime.datetime.now()
    case = (getattr(scene, "perm_case_name", "") or "").strip()
    out = []
    S = {}

    def head(t):
        out.append(""); out.append(t); out.append("-" * len(t))

    # ---- elements
    sec_of = _element_face_tag_names(conv, 'section')
    sec_of.update(getattr(conv, '_plate_sections', None) or {})             # plate, fill and tube-wall elements
    eids, cen, mats, secs, thk, steel, area, Ts, corner = [], [], [], [], [], [], [], [], []
    enodes = []                                                  # node indices of each element
    for eid, e in conv.elements.items():
        if _perm_is_line(e) or getattr(e, 'viz_only', False):
            continue
        idx = [nid2i[n] for n in e.node_ids]
        nc = 3 if getattr(e.element_type, 'name', '').startswith('TRI') else 4
        Q = P[idx[:nc]]
        enodes.append(idx[:nc])
        m = conv.materials.get(e.material_id, conv.materials[0])
        eids.append(eid); cen.append(Q.mean(0)); corner.append(Q)
        mats.append(m.name); steel.append(m.youngs_modulus > 1.0e11)
        secs.append(sec_of.get(eid, "(none)")); thk.append(float(e.thickness))
        area.append(solver._compute_element_area(solver._get_element_coords(e), e.element_type))
        T = getattr(solver, '_element_T_cache', {}).get(eid)
        if T is None:
            _, T = solver._compute_local_system(Q)
        Ts.append(T)
    cen = np.array(cen); thk = np.array(thk); area = np.array(area); steel = np.array(steel)
    mats = np.array(mats, dtype=object)
    rho = np.array([conv.materials.get(conv.elements[i].material_id, conv.materials[0]).density for i in eids])
    mass = rho * thk * area
    elen = np.median([np.linalg.norm(Q[1] - Q[0]) for Q in corner[::max(1, len(corner) // 2000)]]) if corner else 0.05

    # ---- supports and the span axis
    sups = []
    for name, bc in support_groups:
        ids = [nid2i[n] for n in conv.node_sets.get(name, set()) if n in nid2i]
        if not ids:
            continue
        c = P[ids].mean(0)
        _pads = [p_ for p_ in (getattr(solver, '_bearing_pads', None) or []) if p_['group'] == name]
        if _pads:                            # the support acts at the weighted centres of its bearing pads
            c = np.mean([(p_['weights'][:, None] * P[p_['nodes']]).sum(0) for p_ in _pads], axis=0)
        F = Rf[ids, :3].sum(0)
        M = Rf[ids, 3:6].sum(0) + np.cross(P[ids] - c, Rf[ids, :3]).sum(0)
        sups.append(dict(name=name.replace("support_", ""), bc=bc, ids=ids, c=c, F=F, M=M))
    L = 0.0; c1 = c2 = None
    ax = np.array([1.0, 0.0, 0.0])
    GIRDER = bool(PERM_GIRDER_CHECKS)          # span, girder-line and midspan parts of the report
    if len(sups) >= 2 and GIRDER:
        best = (0.0, 0, 1)
        for i in range(len(sups)):
            for j in range(i + 1, len(sups)):
                d = (sups[j]['c'] - sups[i]['c']) * np.array([1.0, 1.0, 0.0])
                if np.linalg.norm(d) > best[0]:
                    best = (float(np.linalg.norm(d)), i, j)
        A_, B_ = sups[best[1]], sups[best[2]]
        d = B_['c'] - A_['c']
        if (d[0] if abs(d[0]) >= abs(d[1]) else d[1]) < 0:
            A_, B_ = B_, A_
        c1, c2 = A_['c'], B_['c']
        ax = (c2 - c1) * np.array([1.0, 1.0, 0.0]); L = float(np.linalg.norm(ax))
        if L > 1e-9:
            ax = ax / L
            sup_first, sup_last = A_['name'], B_['name']
        else:                                  # the support centres lie above each other: no span
            L = 0.0; ax = np.array([1.0, 0.0, 0.0])
    else:
        ext = P.max(0) - P.min(0)
        ax = np.array([1.0, 0.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0, 0.0])
        c1 = P.min(0)
    across = np.cross([0.0, 0.0, 1.0], ax)
    s_node = (P - c1) @ ax
    lat_node = (P - c1) @ across

    # ---- load cases (each one re-assembled on its own, so they can be told apart)
    cases = []
    F_saved = solver._F_global
    for lc in getattr(solver, 'load_cases', []):
        solver._F_global = np.zeros(solver.n_dofs)
        with contextlib.redirect_stdout(io.StringIO()):
            if lc.load_type == LoadType.NODAL_FORCE:
                solver._apply_nodal_force(lc)
            elif lc.load_type == LoadType.GRAVITY:
                solver._apply_gravity(lc)
            elif lc.load_type == LoadType.PRESSURE:
                solver._apply_pressure(lc)
            elif lc.load_type == LoadType.DISTRIBUTED:
                solver._apply_surface_load(lc)
        cases.append((lc, solver._F_global.reshape(-1, solver.DOFS_PER_NODE)[:, :3].copy()))
    solver._F_global = F_saved
    F_all = sum((f for _, f in cases), np.zeros((len(P), 3)))

    # ---- connection zones -- support nodes, point-load nodes, railing-post bases
    sup_set = set(i for sp in sups for i in sp['ids'])
    load_set = set()
    for lc, f in cases:
        if lc.load_type == LoadType.NODAL_FORCE:
            load_set.update(int(i) for i in np.nonzero(np.linalg.norm(f, axis=1) > 0)[0])
    st_nodes, co_nodes = set(), set()
    for k, ids in enumerate(enodes):
        (st_nodes if steel[k] else co_nodes).update(ids)
    joint_set = st_nodes & co_nodes                              # steel meets concrete: post bases
    conn = sorted(sup_set | load_set | joint_set)
    local = np.zeros(len(eids), dtype=bool)
    if conn and PERM_LOCAL_ZONE_R > 0 and len(eids):
        try:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(P[conn]).query(cen, k=1)
        except Exception:
            pts = P[conn]
            dist = np.array([float(np.min(np.linalg.norm(pts - c, axis=1))) for c in cen])
        local = dist <= PERM_LOCAL_ZONE_R

    # ================================================================ header
    out.append(f"PERM RESULTS REPORT ({PERM_VERSION})  -  {now:%Y-%m-%d %H:%M:%S}")
    out.append("=" * 72)
    out.append(f"Case          : {case or '(no case name: set it in 8. Analysis)'}")
    out.append(f"Model         : {src_obj.name}")
    out.append(f"Blender file  : {bpy.data.filepath or '(not saved)'}")
    n_off = sum(1 for e in conv.elements.values() if abs(getattr(e, 'offset', 0.0)) > 0)
    out.append(f"Mesh          : {len(P)} nodes, {len(eids)} shell elements, {solver.n_dofs} DOF, "
               f"{n_off} elements with a shell offset")
    out.append(f"Linear solver : {getattr(solver, '_solve_engine', '?')}, "
               f"{getattr(solver.results, 'solve_time', 0.0):.1f} s")
    out.append("Formulation   : QUAD4 QM6 + MITC4, TRI3 CST + MITC3, Hughes-Brezzi drilling "
               f"(gamma {PERMSolver.DRILL_GAMMA:g}); warped quads linked to their mean plane")
    _det = []
    patched = getattr(solver, '_patched_loads', None) or {}
    if patched:
        _det.append(f"{len(patched)} point load(s) on {PERM_PATCH_SIZE[0]:.2f} x {PERM_PATCH_SIZE[1]:.2f} m patches")
    if PERM_BEARING_PADS and getattr(solver, '_bearing_pads', None):
        _det.append(f"{len(solver._bearing_pads)} bearing pads {PERM_PAD_LENGTH:.2f} m x {PERM_PAD_WIDTH:.2f} m")
    det_info = getattr(solver, '_detail_info', {}) or {}
    if PERM_POST_PLATES and det_info.get('plates'):
        _det.append(f"{det_info['plates']} posts on rigid {PERM_PLATE_SIZE:.2f} x {PERM_PLATE_SIZE:.2f} m base plates")
    plate_info = getattr(conv, '_plate_info', {}) or {}
    if PERM_POST_PLATES and plate_info.get('done'):
        _det.append(f"{plate_info['done']} posts on {PERM_PLATE_SIZE:.2f} x {PERM_PLATE_SIZE:.2f} x {PERM_PLATE_THICKNESS * 1e3:.0f} mm "
                    f"steel base plates (shell, bonded on the deck; deck holes filled; tube walls welded to the plate"
                    + (f"; {plate_info['enlarged']} plates enlarged to the scanned base" if plate_info.get('enlarged') else "")
                    + (f"; {plate_info['failed']} posts left as scanned" if plate_info.get('failed') else "") + ")")
    if PERM_TIE_OVERHANG and det_info.get('ties'):
        _det.append(f"overhang layers tied at {det_info['ties']} nodes")
    out.append("Connections   : " + ("; ".join(_det) if _det else "point connections") + f" ({PERM_VERSION})")
    if not GIRDER:
        out.append("Options       : connection details " + ("as listed above" if _det else "off")
                   + "; girder checks and span report off (8b. Connection details in the panel)")
    out.append("Units         : m, kN, kN·m, mm (displacement), MPa (stress). Tension +. "
               "Deflection: downward = negative uz.")
    S.update(date=f"{now:%Y-%m-%d %H:%M:%S}", case=case, model=src_obj.name)

    # ================================================================ materials, sections, mass
    head("1. MATERIALS AND SECTIONS")
    for mid, m in sorted(conv.materials.items()):
        out.append(f"  material {m.name:<18} E = {m.youngs_modulus/1e9:8.3f} GPa  nu = {m.poissons_ratio:.2f}  "
                   f"rho = {m.density:7.1f} kg/m3")
    for pg in getattr(scene, 'perm_sections', []):
        rc = ""
        if pg.rebar_ratio > 0 and pg.steel_modulus > 0:
            rc = f"  rebar rho = {pg.rebar_ratio:.4f}, cover {pg.rebar_cover:.3f} m"
        out.append(f"  section  {pg.name:<18} t = {pg.thickness:.4f} m{rc}")
    head("2. MASS AND SELF-WEIGHT")
    groups = defaultdict(lambda: [0.0, 0.0, 0])
    for k in range(len(eids)):
        g = groups[(mats[k], secs[k], round(thk[k], 5))]
        g[0] += area[k]; g[1] += mass[k]; g[2] += 1
    for (mn, sn, t), (a, ms, n) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        out.append(f"  {mn:<18} {sn:<16} t={t*1e3:6.1f} mm : {n:7d} faces {a:9.2f} m2 {ms:10.1f} kg")
    tot_m = float(mass.sum())
    deck_len = float(s_node.max() - s_node.min())
    if GIRDER:
        out.append(f"  TOTAL mass {tot_m:,.1f} kg  ->  weight {tot_m*9.81/1e3:.2f} kN   "
                   f"(over the model length {deck_len:.3f} m: w = {tot_m*9.81/1e3/max(deck_len,1e-9):.3f} kN/m)")
    else:
        _bb = P.max(0) - P.min(0)
        out.append(f"  TOTAL mass {tot_m:,.1f} kg  ->  weight {tot_m*9.81/1e3:.2f} kN at 9.81 m/s2   "
                   f"(shell faces only; model size {_bb[0]:.3f} x {_bb[1]:.3f} x {_bb[2]:.3f} m)")
    S.update(mass_kg=round(tot_m, 1), weight_kN=round(tot_m * 9.81 / 1e3, 3))

    # ================================================================ supports
    head("3. SUPPORTS, SPAN AND REACTIONS" if GIRDER else "3. SUPPORTS AND REACTIONS")
    for sp in sups:
        c, F, M = sp['c'], sp['F'], sp['M']
        out.append(f"  {sp['name']:<12} {sp['bc']:<9} {len(sp['ids']):4d} nodes  centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")
        out.append(f"      reaction F = ({F[0]/1e3:+9.3f}, {F[1]/1e3:+9.3f}, {F[2]/1e3:+9.3f}) kN   "
                   f"M about its centre = ({M[0]/1e3:+8.3f}, {M[1]/1e3:+8.3f}, {M[2]/1e3:+8.3f}) kN·m")
        if len(sp['ids']) <= 8:
            # the forces in each support node. A pinned node carries no
            # moment; 'M about its centre' is the moment of these forces about c.
            for i in sp['ids']:
                p, r = P[i], Rf[i, :3]
                out.append(f"        node at ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
                           f"R = ({r[0]/1e3:+9.3f}, {r[1]/1e3:+9.3f}, {r[2]/1e3:+9.3f}) kN")
    Rt = Rf[:, :3].sum(0); Ft = F_all.sum(0)
    out.append(f"  sum of reactions  = ({Rt[0]/1e3:+.3f}, {Rt[1]/1e3:+.3f}, {Rt[2]/1e3:+.3f}) kN")
    out.append(f"  sum of loads      = ({Ft[0]/1e3:+.3f}, {Ft[1]/1e3:+.3f}, {Ft[2]/1e3:+.3f}) kN   "
               f"(equilibrium error {np.linalg.norm(Rt + Ft)/max(np.linalg.norm(Ft),1e-9)*100:.2e} %)")
    if L > 0:
        axn = "X" if abs(ax[0]) > 0.98 else ("Y" if abs(ax[1]) > 0.98 else "bridge axis")
        out.append(f"  span L between support centres = {L:.4f} m along {axn} "
                   f"(from {sup_first} to {sup_last}); midspan at "
                   f"({(c1 + ax*L/2)[0]:+.3f}, {(c1 + ax*L/2)[1]:+.3f})")
    S.update(L_m=round(L, 4))
    for k, sp in enumerate(sups[:2]):
        S[f"R{k+1}_{sp['name']}_Fz_kN"] = round(sp['F'][2] / 1e3, 3)

    # ================================================================ loads
    head("4. LOADS")
    P_point = np.zeros(3)
    load_rows = []
    for lc, f in cases:
        nz = np.nonzero(np.linalg.norm(f, axis=1) > 0)[0]
        if not len(nz):
            continue
        Ftot = f[nz].sum(0); wgt = np.linalg.norm(f[nz], axis=1)
        p = (P[nz] * wgt[:, None]).sum(0) / wgt.sum()
        kind = ("self-weight" if lc.load_type == LoadType.GRAVITY else
                ("pressure" if lc.load_type == LoadType.PRESSURE else
                 ("surface load" if lc.load_type == LoadType.DISTRIBUTED else "point load")))
        name = lc.node_set.replace("load_", "") if lc.node_set else lc.name
        line = (f"  {kind:<11} {name:<20} {len(nz):6d} nodes  F = ({Ftot[0]/1e3:+.3f}, {Ftot[1]/1e3:+.3f}, "
                f"{Ftot[2]/1e3:+.3f}) kN  at ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        out.append(line)
        if L > 0:
            a = float((p - c1) @ ax)
            e = float((p - c1) @ across)
            out.append(f"      a = {a:.3f} m from {sup_first},  b = {L - a:.3f} m,  a/L = {a/L:.3f},  "
                       f"e = {e:+.3f} m across from the line of the support centres")
        if kind == "point load":
            P_point += Ftot
            uz = (U[nz, 2] * wgt).sum() / wgt.sum()
            out.append(f"      deflection under this load: uz = {uz*1e3:+.4f} mm (force-weighted mean of its nodes)")
            load_rows.append((name, Ftot, p, uz))
    if load_rows:
        S["P_total_kN"] = round(float(-P_point[2]) / 1e3, 4)
        S["uz_under_load1_mm"] = round(load_rows[0][3] * 1e3, 4)

    # ================================================================ displacements
    head("5. DISPLACEMENTS")
    um = np.linalg.norm(U[:, :3], axis=1)
    i1, i2, i3 = int(np.argmax(um)), int(np.argmin(U[:, 2])), int(np.argmax(U[:, 2]))
    def at(i):
        return f"at ({P[i,0]:+.3f}, {P[i,1]:+.3f}, {P[i,2]:+.3f})" + (f"  a/L = {s_node[i]/L:.3f}" if L > 0 else "")
    out.append(f"  max |u|          = {um[i1]*1e3:9.4f} mm  {at(i1)}")
    out.append(f"  max downward uz  = {U[i2,2]*1e3:9.4f} mm  {at(i2)}")
    out.append(f"  max upward uz    = {U[i3,2]*1e3:9.4f} mm  {at(i3)}")
    S.update(max_u_mm=round(um[i1] * 1e3, 4), max_down_uz_mm=round(U[i2, 2] * 1e3, 4))
    if L > 0:
        tol = max(0.75 * elen, 0.01)
        out.append("  Deflection along the span, LOWEST nodes of the section (soffit), per girder line")
        out.append(f"  (nodes within +-{tol*1e3:.0f} mm of each station; girders split where the soffit has a gap > 0.3 m):")
        hdr_done = False
        prof = []
        for k in range(11):
            sk = L * k / 10.0
            m = np.abs(s_node - sk) < tol
            if not m.any():
                continue
            zmin = P[m, 2].min()
            so = np.nonzero(m & (P[:, 2] < zmin + 0.03))[0]
            order = so[np.argsort(lat_node[so])]
            cl = np.split(order, np.nonzero(np.diff(lat_node[order]) > 0.3)[0] + 1)
            vals = [(float(lat_node[c].mean()), float(U[c, 2].mean())) for c in cl]
            if not hdr_done:
                out.append("     a/L     s (m)   " + "  ".join(f"girder {g+1} (e={v[0]:+.2f} m)" for g, v in enumerate(vals))
                           + "   all soffit mean")
                hdr_done = True
            out.append(f"    {k/10:5.2f}  {sk:7.3f}   " + "  ".join(f"{v[1]*1e3:+20.4f}" for v in vals)
                       + f"   {U[so, 2].mean()*1e3:+14.4f}")
            prof.append((k, vals, float(U[so, 2].mean())))
        mid = [p for p in prof if p[0] == 5]
        if mid:
            S["mid_soffit_uz_mm"] = round(mid[0][2] * 1e3, 4)
            for g, v in enumerate(mid[0][1][:2]):
                S[f"mid_girder{g+1}_uz_mm"] = round(v[1] * 1e3, 4)
            if len(mid[0][1]) >= 2:
                d = [abs(v[1]) for v in mid[0][1][:2]]
                out.append(f"  midspan share between the two girders: {d[0]/sum(d):.3f} / {d[1]/sum(d):.3f} "
                           f"(0.5/0.5 = equal; compare with the load sharing bounds)")

    # ================================================================ stresses
    if GIRDER:
        head("6. STRESS EXTREMES (MPa) per material  [top/bottom = the element's +z/-z face; 2nd line = 0.1-0.9 L; "
             "3rd line = 0.1-0.9 L and away from connections]")
    else:
        head("6. STRESS EXTREMES (MPa) per material  [top/bottom = the element's +z/-z face"
             + ("; 2nd line = away from connections]" if PERM_LOCAL_ZONE_R > 0 else "]"))
    if PERM_LOCAL_ZONE_R > 0 and GIRDER:
        out.append(f"  connections (their local peaks are artefacts of the idealised joint, not real stresses): "
                   f"{len(sup_set)} support nodes, {len(load_set)} point-load nodes, {len(joint_set)} railing-post base nodes;")
        out.append(f"  the 3rd line leaves out the {int(local.sum())} elements within {PERM_LOCAL_ZONE_R:.2f} m of them "
                   f"(PERM_LOCAL_ZONE_R)")
    elif PERM_LOCAL_ZONE_R > 0:
        out.append(f"  connections (their local peaks are artefacts of the idealised joint, not real stresses): "
                   f"{len(sup_set)} support nodes, {len(load_set)} point-load nodes, {len(joint_set)} nodes where a "
                   f"stiff material (E >= 100 GPa) meets a softer one;")
        out.append(f"  the 2nd line leaves out the {int(local.sum())} elements within {PERM_LOCAL_ZONE_R:.2f} m of them "
                   f"(local zone radius)")
    fld = [("von Mises top", res.von_mises_top, None, 'max'), ("von Mises bottom", res.von_mises_bot, None, 'max'),
           ("sigma1 (max principal) top", res.principal_stresses_top, 0, 'max'),
           ("sigma1 (max principal) bottom", res.principal_stresses_bot, 0, 'max'),
           ("sigma2 (min principal) top", res.principal_stresses_top, 1, 'min'),
           ("sigma2 (min principal) bottom", res.principal_stresses_bot, 1, 'min')]
    for mn in sorted(set(mats.tolist())):
        sel = [k for k in range(len(eids)) if mats[k] == mn]
        out.append(f"  {mn}  ({'steel' if steel[sel[0]] else 'concrete / other'}, {len(sel)} elements)" if GIRDER
                   else f"  {mn}  ({len(sel)} elements)")
        tag = "steel" if steel[sel[0]] else "conc"
        for title, dct, comp, how in fld:
            vals = []
            for k in sel:
                v = (dct or {}).get(eids[k])
                if v is None:
                    continue
                vals.append((float(v if comp is None else v[comp]), k))
            if not vals:
                continue
            key = {"von Mises top": "VM_top", "von Mises bottom": "VM_bot"}.get(title)
            if key is None:
                key = ("s1" if "sigma1" in title else "s2") + ("_top" if title.endswith("top") else "_bot")
            # whole model, then away from the bearings (0.1-0.9 L: no support hot spots)
            span_vals = [(v, k) for v, k in vals if L > 0 and 0.1 <= ((cen[k] - c1) @ ax) / L <= 0.9]
            zones = [("", vals, ""), ("0.1-0.9 L", span_vals, "_span")]
            if PERM_LOCAL_ZONE_R > 0:
                zones.append(("away from connections", [(v, k) for v, k in span_vals if not local[k]], "_away"))
            if not GIRDER:                                 # no span -> whole model, then away from the connections
                zones = [("", vals, "")]
                if PERM_LOCAL_ZONE_R > 0:
                    zones.append(("away from connections", [(v, k) for v, k in vals if not local[k]], "_away"))
            for zone, vv, suffix in zones:
                if not vv or (suffix and L <= 0 and GIRDER):
                    continue
                v, k = (max if how == 'max' else min)(vv)
                c = cen[k]
                lab = title if not suffix else f"{zone:>31}"
                out.append(f"     {lab:<31} {v/1e6:+10.3f}   element {eids[k]:7d} at ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})"
                           + (f"  a/L = {((c - c1) @ ax)/L:.3f}" if L > 0 else ""))
                prev = S.get(f"{key}_{tag}{suffix}_MPa")
                if prev is None or (how == 'max' and v / 1e6 > prev) or (how == 'min' and v / 1e6 < prev):
                    S[f"{key}_{tag}{suffix}_MPa"] = round(v / 1e6, 4)

    # ================================================================ midspan section
    if L > 0:
        head("7. MIDSPAN SECTION (compare with sigma = M (zbar - z) / I)")
        p0 = c1 + ax * (L / 2.0)
        segs = []
        for k in range(len(eids)):
            cut = _perm_poly_cut(corner[k], p0, ax)
            if cut is None:
                continue
            q0, q1 = cut
            l = float(np.linalg.norm(q1 - q0))
            if l < 1e-9:
                continue
            T = Ts[k]; u = T.T @ ax; nn = math.hypot(u[0], u[1])
            if nn < 0.3:
                continue
            ux, uy = u[0] / nn, u[1] / nn
            def saa(s):
                return ux * ux * s[0] + uy * uy * s[1] + 2.0 * ux * uy * s[2]
            eid = eids[k]
            sm = res.stresses.get(eid); st = res.stresses_top.get(eid); sb = res.stresses_bot.get(eid)
            if sm is None:
                continue
            mloc = res.moments.get(eid, np.zeros(3))
            up = T[2, 2]
            e_el = conv.elements[eid]
            m_el = conv.materials.get(e_el.material_id, conv.materials[0])
            fm, fb = solver._rebar_factors(e_el, m_el.youngs_modulus, e_el.thickness)
            # recovered stresses are CONCRETE stresses; the smeared rebar carries the rest
            # (x f_m), and an offset shell carries its force on its own mid-surface
            zc = 0.5 * (q0[2] + q1[2]) + float(getattr(e_el, 'offset', 0.0) or 0.0) * up
            top_up = st is not None and up > 0.3
            bot_up = sb is not None and up < -0.3
            upper = saa(st) if top_up else (saa(sb) if bot_up else None)
            lower = saa(sb) if top_up else (saa(st) if bot_up else None)
            m_own = (-saa(mloc) * fb * np.sign(up) * l) if abs(up) > 0.7 else 0.0
            segs.append(dict(k=k, z=zc, l=l, t=thk[k], s=saa(sm), up=upper, lo=lower, fm=fm,
                             steel=bool(steel[k]), m_own=m_own, lat=float((0.5 * (q0 + q1) - c1) @ across)))
        if segs:
            conc = [g for g in segs if not g['steel']]
            fitset = conc if len(conc) >= 3 else segs
            w = np.array([g['t'] * g['l'] for g in fitset]); z = np.array([g['z'] for g in fitset])
            sv = np.array([g['s'] for g in fitset])
            Am = np.c_[np.ones_like(z), z] * np.sqrt(w)[:, None]
            (alpha, beta), *_ = np.linalg.lstsq(Am, sv * np.sqrt(w), rcond=None)
            zbar = -alpha / beta if abs(beta) > 1e-12 else float('nan')
            N_tot = sum(g['s'] * g['t'] * g['l'] * g['fm'] for g in segs)
            zr = zbar if np.isfinite(zbar) else float(np.mean(z))
            M_fe = -sum(g['s'] * g['t'] * g['l'] * g['fm'] * (g['z'] - zr) for g in segs) + sum(g['m_own'] for g in segs)
            # statics: everything on the first-support side of the section (loads + reactions)
            left = s_node < L / 2.0
            f_tot = F_all + Rf[:, :3]
            b_ax = np.cross([0.0, 0.0, 1.0], ax)
            O = p0.copy(); O[2] = zr
            M_stat = float((np.cross(P[left] - O, f_tot[left]) @ b_ax).sum())
            V_stat = float(f_tot[left, 2].sum())
            I_eff = -M_stat / beta if abs(beta) > 1e-12 else float('nan')
            out.append(f"  section plane at s = L/2 = {L/2:.4f} m; {len(segs)} element cuts "
                       f"({len(conc)} concrete, {len(segs)-len(conc)} steel)")
            out.append(f"  bending moment from statics (loads + reactions on one side) M = {M_stat/1e3:+.3f} kN·m   "
                       f"(sagging +), shear V = {V_stat/1e3:+.3f} kN")
            out.append(f"  bending moment from the FE stresses in the cut         M = {M_fe/1e3:+.3f} kN·m   "
                       f"(axial force N = {N_tot/1e3:+.3f} kN, should be ~0; rebar and shell offsets included)")
            out.append(f"  neutral axis from the concrete membrane stresses: zbar = {zbar:.4f} m "
                       f"(sigma = {alpha/1e6:+.3f} {beta/1e6:+.3f} z MPa)")
            out.append(f"  effective I (concrete units, includes rebar/steel as the FE sees it) "
                       f"= M / slope = {I_eff:.5f} m4")
            if conc:
                gmin_z = min(conc, key=lambda g: g['z']); gmax_z = max(conc, key=lambda g: g['z'])
                tens = max(conc, key=lambda g: max(g['s'], g['lo'] if g['lo'] is not None else -1e30))
                comp = min(conc, key=lambda g: min(g['s'], g['up'] if g['up'] is not None else 1e30))
                vt = max(tens['s'], tens['lo'] if tens['lo'] is not None else -1e30)
                vc = min(comp['s'], comp['up'] if comp['up'] is not None else 1e30)
                out.append(f"  concrete, along the span: max tension     {vt/1e6:+.3f} MPa at z = {tens['z']:.3f} m "
                           f"(e = {tens['lat']:+.3f})")
                out.append(f"                            max compression {vc/1e6:+.3f} MPa at z = {comp['z']:.3f} m "
                           f"(e = {comp['lat']:+.3f})")
                out.append(f"  concrete membrane stress at the lowest cut (z = {gmin_z['z']:.3f}) {gmin_z['s']/1e6:+.3f} MPa, "
                           f"at the highest cut (z = {gmax_z['z']:.3f}) {gmax_z['s']/1e6:+.3f} MPa")
                S.update(mid_sigma_tension_MPa=round(vt / 1e6, 4), mid_sigma_compression_MPa=round(vc / 1e6, 4))
            st_ = [g for g in segs if g['steel']]
            if st_:
                gs = max(st_, key=lambda g: abs(g['s']))
                out.append(f"  steel (railing) in the cut: largest membrane stress {gs['s']/1e6:+.3f} MPa at z = {gs['z']:.3f} m")
            out.append("  z (m)*     sigma_membrane (MPa)  upper face  lower face   t (m)   length (m)   steel")
            def _f(v):
                return f"{v/1e6:+8.3f}" if v is not None else "       -"
            for g in sorted(segs, key=lambda g: g['z'])[:: max(1, len(segs) // 40)]:
                out.append(f"  {g['z']:7.3f}   {g['s']/1e6:+10.3f}        {_f(g['up'])}   {_f(g['lo'])}   "
                           f"{g['t']:.3f}   {g['l']:.4f}   {'yes' if g['steel'] else ''}")
            out.append("  (* z of the shell mid-surface, offset included. Upper/lower face: horizontal plates only;")
            out.append("   '-' = vertical plate (web): use the membrane value. Stresses are concrete stresses.)")
            S.update(M_mid_statics_kNm=round(M_stat / 1e3, 3), M_mid_FE_kNm=round(M_fe / 1e3, 3),
                     zbar_m=round(float(zbar), 4), I_eff_m4=round(float(I_eff), 6))

    # ================================================================ energy
    head("8. CHECKS")
    out.append(f"  strain energy U = {getattr(res, 'total_strain_energy', 0.0):.4e} J")
    # closure of the reactions with the loads, forces and moments (about
    # the model centre). Both should be at round-off level.
    O = P.mean(0)
    Fc = Rf[:, :3].sum(0) + F_all.sum(0)
    Mc = (Rf[:, 3:6].sum(0) + np.cross(P - O, Rf[:, :3]).sum(0) + np.cross(P - O, F_all).sum(0))
    m_ref = max([float(np.linalg.norm(sp['M'])) for sp in sups] + [1e-9])
    out.append(f"  equilibrium, forces : sum of reactions + loads = ({Fc[0]/1e3:+.4f}, {Fc[1]/1e3:+.4f}, "
               f"{Fc[2]/1e3:+.4f}) kN  ({np.linalg.norm(Fc)/max(np.linalg.norm(F_all.sum(0)),1e-9)*100:.2e} % of the load)")
    out.append(f"  equilibrium, moments: about the model centre = ({Mc[0]/1e3:+.4f}, {Mc[1]/1e3:+.4f}, "
               f"{Mc[2]/1e3:+.4f}) kN·m  (largest support couple {m_ref/1e3:.3f} kN·m)")
    out.append(f"  max von Mises (all) = {res.max_von_mises/1e6:.3f} MPa, "
               f"max sigma1 = {getattr(res, 'max_principal_tension', 0.0)/1e6:+.3f} MPa, "
               f"min sigma2 = {getattr(res, 'min_principal_compression', 0.0)/1e6:+.3f} MPa")

    # ---- beams, trusses and connections
    _bf = getattr(res, 'beam_forces', None) or {}
    _tf = getattr(res, 'truss_axial_forces', None) or {}
    if _bf or _tf:
        head("9. BEAMS AND TRUSSES")
    if _bf:
        out.append("  Beams: internal forces on the positive face in the local axes (x along the member from its first "
                   "node, z up, y = z × x); N > 0 tension.")
        out.append("  Stresses: N/A ± |My| c_z/Iy ± |Mz| c_y/Iz at the corners (round sections: the resultant moment); "
                   "largest and smallest of the two ends and mid-length.")
        _st = ('i', 'mid', 'j')
        out.append(f"  {len(_bf)} beam elements; largest |N| "
                   f"{max(abs(b[s][0]) for b in _bf.values() for s in _st)/1e3:.3f} kN, largest |M| "
                   f"{max(math.hypot(b[s][4], b[s][5]) for b in _bf.values() for s in _st)/1e3:.3f} kN·m, stress "
                   f"{min(b['smin'] for b in _bf.values())/1e6:+.2f} to {max(b['smax'] for b in _bf.values())/1e6:+.2f} MPa")
        out.append("  element  section          L (m)  station     N kN    Vy kN    Vz kN  T kN·m  My kN·m  Mz kN·m"
                   "   smax MPa  smin MPa")
        _rk = sorted(_bf.items(), key=lambda kv: -max(abs(kv[1]['smax']), abs(kv[1]['smin'])))
        for eid, b in _rk[:20]:
            for s in _st:
                _Sv = b[s]
                out.append(f"  {eid:>7}  {str(b['section'])[:15]:<15} {b['L']:6.3f}  {s:>7} "
                           + " ".join(f"{x/1e3:+8.3f}" for x in _Sv)
                           + (f"  {b['smax']/1e6:+9.3f} {b['smin']/1e6:+9.3f}" if s == 'i' else ""))
        if len(_rk) > 20:
            out.append(f"  ... {len(_rk) - 20} more beams (the 20 with the largest stress are listed)")
    if _tf:
        _ts = getattr(res, 'truss_axial_stresses', None) or {}
        out.append(f"  Trusses: {len(_tf)} bars, N {min(_tf.values())/1e3:+.3f} to {max(_tf.values())/1e3:+.3f} kN, "
                   f"stress {min(_ts.values())/1e6:+.2f} to {max(_ts.values())/1e6:+.2f} MPa")
    _cf = getattr(res, 'connection_forces', None) or {}
    if _cf:
        head("10. CONNECTIONS (what side A transmits to side B, about the joint centre, global axes)")
        out.append("  joint        type    method  nodes A/B  split  centre x, y, z (m)          F x, y, z (kN)"
                   "                M x, y, z (kN·m)              turn (deg)  M about the hinge axis")
        for nm, c in _cf.items():
            ctr, F_, M_ = c['centre'], c['F'], c['M']
            if c['method'] == 'merged':
                out.append(f"  {nm[:12]:<12} {c['ctype']:<7} {c['method']:<6} {c['nA']:>4}/{c['nB']:<4} {c['split']:>5}  "
                           f"{ctr[0]:+7.3f} {ctr[1]:+7.3f} {ctr[2]:+7.3f}   continuous: the two sides share their nodes "
                           "(coincident vertices are merged), so no link carries a force here")
                continue
            out.append(f"  {nm[:12]:<12} {c['ctype']:<7} {c['method']:<6} {c['nA']:>4}/{c['nB']:<4} {c['split']:>5}  "
                       f"{ctr[0]:+7.3f} {ctr[1]:+7.3f} {ctr[2]:+7.3f}   "
                       f"{F_[0]/1e3:+8.3f} {F_[1]/1e3:+8.3f} {F_[2]/1e3:+8.3f}   "
                       f"{M_[0]/1e3:+8.3f} {M_[1]/1e3:+8.3f} {M_[2]/1e3:+8.3f}   "
                       + (f"{np.degrees(c.get('rot', 0.0)):10.4f}  " if c['ctype'] != 'FIXED' else f"{'-':>10}  ")
                       + (f"{c['M_axis']/1e3:+.4f} (axis {c['axis'][0]:+.3f} {c['axis'][1]:+.3f} {c['axis'][2]:+.3f})"
                          if c['M_axis'] is not None else "-"))
        out.append("  FIXED: every side-B node tied to the nearest side-A node (all six components); PINNED: force "
                   "only, through the joint centre; HINGE: rotation about the axis free ('pairs' = a hinge line).")
        _big = [(n_, c_['rot']) for n_, c_ in _cf.items() if c_['ctype'] != 'FIXED' and c_.get('rot', 0.0) > PERM_CONN_ROT_WARN]
        if _big:
            out.append(f"  ⚠ {len(_big)} joint(s) turn more than {np.degrees(PERM_CONN_ROT_WARN):.2f}° (largest "
                       f"{np.degrees(max(r for _, r in _big)):.2f}°): a part may be free to turn about its joints (a "
                       "mechanism; pinned joints on one line let the part topple about that line). These results are "
                       "not valid; use HINGE about an axis the structure holds, or FIXED.")

    # ================================================================ write
    folder = _perm_results_dir(scene)
    base = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (case or src_obj.name))[:60]
    path = os.path.join(folder, f"{base}_{now:%Y%m%d_%H%M%S}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    S["report_file"] = os.path.basename(path)
    cols = ["date", "case", "model", "mass_kg", "weight_kN", "L_m", "P_total_kN", "uz_under_load1_mm",
            "max_u_mm", "max_down_uz_mm", "mid_soffit_uz_mm", "mid_girder1_uz_mm", "mid_girder2_uz_mm",
            "VM_top_conc_MPa", "VM_bot_conc_MPa", "s1_top_conc_MPa", "s1_bot_conc_MPa", "s2_top_conc_MPa",
            "s2_bot_conc_MPa", "VM_top_steel_MPa", "VM_bot_steel_MPa", "s1_top_steel_MPa", "s1_bot_steel_MPa",
            "s2_top_steel_MPa", "s2_bot_steel_MPa",
            "VM_top_conc_span_MPa", "VM_bot_conc_span_MPa", "s1_top_conc_span_MPa", "s1_bot_conc_span_MPa",
            "s2_top_conc_span_MPa", "s2_bot_conc_span_MPa",
            "VM_top_conc_away_MPa", "VM_bot_conc_away_MPa", "s1_top_conc_away_MPa", "s1_bot_conc_away_MPa",
            "s2_top_conc_away_MPa", "s2_bot_conc_away_MPa",
            "VM_top_steel_away_MPa", "VM_bot_steel_away_MPa", "s1_top_steel_away_MPa", "s1_bot_steel_away_MPa",
            "s2_top_steel_away_MPa", "s2_bot_steel_away_MPa",
            "mid_sigma_tension_MPa", "mid_sigma_compression_MPa",
            "M_mid_statics_kNm", "M_mid_FE_kNm", "zbar_m", "I_eff_m4"]
    cols += sorted(k for k in S if k.startswith("R") and k.endswith("_Fz_kN"))
    cols += ["report_file"]
    log = os.path.join(folder, "PERM_results_log.csv")
    new = not os.path.exists(log)
    if not new:
        with open(log, encoding="utf-8") as fh:
            first = fh.readline().strip()
        if first != ",".join(cols):
            log = os.path.join(folder, "PERM_results_log_" + f"{now:%Y%m%d_%H%M%S}.csv"); new = True
    with open(log, "a", encoding="utf-8") as fh:
        if new:
            fh.write(",".join(cols) + "\n")
        fh.write(",".join('"' + str(S.get(c, "")).replace('"', "'") + '"' if c in ("case", "model", "report_file")
                          else str(S.get(c, "")) for c in cols) + "\n")
    try:
        txt = bpy.data.texts.get("PERM_Results.txt") or bpy.data.texts.new("PERM_Results.txt")
        txt.clear(); txt.write("\n".join(out) + "\n")
    except Exception:
        pass
    scene["perm_last_report"] = path
    print(f"[PERM Results] report saved: {path}")
    print(f"[PERM Results] one line added to: {log}")
    return path


class PERM_OT_OpenResultsFolder(bpy.types.Operator):
    """Open the folder with the results reports (one .txt per analysis + PERM_results_log.csv)"""
    bl_idname = "perm.open_results_folder"
    bl_label = "Open Results Folder"

    def execute(self, context):
        d = _perm_results_dir(context.scene)
        try:
            bpy.ops.wm.path_open(filepath=d)
        except Exception as e:                         # no file manager (a server, a container)
            print(f"[PERM] could not open the folder: {e}")
        self.report({'INFO'}, f"PERM reports are in {d}")
        return {'FINISHED'}


class PERM_PT_SolverPanel(bpy.types.Panel):
    bl_label = "PERM Solver"
    bl_idname = "PERM_PT_solver"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PERM'

    def _draw_results(self, context, layout, scn):
        # ── 9. RESULTS / VISUALIZATION ───────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="9. Results Visualization", icon='COLOR')
        row = box.row(align=True)
        row.prop(scn, "perm_principal_colormap", text="σ1, σ2")
        row.prop(scn, "perm_show_color_legend", text="Legend", toggle=True)
        # which faces are coloured, and the colour range
        cb = box.box()
        _perm_label_wrapped(cb, "Stress colours: which faces, which range", icon='MOD_MASK')
        cb.row(align=True).prop(scn, "perm_viz_filter", expand=True)
        _mc = _perm_material_classes(scn)
        for _c, _lab in ((1, "E < 100 GPa"), (2, "E ≥ 100 GPa"), (3, "Massless")):
            _nm = [k for k, v in _mc.items() if v == _c]
            if _nm:
                cb.label(text=f"{_lab}: {', '.join(_nm)}")
        if scn.perm_viz_filter == 'CONCRETE' and any(v == 3 for v in _mc.values()):
            cb.prop(scn, "perm_viz_links")
        cb.row(align=True).prop(scn, "perm_color_range", expand=True)
        if scn.perm_color_range == 'AUTO':
            cb.prop(scn, "perm_clip_percent")
        _tgt = _perm_range_target(context)
        if _tgt is not None:
            _perm_label_wrapped(cb, f"Range of '{_tgt.name}' ({_PERM_FILTER_NOTE.get(scn.perm_viz_filter, 'all faces')}):")
            row = cb.row(align=True)
            row.prop(scn, "perm_range_min", text="Min")
            row.prop(scn, "perm_range_max", text="Max")
            row.operator("perm.auto_color_range", text="", icon='LOOP_BACK')
            _perm_label_wrapped(cb, "Typing a value switches to Manual for this map", icon='INFO')
        cb.operator("perm.recolor_results", text="Repaint All Result Maps", icon='BRUSH_DATA')
        box.label(text="Deformation scale (live slider):")
        row = box.row(align=True)
        row.prop(scn, "perm_deformation_scale", text="", slider=True)
        row.operator("perm.refresh_visualization", text="", icon='FILE_REFRESH')

        box.separator()
        _perm_label_wrapped(box, "Result opacity (texture shows through below 1.0):")
        row = box.row(align=True)
        row.prop(scn, "perm_result_opacity", text="", slider=True)
        row.operator("perm.rebuild_result_materials", text="", icon='SHADING_TEXTURE')
        box.separator()
        _perm_label_wrapped(box, "Inspect (hover a node for its number and values):")
        _running = bool(getattr(scn, 'perm_inspect_running', False))
        box.operator("perm.inspect_results",
                     text="Stop Inspecting" if _running else "Inspect Results (hover)",
                     icon='PAUSE' if _running else 'EYEDROPPER', depress=_running)
        row = box.row(align=True)
        row.prop(scn, "perm_show_reaction_labels", text="Reactions", toggle=True)
        row.prop(scn, "perm_reaction_scale", text="Arrows", slider=True)
        box.separator()
        box.operator("perm.clear_results", text="Clear Results (result meshes)", icon='TRASH')
        if getattr(scn, 'perm_result_opacity', 1.0) < 1.0:
            has_mat_view = any(
                sp.shading.type in ('MATERIAL', 'RENDERED')
                for ar in context.screen.areas if ar.type == 'VIEW_3D'
                for sp in ar.spaces if sp.type == 'VIEW_3D')
            if not has_mat_view:
                _perm_label_wrapped(box, "Switch viewport to Material Preview to see it", icon='INFO')


    def draw(self, context):
        layout = self.layout
        scn = context.scene
        obj = context.active_object
        src = _perm_resolve_source(context)
        if src is None:
            _perm_draw_scipy_notice(layout)
            layout.label(text="Select the mesh to analyse", icon='ERROR'); return
        _perm_draw_scipy_notice(layout)
        _perm_draw_unit_notice(layout, scn)
        hdr = layout.box()
        hdr.label(text=f"Model: {src.name}", icon='PHYSICS')
        row = hdr.row(align=True)
        row.operator("perm.edit_model", text="Edit Model", icon='EDITMODE_HLT')
        row.operator("perm.show_results", text="Show Results", icon='HIDE_OFF')
        row = hdr.row(align=True)
        row.operator("perm.select_problem_faces", text="Problems", icon='ERROR')
        _cnt = scn.get("perm_unassigned_counts")
        row2 = hdr.row(align=True)
        op = row2.operator("perm.select_unassigned_faces", text="Unassigned Faces",
                           icon='SELECT_SET')
        op.kind = 'ANY'
        if _cnt is not None:
            _ns, _nm = int(_cnt[0]), int(_cnt[1])
            row2.label(text=("all assigned" if not (_ns or _nm)
                             else f"{_ns} no section · {_nm} no material"),
                       icon='CHECKMARK' if not (_ns or _nm) else 'ERROR')
        row.operator("perm.select_peak_faces", text="Peak Stress", icon='RESTRICT_SELECT_OFF')
        if obj is not src:
            _perm_label_wrapped(hdr, f"Active: {getattr(obj, 'name', 'none')} - showing the model's settings", icon='INFO')
            _perm_label_wrapped(hdr, "Assign in Edit Mode on the model ('Edit Model' selects it)")
        obj = src            # every section below shows the MODEL's assignments

        # ── 1. MATERIALS ─────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="1. Materials", icon='MATERIAL')
        row = box.row()
        row.template_list("PERM_UL_Materials", "", scn, "perm_materials",
                          scn, "perm_material_index", rows=3)
        col = row.column(align=True)
        col.operator("perm.add_material", icon='ADD', text="")
        col.operator("perm.delete_material", icon='REMOVE', text="")
        if 0 <= scn.perm_material_index < len(scn.perm_materials):
            mat = scn.perm_materials[scn.perm_material_index]
            col = box.column(align=True)
            col.prop(mat, "name")
            col.prop(mat, "youngs_modulus")
            col.prop(mat, "poissons_ratio")
            col.prop(mat, "density")
            box.label(text="Edit Mode → select faces → assign:")
            row = box.row(align=True)
            row.operator("perm.assign_material_to_faces",
                         text="Assign", icon='CHECKMARK')
            row.operator("perm.select_material_faces",
                         text="Select", icon='RESTRICT_SELECT_OFF')
            row.operator("perm.unassign_material",
                         text="Clear", icon='X')
            box.operator("perm.select_unassigned_faces", text="Select Faces Without Material",
                         icon='SELECT_SET').kind = 'MATERIAL'

        # ── 2. CROSS-SECTIONS ────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="2. Cross-Sections", icon='MESH_CYLINDER')
        row = box.row()
        row.template_list("PERM_UL_Sections", "", scn, "perm_sections",
                          scn, "perm_section_index", rows=3)
        col = row.column(align=True)
        col.operator("perm.add_section", icon='ADD', text="")
        col.operator("perm.delete_section", icon='REMOVE', text="")
        if 0 <= scn.perm_section_index < len(scn.perm_sections):
            sec = scn.perm_sections[scn.perm_section_index]
            col = box.column(align=True)
            col.prop(sec, "name")
            col.prop(sec, "thickness")
            col.prop(sec, "area")
            # beam section (BEAM_B2 elements)
            bx = box.box()
            bx.label(text="Beam section (BEAM_B2):", icon='MOD_WIREFRAME')
            bx.prop(sec, "beam_shape", text="Shape")
            c2 = bx.column(align=True)
            if sec.beam_shape == 'GENERIC':
                c2.label(text="A = the area above")
                c2.prop(sec, "Iy"); c2.prop(sec, "Iz"); c2.prop(sec, "J")
                c2.prop(sec, "shear_factor"); c2.prop(sec, "fibre_y"); c2.prop(sec, "fibre_z")
            else:
                if sec.beam_shape in ('RECT', 'RHS'):
                    c2.prop(sec, "dim_b"); c2.prop(sec, "dim_h")
                if sec.beam_shape in ('CIRC', 'CHS'):
                    c2.prop(sec, "dim_d")
                if sec.beam_shape in ('RHS', 'CHS'):
                    c2.prop(sec, "dim_t")
                _bp = _perm_beam_props(sec)
                _c3 = bx.column(align=True)
                _c3.label(text=f"A {_bp['A']:.4g} m²,  J {_bp['J']:.4g} m⁴")
                _c3.label(text=f"Iy {_bp['Iy']:.4g},  Iz {_bp['Iz']:.4g} m⁴")
            bx.prop(sec, "roll_deg")
            _perm_label_wrapped(bx, "Local axes: x along the edge, z up, y = z × x")
            # Reinforced concrete (linear transformed section). ρ=0 → plain concrete.
            rc = box.box()
            rc.label(text="Smeared rebar (ρ = 0 → none):", icon='MOD_LATTICE')
            rc.prop(sec, "rebar_ratio")
            if sec.rebar_ratio > 0.0:
                rc.prop(sec, "rebar_cover")
                rc.prop(sec, "steel_modulus")
            _perm_label_wrapped(box, "Shell sections (thickness): assign to faces")
            row = box.row(align=True)
            row.operator("perm.assign_section_to_faces",
                         text="Assign", icon='CHECKMARK')
            row.operator("perm.select_section_faces",
                         text="Select", icon='RESTRICT_SELECT_OFF')
            row.operator("perm.unassign_section",
                         text="Clear", icon='X')
            box.operator("perm.select_unassigned_faces", text="Select Faces Without a Section",
                         icon='SELECT_SET').kind = 'SECTION'
            _perm_label_wrapped(box, "Truss area: read from its section at solve time", icon='INFO')

        # ── 2b. SHELL OFFSET ───────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="2b. Shell Offset", icon='MOD_SOLIDIFY')
        _perm_label_wrapped(box, "Which side of the mesh surface the thickness lies on")
        col = box.column(align=True)
        col.prop(scn, "perm_offset_half")
        if not scn.perm_offset_half:
            col.prop(scn, "perm_offset_distance")
        col.prop(scn, "perm_offset_direction")
        box.label(text="Edit Mode → select faces → set:")
        row = box.row(align=True)
        row.operator("perm.set_shell_offset", text="Set", icon='CHECKMARK')
        row.operator("perm.select_offset_faces", text="Select", icon='RESTRICT_SELECT_OFF')
        row.operator("perm.clear_shell_offset", text="Clear", icon='X')
        row = box.row(align=True)
        row.operator("perm.preview_thickness", text="Preview", icon='HIDE_OFF')
        row.operator("perm.clear_thickness_preview", text="End Preview", icon='TRASH')
        _n_off = int(obj.get("perm_offset_count", 0) or 0)
        _perm_label_wrapped(box, (f"{_n_off} faces carry an offset" if _n_off else "No offsets: thickness centred on the mesh"),
                            icon='INFO')

        # ── 3. ELEMENT TYPES ─────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="3. Element Types", icon='MESH_DATA')
        # Mesh topology target — decides quadrangulate vs triangulate before the
        # FE model is built. Belongs here, not in the Run dialog: it is an
        # element-level decision.
        col = box.column(align=True)
        col.prop(scn, "perm_mesh_topology", text="Mesh")
        if scn.perm_mesh_topology == 'TRI3':
            _perm_label_wrapped(col, "Triangles (CST + MITC3): less accurate than quads on the same mesh", icon='INFO')
        box.separator()
        _perm_label_wrapped(box, "Per-face types (unassigned faces = full bending shell):")
        row = box.row()
        row.template_list("PERM_UL_ElementTypes", "", scn, "perm_element_types",
                          scn, "perm_element_type_index", rows=3)
        col = row.column(align=True)
        col.operator("perm.add_element_type", icon='ADD', text="")
        col.operator("perm.delete_element_type", icon='REMOVE', text="")
        if 0 <= scn.perm_element_type_index < len(scn.perm_element_types):
            et = scn.perm_element_types[scn.perm_element_type_index]
            col = box.column(align=True)
            col.prop(et, "name")
            col.prop(et, "kind")
            if et.kind in _PERM_LINE_TYPES:
                _perm_label_wrapped(box, "Select EDGES in Edit Mode → Assign (1 bar per edge)", icon='INFO')
                _m = (scn.perm_materials[scn.perm_material_index].name
                      if 0 <= scn.perm_material_index < len(scn.perm_materials) else "—")
                _s = (scn.perm_sections[scn.perm_section_index].name
                      if 0 <= scn.perm_section_index < len(scn.perm_sections) else "—")
                _perm_label_wrapped(box, f"Uses active material '{_m}' + section '{_s}'")
                box.label(text=f"({et.truss_count} {'beams' if et.kind == 'BEAM_B2' else 'truss bars'})")
                box.operator("perm.remove_truss_from_edges",
                             text="Remove from Selected Edges", icon='X')
            else:
                box.label(text="Select faces in Edit Mode → Assign", icon='INFO')
            row = box.row(align=True)
            row.operator("perm.assign_element_type",
                         text="Assign", icon='CHECKMARK')
            row.operator("perm.select_element_type_faces",
                         text="Select", icon='RESTRICT_SELECT_OFF')
            row.operator("perm.unassign_element_type",
                         text="Clear", icon='X')

        # ── 4. RESTRAINTS (BCs) ──────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="4. Restraints (BCs)", icon='CONSTRAINT')
        support_groups = find_perm_support_groups(obj)
        if support_groups:
            for group_name, suffix, bc_type in support_groups:
                ids = get_group_vertex_indices(obj, group_name)
                row = box.row(align=True)
                _bt = bc_type if bc_type != 'CUSTOM' else "[" + " ".join(perm_support_dofs(obj, suffix)) + "]"
                row.label(text=f"{suffix}: {_bt} ({len(ids)} verts)",
                          icon='CHECKMARK')
                op = row.operator("perm.select_assignment", text="", icon='RESTRICT_SELECT_OFF')
                op.group_name = group_name
                op = row.operator("perm.deselect_assignment", text="", icon='RESTRICT_SELECT_ON')
                op.group_name = group_name
                op = row.operator("perm.remove_assignment", text="", icon='X')
                op.group_name = group_name
                _draw_node_list_for_group(box, obj, group_name, scn)
        else:
            box.label(text="No restraints assigned yet", icon='INFO')
        box.label(text="Edit mode → select verts → Assign:")
        box.operator("perm.assign_bc", text="Assign Restraint to Selection",
                     icon='ADD')

        # ── 4b. CONNECTIONS ────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="4b. Connections", icon='LINKED')
        _perm_label_wrapped(box, "Joints between separate parts of the mesh")
        row = box.row()
        row.template_list("PERM_UL_Connections", "", obj, "perm_connections", obj, "perm_connection_index", rows=3)
        col = row.column(align=True)
        col.operator("perm.add_connection", icon='ADD', text="")
        col.operator("perm.remove_connection", icon='REMOVE', text="")
        row = box.row(align=True)
        row.operator("perm.detect_contacts", text="Detect Contacts", icon='VIEWZOOM')
        row.operator("perm.clear_connections", text="Remove All", icon='TRASH')
        if 0 <= obj.perm_connection_index < len(obj.perm_connections):
            _it = obj.perm_connections[obj.perm_connection_index]
            col = box.column(align=True)
            col.prop(_it, "name")
            col.prop(_it, "ctype")
            if _it.ctype == 'HINGE':
                col.prop(_it, "axis")
            col.operator("perm.connections_apply_all", text="Apply this type to all joints", icon='DUPLICATE')
            row = box.row(align=True)
            for _side, _txt in (('BOTH', "Select"), ('A', "Side A"), ('B', "Side B")):
                row.operator("perm.select_connection", text=_txt).side = _side
            box.label(text={'FIXED': "Rigid: force and moment transmitted",
                            'PINNED': "Ball joint at the centre: force, no moment",
                            'HINGE': "Rotation about the hinge axis free"}[_it.ctype], icon='INFO')
        _perm_label_wrapped(box, "Edit Mode → select the joint verts of 2 parts → +")

        # ── 5. LOADS ─────────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="5. Loads (point forces)", icon='FORCE_FORCE')
        load_groups = find_perm_load_groups(obj)
        if load_groups:
            for group_name, suffix, fx, fy, fz in load_groups:
                ids = get_group_vertex_indices(obj, group_name)
                N = len(ids) or 1
                pn = get_per_node_forces(obj, suffix) if not suffix.startswith("(") else {}
                row = box.row(align=True)
                _tf = [sum(v[k] for v in pn.values()) for k in range(3)] if pn else [fx, fy, fz]
                label = f"{suffix}: {N} verts"
                row.label(text=label, icon='CHECKMARK')
                op = row.operator("perm.select_assignment", text="", icon='RESTRICT_SELECT_OFF')
                op.group_name = group_name
                op = row.operator("perm.deselect_assignment", text="", icon='RESTRICT_SELECT_ON')
                op.group_name = group_name
                if not suffix.startswith("("):
                    op = row.operator("perm.open_per_node_editor", text="", icon='OUTLINER_OB_FONT')
                    op.group_name = group_name
                op = row.operator("perm.remove_assignment", text="", icon='X')
                op.group_name = group_name
                box.label(text=f"ΣF = ({_tf[0]:.0f}, {_tf[1]:.0f}, {_tf[2]:.0f}) N", icon='FORCE_FORCE')
                _at = obj.get(f"perm_load_{suffix}_at")
                if _at is not None and len(_at) == 3:
                    box.label(text=f"at {_at[0]:+.3f}, {_at[1]:+.3f}, {_at[2]:+.3f} m",
                              icon='EMPTY_AXIS')
                _draw_node_list_for_group(box, obj, group_name, scn)
        else:
            box.label(text="No point loads yet", icon='INFO')
        box.label(text="Edit mode → select verts → Assign:")
        box.operator("perm.assign_load_per_node", text="Assign Load (per-node)",
                     icon='ADD')
        box.operator("perm.report_loads", text="Show Load Coordinates", icon='EMPTY_AXIS')

        # ── 5b. SURFACE LOADS ─────────────────────────────────────
        layout.separator()
        _perm_draw_surface_loads(layout, scn, obj)

        # ── 5.5. SELF-WEIGHT / GRAVITY ──────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="5.5. Self-Weight (Gravity)", icon='FORCE_FORCE')
        box.prop(scn, "perm_apply_gravity", text="Apply self-weight to all elements")
        if scn.perm_apply_gravity:
            row = box.row(align=True)
            row.prop(scn, "perm_gravity_x", text="gx"); row.prop(scn, "perm_gravity_y", text="gy")
            row.prop(scn, "perm_gravity_z", text="gz")
            _g = (scn.perm_gravity_x, scn.perm_gravity_y, scn.perm_gravity_z)
            col = box.column(align=True)
            col.label(text=f"|g| = {math.sqrt(sum(c * c for c in _g)):.2f} m/s² (global X, Y, Z)", icon='INFO')
            col.label(text="mass = density × thickness × area")

        # ── 6. MARKERS ───────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="6. Markers", icon='OUTLINER_OB_EMPTY')
        box.prop(scn, "perm_marker_scale")
        row = box.row(align=True)
        row.operator("perm.show_markers", text="Show/Refresh", icon='HIDE_OFF')
        row.operator("perm.clear_markers", text="Clear", icon='HIDE_ON')
        col = box.column(align=True)
        col.label(text="FIXED → red cube")
        col.label(text="PINNED → blue cone")
        col.label(text="ROLLER_X → yellow cylinder")
        col.label(text="ROLLER_Y → green, ROLLER_Z → pink")
        col.label(text="CUSTOM → orange cube")

        # ── 7. EDIT HELPERS ──────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="7. Edit Helpers", icon='SCULPTMODE_HLT')
        box.label(text="Remove accidentally-included verts:")
        box.operator("perm.remove_from_groups",
                     text="Remove Selected from PERM Groups",
                     icon='UNLINKED')
        box.operator("perm.remove_marker",
                     text="Remove Selected Marker(s)",
                     icon='RESTRICT_SELECT_OFF')

        # ── 8. ANALYSIS ──────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="8. Analysis", icon='PLAY')
        has_any_support = bool(support_groups)
        if not has_any_support:
            box.label(text="⚠ Add at least one restraint first", icon='ERROR')
        else:
            _perm_label_wrapped(box, "Click RUN ANALYSIS, pick which results to show:")
            box.operator("perm.run_solver", text="RUN ANALYSIS", icon='PLAY')
            col = box.column(align=True)
            col.prop(scn, "perm_case_name", text="Case")
            row = col.row(align=True)
            row.prop(scn, "perm_results_dir", text="Reports")
            row.operator("perm.open_results_folder", text="", icon='FILE_FOLDER')
            _lr = scn.get("perm_last_report")
            if _lr:
                box.label(text=f"Last report: {os.path.basename(_lr)}", icon='FILE_TEXT')
            box.operator("perm.export_json", text="Export FEA JSON",
                         icon='FILE_TEXT')

        # ── 8b. CONNECTION DETAILS (options, off for a new model) ──
        _perm_draw_detail_options(layout, scn)

        self._draw_results(context, layout, scn)

        # ── 10. FIND ELEMENT ───────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="10. Find Element", icon='VIEWZOOM')
        _perm_label_wrapped(box, "Element number, from the console or the report")
        row = box.row(align=True)
        row.prop(scn, "perm_find_element", text="#")
        row.operator("perm.find_element", text="Find & Zoom", icon='ZOOM_SELECTED')
        row.operator("perm.clear_find_element", text="", icon='X')
        row = box.row(align=True)
        row.prop(scn, "perm_find_show", expand=True)
        box.prop(scn, "perm_find_distance")
        _fi = scn.get("perm_find_info")
        if _fi:
            col = box.column(align=True)
            for _ln in str(_fi).split("\n"):
                _perm_label_wrapped(col, _ln)

        # ── 11. Clear All ────────────────────────────────────────────────
        layout.separator()
        layout.operator("perm.clear_assignments",
                        text="Clear ALL Supports and Loads",
                        icon='TRASH')



class PERM_OT_ConvertMesh(bpy.types.Operator):
    """PERM: Convert surface representation to Articulated Data"""
    bl_idname = "perm.convert_mesh"
    bl_label = "PERM: Convert to Articulated Data"
    bl_options = {'REGISTER', 'UNDO'}

    element_type: bpy.props.EnumProperty(
        name="Element Type",
        items=[
            ('TRI3', "Tri3 (Linear Triangle)", "3-node linear triangle"),
            ('TRI6', "Tri6 (Quadratic Triangle)", "6-node quadratic triangle"),
            ('QUAD4', "Quad4 (Linear Quad)", "4-node linear quadrilateral"),
            ('QUAD8', "Quad8 (Quadratic Quad)", "8-node quadratic quadrilateral"),
        ],
        default='TRI3'
    )

    thickness: bpy.props.FloatProperty(
        name="Shell Thickness", default=0.01, min=0.0001, unit='LENGTH',
        description="Shell thickness in SOURCE mesh units. Scaled by Scale "
                    "Factor along with the coordinates, so the model stays in "
                    "one unit system")
    scale_factor: bpy.props.FloatProperty(
        name="Scale Factor", default=1.0, min=0.0001,
        description="Multiplies node coordinates AND shell thickness. Use it "
                    "to convert the source units to metres (e.g. 0.001 for a "
                    "mesh modelled in millimetres)")
    clean_mesh: bpy.props.BoolProperty(name="Clean Representation First", default=True)
    compute_quality: bpy.props.BoolProperty(name="Compute Quality Metrics", default=True)
    visualize: bpy.props.BoolProperty(name="Visualize Quality", default=True)
    export_format: bpy.props.EnumProperty(
        name="Export Format",
        items=[
            ('NONE', "None", "Don't export"),
            ('ABAQUS', "Abaqus (.inp)", "Export to Abaqus input format"),
            ('NASTRAN', "Nastran (.bdf)", "Export to Nastran bulk data"),
            ('JSON', "JSON", "Export to JSON format"),
        ],
        default='NONE'
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a representation object first")
            return {'CANCELLED'}

        etype_map = {'TRI3': ElementType.TRI3, 'TRI6': ElementType.TRI6, 'QUAD4': ElementType.QUAD4, 'QUAD8': ElementType.QUAD8}

        try:
            converter = PERMMeshConverter(obj)

            if self.clean_mesh:
                converter.clean_mesh()

            converter.build_fe_model(
                target_type=etype_map[self.element_type],
                thickness=self.thickness,
                scale=self.scale_factor
            )

            # N-gon faces produce no element at all: holes in the structure,
            # not cosmetic omissions. Say so where the user is looking.
            if converter.skipped_ngons:
                self.report({'WARNING'},
                    f"{len(converter.skipped_ngons)} n-gon face(s) produced NO "
                    f"element - the model has holes there. Triangulate or "
                    f"quadrangulate them (Ctrl+T in Edit Mode) and re-run.")

            if self.compute_quality:
                report = converter.compute_quality_metrics()
                if report.inverted_elements:
                    self.report({'WARNING'},
                        f"{report.inverted_elements} inverted element(s) "
                        f"(scaled Jacobian < 0). They are NOT skipped - they get corrupt "
                        f"stiffness and cause false stress peaks. Use 'Select Problem Faces'.")
                elif report.min_jacobian < 0.05:
                    self.report({'WARNING'},
                        f"Worst scaled Jacobian {report.min_jacobian:.3f} - near "
                        f"degenerate. Those elements are the ones most likely to "
                        f"fail in the solver.")

            if self.visualize and self.compute_quality:
                converter.visualize_quality_on_mesh(metric='skewness')

            if self.export_format != 'NONE':
                base_path = bpy.path.abspath("//")
                if not base_path: base_path = os.path.expanduser("~/")

                if self.export_format == 'ABAQUS':
                    converter.export_abaqus_inp(os.path.join(base_path, f"{obj.name}_PERM.inp"))
                elif self.export_format == 'NASTRAN':
                    converter.export_nastran_bdf(os.path.join(base_path, f"{obj.name}_PERM.bdf"))
                elif self.export_format == 'JSON':
                    converter.export_json(os.path.join(base_path, f"{obj.name}_PERM.json"))

            print(converter.get_model_summary())

            obj['perm_fe_model'] = True
            obj['perm_element_type'] = self.element_type
            obj['perm_node_count'] = len(converter.nodes)
            obj['perm_element_count'] = len(converter.elements)

            self.report({'INFO'}, f"PERM: {len(converter.nodes)} nodes, {len(converter.elements)} elements created")

        except Exception as e:
            self.report({'ERROR'}, f"PERM Error: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "element_type")
        layout.prop(self, "thickness")
        layout.prop(self, "scale_factor")
        layout.separator()
        layout.prop(self, "clean_mesh")
        layout.prop(self, "compute_quality")
        layout.prop(self, "visualize")
        layout.separator()
        layout.prop(self, "export_format")


class PERM_PT_MainPanel(bpy.types.Panel):
    bl_label = "PERM Articulated Data"
    bl_idname = "PERM_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PERM'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if not obj or obj.type != 'MESH':
            layout.label(text="Select representation object", icon='ERROR')
            return

        layout.label(text=f"Object: {obj.name}", icon='MESH_DATA')
        layout.label(text=f"Vertices: {len(obj.data.vertices)}")
        layout.label(text=f"Faces: {len(obj.data.polygons)}")

        layout.separator()
        layout.operator("perm.convert_mesh", icon='MOD_TRIANGULATE')

        if obj.get('perm_fe_model'):
            layout.separator()
            box = layout.box()
            box.label(text="Articulated Data Info", icon='PHYSICS')
            box.label(text=f"Type: {obj.get('perm_element_type', 'N/A')}")
            box.label(text=f"Nodes: {obj.get('perm_node_count', 0)}")
            box.label(text=f"Elements: {obj.get('perm_element_count', 0)}")



class PERM_OT_RebuildResultMaterials(bpy.types.Operator):
    """PERM: rebuild the blended material on every result object.

    Needed for result objects from older files, or after the
    source material has changed. Also switches the viewport to Material
    Preview, without which the blend cannot be seen.
    """
    bl_idname = "perm.rebuild_result_materials"
    bl_label = "PERM: Rebuild Result Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        objs = perm_result_viz_objects()
        if not objs:
            self.report({'WARNING'},
                        "No PERM result objects found. Run the analysis first.")
            return {'CANCELLED'}
        built = 0
        for o in objs:
            layer = o.get("perm_result_layer")
            if not layer:
                comp = o.get("perm_disp_component")
                layer = f"DISP_{comp}" if comp else None
            if not layer:
                # Last resort: whatever colour attribute the mesh carries.
                ca = getattr(o.data, 'color_attributes', None)
                layer = ca[0].name if ca and len(ca) else None
            if not layer:
                print(f"[PERM Solver] '{o.name}': no colour attribute to map, skipped")
                continue
            o["perm_result_viz"] = True
            o["perm_result_layer"] = layer
            if perm_build_result_material(o, layer):
                built += 1
        _perm_set_material_preview()
        _update_result_opacity(self, context)
        self.report({'INFO'},
                    f"PERM: rebuilt {built} result material(s). "
                    f"Opacity {getattr(context.scene, 'perm_result_opacity', 1.0):.2f}")
        return {'FINISHED'}


classes = (
    # Stage 1: mesh conversion
    PERM_OT_ConvertMesh, PERM_PT_MainPanel,
    # Solve + Export + Run (kept)
    PERM_OT_RunSolver, PERM_OT_ExportJSON,
    PERM_OT_RefreshVisualization, PERM_OT_RebuildResultMaterials,
    # Per-node force editor
    PERM_PerNodeForce, PERM_UL_PerNodeForces,
    PERM_OT_OpenPerNodeEditor, PERM_OT_BulkSetPerNode,
    # Individual removal
    PERM_OT_RemoveSelectedFromGroups, PERM_OT_RemoveMarker,
    # Visual markers
    PERM_OT_ShowMarkers, PERM_OT_ClearMarkers,
    # Clear all
    PERM_OT_ClearAssignments,
    # Select group helper (used by element-type panel rows)
    PERM_OT_SelectGroup,
    # === Structured BC/Load operators ===
    PERM_OT_AssignBC, PERM_OT_AssignLoadPerNode, PERM_OT_ReportLoads,
    PERM_OT_SelectAssignment, PERM_OT_DeselectAssignment, PERM_OT_RemoveAssignment,
    # === Materials & sections ===
    PERM_MaterialPG, PERM_SectionPG,
    PERM_UL_Materials, PERM_UL_Sections,
    PERM_OT_AddMaterial, PERM_OT_DeleteMaterial,
    PERM_OT_AddSection, PERM_OT_DeleteSection,
    PERM_OT_AssignMaterialToFaces, PERM_OT_AssignSectionToFaces,
    PERM_OT_SelectMaterialFaces, PERM_OT_SelectSectionFaces,
    PERM_OT_UnassignMaterial, PERM_OT_UnassignSection,
    # === Connections ===
    PERM_ConnectionPG, PERM_UL_Connections, PERM_OT_AddConnection, PERM_OT_DetectContacts,
    PERM_OT_RemoveConnection, PERM_OT_ClearConnections, PERM_OT_SelectConnection, PERM_OT_ConnectionsApplyAll,
    # === Element Types (replaces Add/Edit/DeleteTruss) ===
    PERM_ElementTypePG, PERM_UL_ElementTypes,
    PERM_OT_AddElementType, PERM_OT_DeleteElementType,
    PERM_OT_AssignElementType,
    PERM_OT_SelectElementTypeFaces, PERM_OT_UnassignElementType,
    PERM_OT_RemoveTrussFromEdges,
    PERM_OT_InspectResults,
    PERM_OT_EditModel, PERM_OT_ShowResults, PERM_OT_ClearResults,
    PERM_OT_SelectProblemFaces, PERM_OT_SelectPeakFaces, PERM_OT_SelectUnassignedFaces,
    # Find element
    PERM_OT_FindElement, PERM_OT_ClearFindElement,
    # Colour filter and range
    PERM_OT_RecolorResults, PERM_OT_AutoColorRange,
    # Node-list expand toggle
    PERM_OT_ToggleNodeList,
    # Shell offset
    PERM_OT_SetShellOffset, PERM_OT_ClearShellOffset, PERM_OT_SelectOffsetFaces,
    PERM_OT_PreviewThickness, PERM_OT_ClearThicknessPreview,
    # Results report
    PERM_OT_OpenResultsFolder,
    # Detail options
    PERM_OT_BridgePreset, PERM_OT_ResetDetails, PERM_OT_InstallSciPy,
    # Surface loads
    PERM_OT_SetSurfaceLoad, PERM_OT_ClearSurfaceLoad, PERM_OT_SelectSurfaceLoadFaces,
    # Panel last
    PERM_PT_SolverPanel,
)

def register():
    global _PERM_LABEL_HANDLE
    for cls in classes: bpy.utils.register_class(cls)
    # colour scale, legend, results report
    bpy.types.Scene.perm_principal_colormap = bpy.props.EnumProperty(
        name="Principal stress colours",
        items=[('RAINBOW', "Rainbow (RGB)", "Blue = lowest, red = highest (like von Mises); read values on the legend"),
               ('DIVERGING', "Blue-white-red", "Blue = compression, white = zero, red = tension")],
        default='RAINBOW', update=_perm_update_colors,
        description="Colour scale for sigma1/sigma2 maps")
    bpy.types.Scene.perm_show_color_legend = bpy.props.BoolProperty(
        name="Colour legend", default=True,
        description="Show the colour bar with values for the visible result mesh")
    bpy.types.Scene.perm_case_name = bpy.props.StringProperty(
        name="Case", default="",
        description="Name of this analysis (e.g. V1_1kN_midspan); used for the report file name")
    bpy.types.Scene.perm_results_dir = bpy.props.StringProperty(
        name="Reports folder", default="//PERM_results/", subtype='DIR_PATH',
        description="Folder for the results reports ('//' = next to the .blend file)")
    global _PERM_LEGEND_HANDLE
    if _PERM_LEGEND_HANDLE is None:
        try:
            _PERM_LEGEND_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _perm_color_legend_draw, (), 'WINDOW', 'POST_PIXEL')
        except Exception as e:
            print(f"[PERM] Could not add the colour legend overlay: {e}")
    # Shell offset
    bpy.types.Scene.perm_offset_half = bpy.props.BoolProperty(
        name="Half the section thickness", default=True,
        description="Offset = half the thickness of each face's Cross-Section, so all of "
                    "the face's material lies on one side of the mesh surface")
    bpy.types.Scene.perm_offset_distance = bpy.props.FloatProperty(
        name="Offset", default=0.05, min=0.0, soft_max=0.5, precision=3, unit='LENGTH',
        description="Distance from the mesh surface to the shell mid-surface")
    bpy.types.Scene.perm_offset_direction = bpy.props.EnumProperty(
        name="Side",
        items=[('DOWN', "Down (top surfaces)", "The thickness lies BELOW the mesh surface (the mesh is the top of a slab)"),
               ('UP', "Up (underside surfaces)", "The thickness lies ABOVE the mesh surface (the mesh is the underside of a slab)"),
               ('AGAINST', "Against the face normal", "Behind the face, opposite to its normal"),
               ('ALONG', "Along the face normal", "In front of the face, along its normal")],
        default='DOWN')
    # Interactive results
    bpy.types.Scene.perm_inspect_running = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.perm_show_reaction_labels = bpy.props.BoolProperty(
        name="Reaction labels", default=True,
        description="Show the reaction values next to the support arrows")
    bpy.types.Scene.perm_reaction_scale = bpy.props.FloatProperty(
        name="Reaction arrow size", default=1.0, min=0.1, max=10.0,
        description="Size of the support-reaction arrows (relative to the model)",
        update=_update_reaction_scale)
    global _PERM_ARROW_HANDLE
    if _PERM_ARROW_HANDLE is None:
        try:
            _PERM_ARROW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _perm_reaction_arrows_draw, (), 'WINDOW', 'POST_VIEW')
        except Exception as e:
            print(f"[PERM] Could not add reaction arrow overlay: {e}")
    if _PERM_LABEL_HANDLE is None:
        try:
            _PERM_LABEL_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _perm_reaction_labels_draw, (), 'WINDOW', 'POST_PIXEL')
        except Exception as e:
            print(f"[PERM] Could not add reaction label overlay: {e}")
    # Colour filter and range
    bpy.types.Scene.perm_viz_filter = bpy.props.EnumProperty(
        name="Show",
        items=[('ALL', "All", "Every face on one colour scale (an overview: where a model mixes steel and concrete, "
                              "the steel stresses set the scale)"),
               ('CONCRETE', "E < 100 GPa", "Only faces of a material with E < 100 GPa and mass (concrete, timber, "
                                           "masonry, aluminium ...); the rest grey"),
               ('STEEL', "E ≥ 100 GPa", "Only faces of a material with E >= 100 GPa (steel ...); the rest grey")],
        default='ALL', update=_perm_update_colors)
    bpy.types.Scene.perm_viz_links = bpy.props.BoolProperty(
        name="With massless faces (modelling links)", default=False,
        update=_perm_update_colors,
        description="Also paint the faces of a material with density 0 (modelling links) in the 'E < 100 GPa' view, "
                    "on its colour scale. Their values do not set the range and do not change the other colours")
    bpy.types.Scene.perm_color_range = bpy.props.EnumProperty(
        name="Colour range",
        items=[('AUTO', "Auto", "From the faces shown, the top and bottom few % saturated"),
               ('MANUAL', "Manual", "Min / max typed below, kept per map and per filter")],
        default='AUTO', update=_perm_update_colors)
    bpy.types.Scene.perm_clip_percent = bpy.props.FloatProperty(
        name="Saturate top/bottom %", default=2.0, min=0.0, max=20.0, precision=1,
        update=_perm_update_colors,
        description="Share of the highest and of the lowest node values drawn in the end colours "
                    "(0 = colours span the full min-max)")
    bpy.types.Scene.perm_range_min = bpy.props.FloatProperty(
        name="Min", precision=3, get=_perm_range_get(0), set=_perm_range_set(0),
        description="Value at the bottom of the colour bar (MPa; kN·m/m for moments)")
    bpy.types.Scene.perm_range_max = bpy.props.FloatProperty(
        name="Max", precision=3, get=_perm_range_get(1), set=_perm_range_set(1),
        description="Value at the top of the colour bar (MPa; kN·m/m for moments)")
    # Find element
    bpy.types.Scene.perm_find_element = bpy.props.IntProperty(
        name="Element", default=1, min=1,
        description="Element number, as printed in the console (Top 10 list) and in the results report")
    bpy.types.Scene.perm_find_show = bpy.props.EnumProperty(
        name="Show on",
        items=[('RESULT', "Result mesh", "Mark it on the visible result mesh (colours stay on)"),
               ('MODEL', "Model", "Select its face in the model, in Edit Mode")],
        default='RESULT')
    bpy.types.Scene.perm_find_distance = bpy.props.FloatProperty(
        name="View distance", default=0.75, min=0.02, soft_max=10.0, precision=2, unit='LENGTH',
        description="How far the view sits from the element after zooming")
    _perm_find_handlers_add()
    # Per-node editor
    bpy.types.Scene.perm_per_node_list = bpy.props.CollectionProperty(type=PERM_PerNodeForce)
    bpy.types.Scene.perm_per_node_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.perm_per_node_group = bpy.props.StringProperty(default="")
    # Materials and sections
    bpy.types.Scene.perm_materials = bpy.props.CollectionProperty(type=PERM_MaterialPG)
    bpy.types.Scene.perm_material_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.perm_sections = bpy.props.CollectionProperty(type=PERM_SectionPG)
    bpy.types.Scene.perm_section_index = bpy.props.IntProperty(default=0)
    # Connections: per object, next to its vertex groups
    bpy.types.Object.perm_connections = bpy.props.CollectionProperty(type=PERM_ConnectionPG)
    bpy.types.Object.perm_connection_index = bpy.props.IntProperty(default=0)
    # Element Types
    bpy.types.Scene.perm_element_types = bpy.props.CollectionProperty(type=PERM_ElementTypePG)
    bpy.types.Scene.perm_element_type_index = bpy.props.IntProperty(default=0)
    # Self-weight (gravity)
    bpy.types.Scene.perm_apply_gravity = bpy.props.BoolProperty(
        name="Apply self-weight", default=False,
        description="Apply gravitational body force to all elements at solve time")
    bpy.types.Scene.perm_gravity_z = bpy.props.FloatProperty(
        name="gz (m/s²)", default=-9.81,
        description="Gravity acceleration in Z direction (negative = downward)")
    bpy.types.Scene.perm_gravity_x = bpy.props.FloatProperty(
        name="gx (m/s²)", default=0.0,
        description="Acceleration in the X direction (0 unless Z is not the vertical of the model, or for a lateral body force)")
    bpy.types.Scene.perm_gravity_y = bpy.props.FloatProperty(
        name="gy (m/s²)", default=0.0,
        description="Acceleration in the Y direction (0 unless Z is not the vertical of the model, or for a lateral body force)")
    _perm_register_surface_load_props()
    # Mesh topology target — lives in the Element Types panel, read by the solver
    # operator at invoke(). Decides quadrangulate vs triangulate before the FE
    # model is built; per-face element types are assigned separately.
    bpy.types.Scene.perm_mesh_topology = bpy.props.EnumProperty(
        name="Mesh topology",
        items=[
            ('QUAD4', "Quads (recommended)",
             "Quads stay quads, triangle pairs are joined into quads where their shape allows, the rest stay "
             "triangles. Faces with more than four vertices give no element: triangulate them first"),
            ('TRI3', "Triangles",
             "Every face is split into triangles (CST + MITC3 shell), faces with more than four vertices included"),
        ],
        default='QUAD4',
        description="Target topology the mesh is converted to before the FE model "
                    "is built")
    # Marker scale + deformation scale
    bpy.types.Scene.perm_marker_scale = bpy.props.FloatProperty(
        name="Marker Scale", default=1.0, min=0.05, max=20.0,
        description="Visual scale multiplier for BC and load markers")
    bpy.types.Scene.perm_deformation_scale = bpy.props.FloatProperty(
        name="Deformation Scale", default=100.0, min=0.0,
        soft_min=0.0, soft_max=1000.0,
        description="Live multiplier for the deformed-shape visualization. "
                    "Drag to rescale the existing deformed mesh in place — "
                    "no re-solve, no new mesh",
        update=_update_deformation_scale)
    # Result overlay opacity. 1.0 = pure colour map (unlit); 0.0 = the object's own
    # textured material with no overlay at all.
    bpy.types.Scene.perm_result_opacity = bpy.props.FloatProperty(
        name="Result Opacity", default=1.0, min=0.0, max=1.0,
        subtype='FACTOR',
        description="Blend between the result colour map and the mesh's own "
                    "texture. 1.0 = colour map only; lower values let the "
                    "texture show through. Needs Material Preview viewport "
                    "shading",
        update=_update_result_opacity)
    # Expanded-state tracking for node lists (comma-separated group names)
    bpy.types.Scene.perm_expanded_groups = bpy.props.StringProperty(
        name="Expanded Groups", default="",
        description="Internal: tracks which group rows have their node list expanded")
    _perm_register_detail_options()
    print(f"[PERM] {PERM_VERSION} registered (panel: View3D > Sidebar > PERM)")

def unregister():
    global _PERM_LABEL_HANDLE, _PERM_ARROW_HANDLE, _PERM_LEGEND_HANDLE
    if _PERM_LEGEND_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PERM_LEGEND_HANDLE, 'WINDOW')
        except Exception:
            pass
        _PERM_LEGEND_HANDLE = None
    if _PERM_ARROW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PERM_ARROW_HANDLE, 'WINDOW')
        except Exception:
            pass
        _PERM_ARROW_HANDLE = None
        _PERM_ARROW_CACHE.clear()
    if _PERM_LABEL_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PERM_LABEL_HANDLE, 'WINDOW')
        except Exception:
            pass
        _PERM_LABEL_HANDLE = None
    _perm_find_handlers_remove()
    for _p in ("perm_inspect_running", "perm_show_reaction_labels", "perm_reaction_scale",
               "perm_find_element", "perm_find_show", "perm_find_distance",
               "perm_viz_filter", "perm_viz_links", "perm_color_range", "perm_clip_percent",
               "perm_range_min", "perm_range_max"):
        if hasattr(bpy.types.Scene, _p):
            try:
                delattr(bpy.types.Scene, _p)
            except Exception:
                pass
    for attr in ("perm_per_node_list", "perm_per_node_index", "perm_per_node_group",
                 "perm_materials", "perm_material_index",
                 "perm_sections", "perm_section_index",
                 "perm_element_types", "perm_element_type_index",
                 "perm_apply_gravity", "perm_gravity_z", "perm_gravity_x", "perm_gravity_y", "perm_mesh_topology",
                 "perm_sl_mode", "perm_sl_vector", "perm_sl_pressure", "perm_sl_add",
                 "perm_marker_scale", "perm_deformation_scale",
                 "perm_result_opacity", "perm_expanded_groups",
                 "perm_offset_half", "perm_offset_distance", "perm_offset_direction",
                 "perm_principal_colormap", "perm_show_color_legend", "perm_case_name",
                 "perm_results_dir"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for attr in ("perm_connections", "perm_connection_index"):
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    _perm_unregister_detail_options()
    for cls in reversed(classes): bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
