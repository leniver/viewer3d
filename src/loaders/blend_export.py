"""viewer3d: exports the .blend file Blender has just opened to a GLB file.

Run by the viewer as:
    blender --background --factory-startup --disable-autoexec FILE.blend
            --python-exit-code 3 --python blend_export.py -- OUT.glb

The scene is exported as Blender shows it in the viewport: visible objects of
the active scene, modifiers applied; only render backdrops (huge flat planes)
are left out. Animations are grouped so that what plays
together in Blender plays together in the viewer:

- the active actions of all objects (what the Play button shows) become one
  clip; with several objects it is named after the words their action names
  share ("dome_running" + "capsules_running" -> "running"), else after the scene;
- named NLA tracks become one clip per track name (Blender's own grouping);
- actions assigned to nothing (kept with a fake user) are given back to the
  object they were made for; actions of different objects that end with the
  same words are grouped the same way ("dome_jammed" + "capsules_jammed").

Nothing is saved: the changes made here only live in this Blender process.
Lines starting with "VIEWER3D:" are read by the viewer.
"""

import re
import sys
import traceback

import bpy
from mathutils import Vector

SEPARATORS = "_-. "
SLOTTED = hasattr(bpy.types, "ActionSlot")  # Blender 4.4+ (actions with slots)


def say(tag, text=""):
    print(f"VIEWER3D:{tag}:{text}", flush=True)


def common_suffix(names):
    """Longest run of trailing words shared by all names, each name having at
    least one word before it: ("dome_running", "capsules_running") -> "running"."""
    if len(names) < 2:
        return None
    shortest = min(names, key=len)
    for i in range(len(shortest)):
        suffix = shortest[i:]
        if suffix[0] in SEPARATORS:
            continue
        if all(n.endswith(suffix) and len(n) > len(suffix) and n[-len(suffix) - 1] in SEPARATORS for n in names):
            return suffix
    return None


def last_word(name):
    words = [w for w in re.split(r"[_\-. ]+", name) if w]
    return words[-1] if words else name


GEOMETRY = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


def world_box(ob, depsgraph):
    """(min, max) corners of an object's evaluated bounds in world space."""
    ev = ob.evaluated_get(depsgraph)
    pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    return (Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)]))


def drop_backdrops(scene):
    """Hides render backdrops: flat meshes over ten times larger than all the
    rest (a 1200 m floor under a 2 m model). Framed with the model, they would
    shrink it to a dot; the viewer's grid shows the ground instead."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    shown = [ob for ob in scene.objects if ob.type in GEOMETRY and ob.visible_get()]
    boxes = {ob.name: world_box(ob, depsgraph) for ob in shown}
    for ob in shown:
        if ob.type != "MESH":
            continue
        lo, hi = boxes[ob.name]
        size = hi - lo
        if max(size) <= 0 or min(size) > 1e-3 * max(size):
            continue  # not flat
        others = [boxes[o.name] for o in shown if o is not ob]
        if not others:
            continue
        rest_lo = Vector([min(b[0][i] for b in others) for i in range(3)])
        rest_hi = Vector([max(b[1][i] for b in others) for i in range(3)])
        rest = max(rest_hi - rest_lo)
        if rest > 0 and max(size) > 10 * rest:
            ob.hide_set(True)
            dims = sorted(size, reverse=True)
            say("NOTE", f"the plane '{ob.name}' ({dims[0]:g} x {dims[1]:g} m) was left out as a backdrop")


def exported_owners(scene):
    """IDs that can carry exported animation: visible objects and their shape keys."""
    owners = []
    for ob in scene.objects:
        if not ob.visible_get():
            continue
        owners.append(ob)
        keys = getattr(ob.data, "shape_keys", None) if ob.data is not None else None
        if keys is not None:
            owners.append(keys)
    return owners


def add_track(owner, name, action, slot):
    """Puts `action` on a new muted NLA track called `name`. Muting the track
    keeps Blender's own evaluation (and so the exported rest pose) unchanged;
    the glTF exporter still exports single-strip tracks and merges clips by
    track name."""
    ad = owner.animation_data or owner.animation_data_create()
    track = ad.nla_tracks.new()
    try:
        track.name = name
        track.mute = True
        strip = track.strips.new(name, int(action.frame_range[0]), action)
        if SLOTTED:
            strip.action_slot = slot
            if strip.action_slot is None:
                raise RuntimeError("no slot")
        return True
    except Exception as e:  # incompatible slot, locked track...
        ad.nla_tracks.remove(track)
        say("NOTE", f"animation '{action.name}' left out: {e}")
        return False


def unique(name, taken):
    base, n = name, 2
    while name in taken:
        name = f"{base} {n}"
        n += 1
    taken.add(name)
    return name


def group_animations(scene):
    """Prepares NLA tracks for the exporter; returns the name of the clip of the
    active actions (played first in the viewer) or None."""
    owners = exported_owners(scene)
    taken = set()
    for owner in owners:
        ad = owner.animation_data
        if ad is not None:
            taken.update(t.name for t in ad.nla_tracks)

    # Active actions: one clip, as Blender plays them together.
    active = []
    for owner in owners:
        ad = owner.animation_data
        if ad is None or ad.action is None:
            continue
        slot = getattr(ad, "action_slot", None)
        if SLOTTED and slot is None:
            continue
        active.append((owner, ad.action, slot))
    timeline = None
    if len(active) == 1:
        timeline = active[0][1].name
    elif len(active) > 1:
        names = sorted({a.name for _, a, _ in active})
        name = names[0] if len(names) == 1 else (common_suffix(names) or scene.name or "Scene")
        timeline = unique(name, taken)
        for owner, action, slot in active:
            add_track(owner, timeline, action, slot)

    if not SLOTTED:
        return timeline  # older Blender: the exporter's defaults only

    # Actions assigned to nothing: find the IDs their slots were made for.
    objects = {ob.name: ob for ob in owners if isinstance(ob, bpy.types.Object)}
    keys = {k.name: k for k in owners if isinstance(k, bpy.types.Key)}
    loose = []  # (action, [(owner, slot)])
    for action in bpy.data.actions:
        if action.users - int(action.use_fake_user) > 0:
            continue
        targets = []
        for slot in action.slots:
            table = objects if slot.target_id_type == "OBJECT" else keys if slot.target_id_type == "KEY" else {}
            owner = table.get(slot.name_display)
            if owner is not None:
                targets.append((owner, slot))
        if targets:
            loose.append((action, targets))

    # Group single-target actions of different IDs that end with the same word.
    buckets = {}
    for action, targets in loose:
        key = last_word(action.name) if len(targets) == 1 else None
        buckets.setdefault(key, []).append((action, targets))
    for key, items in buckets.items():
        owners_used = [targets[0][0] for _, targets in items]
        group = None
        if key is not None and len(items) > 1 and len(set(map(id, owners_used))) == len(items):
            group = common_suffix([a.name for a, _ in items])
        if group is not None:
            name = unique(group, taken)
            for action, targets in items:
                add_track(targets[0][0], name, action, targets[0][1])
        else:
            for action, targets in items:
                name = unique(action.name, taken)
                for owner, slot in targets:
                    add_track(owner, name, action, slot)
    return timeline


def export_options(out):
    wanted = dict(
        filepath=out,
        export_format="GLB",
        use_visible=True,
        use_active_scene=True,
        export_apply=True,
        export_yup=True,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
        export_tangents=False,
        export_draco_mesh_compression_enable=False,
        export_image_format="AUTO",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_merge_animation="NLA_TRACK",
        export_anim_slide_to_zero=True,
        export_force_sampling=True,
        will_save_settings=False,
    )
    known = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    return {k: v for k, v in wanted.items() if k in known}


def main():
    out = sys.argv[sys.argv.index("--") + 1]
    say("VERSION", bpy.app.version_string)
    if not bpy.data.filepath:
        # Blender could not read the file and kept its default scene.
        say("ERROR", "Blender could not open the file")
        sys.exit(3)
    if tuple(bpy.data.version) > tuple(bpy.app.version_file):
        say("NEWER", "%d.%d" % tuple(bpy.data.version)[:2])
    scene = bpy.context.scene
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")  # flushes edit-mode changes to the mesh
        except Exception:
            pass
    if not hasattr(bpy.ops.export_scene, "gltf"):
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        except Exception:
            pass
    say("STAGE", "Grouping animations")
    drop_backdrops(scene)
    timeline = group_animations(scene)
    if timeline:
        say("TIMELINE", timeline)
    say("STAGE", "Exporting")
    bpy.ops.export_scene.gltf(**export_options(out))
    say("DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    say("ERROR", traceback.format_exc().strip().splitlines()[-1])
    sys.exit(3)
