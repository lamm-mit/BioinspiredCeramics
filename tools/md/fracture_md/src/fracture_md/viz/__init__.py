"""Animations and pictures of harness runs for the README (matplotlib + Pillow, imported lazily).

* ``indentviz``: indentation trajectory (``indentviz-traj/1``) -> three-panel GIF + last-frame PNG.

    from fracture_md.io.indent_traj import Trajectory
    from fracture_md.viz.indentviz import RenderOptions, render_gif
    info = render_gif(Trajectory.load("traj.npz"), "indentation.gif", RenderOptions(field="shear"))
"""

from fracture_md.viz.indentviz import IndentRenderer, RenderOptions, render_gif, select_frames

__all__ = ["IndentRenderer", "RenderOptions", "render_gif", "select_frames"]
