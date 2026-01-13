import os
import numpy as np
from sdf.mesh import Mesh

def generate_shell(stl_fin, stl_fout, surface_offset, thickness,
                   voxel_size, half_width, sdf_step):
    """
    Creates a shell of constant thickness, and constant distance from surface,
    based on a mesh read from an STL file.
    Parameters:
    surface_offset: distance from surface of interior of shell
    thickness: thickness of shell
    voxel_size: edge length of voxel, in same units as the mesh (passed to vdb.createLinearTransform)
    half_width: number of voxels stored on each side of the surface(?) (vdb.FloatGrid.createLevelSetFromPolygons)
    sdf_step: (sdf.save)
    
    Voxel size should be <= thickness/3
    """
    shape = Mesh.from_file(stl_fin)
    f = shape.sdf(voxel_size=voxel_size, half_width=half_width)
    outer = f.dilate(thickness+surface_offset)
    inner = f.dilate(surface_offset)
    shell = outer - inner
    shell.save(stl_fout, step=sdf_step)
    print(f"saved to {stl_fout}")

    
pth = os.getenv("SYNC") + "/m2/3dprint/pro-controller/"
# High-quality mesh from https://sketchfab.com/3d-models/switch-pro-controller-5e09103601b04f469ca6dbe5cfde00d9
# Units are inches. Decimated and exported to STL in Blender.
stl_fin = pth + "pro-controller-decimated0.1.stl"
stl_fout = pth + "procon-shell0.1.stl"

generate_shell(stl_fin, stl_fout, 0.5/25.4, 3.0/25.4, 0.1, 10, 0.1)
