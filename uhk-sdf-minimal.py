#!/usr/bin/env python3
import sdf

in2mm = 25.4

# UHK centroids
Lc = [-(3+5/16)*in2mm, (2+1/4)*in2mm]  # left centroid (relative to bottom-left)
Rc = [(3+3/8)*in2mm, (2+1/4)*in2mm]    # right centroid (relative to bottom-right)
# UHK mounting screw hole locations
Ls = [
    [-(7.5+96),  7.4],               # bottom-right (viewed from above)
    [-(7.5),     7.4+23.5],          # bottom-left
    [-(7.5),     7.4+23.5+55],       # top-left
    [-(7.5+94),  7.4+23.5+55+22.5],  # top-right
]
Rs = [
    [(7.6),      7.4+23.5],         # bottom-right (viewed from above)
    [(7.6+115),  7.4],              # bottom-left
    [(7.6+103),  7.4+23.5+59+18.5], # top-left
    [(7.6),   7.4+23.5+59],         # top-right
]

# design parameters
D1 = 5.31  # diameter for 1/4-20 (https://sendcutsend.com/guidelines/tapping/)
D2 = 5     # diameter for UHK mount screws

R1 = 20
R2 = 7.5
k = 16
Nsamples = 2**12

# compute shapes via sdf
flc = sdf.circle(R1).translate(Lc)
fls = [sdf.line_segment(Lc, p).dilate(R2) for p in Ls]
fl = flc | fls[0].k(k) | fls[1].k(k) | fls[2].k(k) | fls[3].k(k)
outline = fl.contour(samples=Nsamples)
circles = list(zip(Ls, [D2/2]*4))
circles.append((Lc, D1/2))
sdf.write_dxf("/home/alan/d/src/scad/svg/uhk-sdf-left-minimal-2e12.dxf", paths=outline, circles=circles)

frc = sdf.circle(R1).translate(Rc)
frs = [sdf.line_segment(Rc, p).dilate(R2) for p in Rs]
fr = frc | frs[0].k(k) | frs[1].k(k) | frs[2].k(k) | frs[3].k(k)
outline = fr.contour(samples=Nsamples)
circles = list(zip(Rs, [D2/2]*4))
circles.append((Rc, D1/2))
sdf.write_dxf("/home/alan/d/src/scad/svg/uhk-sdf-right-minimal-2e12.dxf", paths=outline, circles=circles)
