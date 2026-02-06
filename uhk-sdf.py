#!/usr/bin/env python3
import sys
import svgwrite
# import ezdxf
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
from geometry.spline import slope_controlled_bezier_curve

import sdf

in2mm = 25.4

from ipdb import iex, set_trace as db

# very approximate
# perimeter corner radius ~2
perimeter_l = np.array([
  [0, 0],
  [-135, 0],
  [-135, 114],
  [-108, 114],
  [-104, 132],
  [0, 114],
  [0, 0],
])

perimeter_r = np.array([
  [0, 0],
  [154, 0],
  [154, 114],
  [154-33, 114],
  [154-33-4, 132],
  [0, 114],
  [0, 0],
])

C = np.exp(1j*np.linspace(0, 2*np.pi, 128+1))

@iex
def main():
    tripod_mount()
    # sdf_test()
    # spline_demo()
    # steiner_tree_demo()


def sdf_test():
    f1 = sphere()
    f2 = capsule((0, 0, 0), (5, 5, 5), 0.5)
    f = f1 | f2.k(0.25)

    pass

def steiner_tree_demo():
    points = np.array([[0, 0], [2, 3], [4, 1], [5, 5], [7, 2]])
    mst = minimum_spanning_tree(points)
    new_points = add_steiner_points(points, mst)
    plt.figure()
    plot_graph(new_points, mst, steiner_points=new_points[len(points):])
    plt.gca().set_aspect('equal')
    plt.title("Approximate Steiner Tree")
    plt.show()

def normalize(v, axis=0):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)

def tripod_mount():
    # define hole locations
    left_screw_holes = np.array([
        [-(7.5+96),  7.4], # bottom-right (viewed from above)
        [-(7.5),     7.4+23.5],  # bottom-left
        [-(7.5),     7.4+23.5+55],  # top-left
        [-(7.5+94),  7.4+23.5+55+22.5],  # top-right
    ])
    right_screw_holes = np.array([
        [(7.6),      7.4+23.5],  # bottom-right (viewed from above)
        [(7.6+115),  7.4], # bottom-left
        [(7.6+103),  7.4+23.5+59+18.5], # top-left
        [(7.6),   7.4+23.5+59],  # top-right
    ])
    left_foot_holes = left_screw_holes + [
        [-21, 2],
        [-2, -21.5],
        [-2, 18.5+.5],
        [-23, -4.5+.5],        
    ]
    right_foot_holes = right_screw_holes + [
        [2, -21.5],
        [16.5, 2],
        [29, -4],
        [2.5, 14.5],
    ]

    
    left_centroid = np.array([-(3+5/16)*in2mm, (2+1/4)*in2mm])
    right_centroid = np.array([(3+3/8)*in2mm, (2+1/4)*in2mm])
    reset_switch = np.array([8.5, 20])

    centroid_diam = 6.5
    centroid_tap_diam = 5.31
    screw_hole_diam = 5
    foot_hole_diam = 6
    reset_diam = 3
        
    # plot diagrams
    plt.figure()

    PLOT_LEFT = True
    PLOT_LEFT_SDF = True
    PLOT_LEFT_SPLINE = False
    PLOT_RIGHT = True
    PLOT_RIGHT_SDF = True
    R1 = 20
    R2 = 7.5
    kvec = [16]


    if PLOT_LEFT:
        for n, c in enumerate(left_screw_holes):
            pth = complex(*c) + screw_hole_diam/2 * C
            plt.plot(pth.real, pth.imag, 'r', label='screw hole' if n==0 else None)
            x, y = [left_centroid[0], c[0]], [left_centroid[1], c[1]]
            # plt.plot(x, y, 'k', label='shim skeleton' if n==0 else None)
        for n, c in enumerate(left_foot_holes):
            pth = complex(*c) + foot_hole_diam/2 * C
            plt.plot(pth.real, pth.imag, 'g', label='foot hole' if n==0 else None)
        plt.plot(perimeter_l[:,0], perimeter_l[:,1], 'm', label='perimeter')
        pth = complex(*left_centroid) + centroid_diam/2 * C
        plt.plot(pth.real, pth.imag, 'b', label='centroid mount')

    if PLOT_LEFT_SDF:
        # compute shape via sdf
        f1 = sdf.circle(R1).translate(left_centroid)
        f2 = sdf.line_segment(left_centroid, left_screw_holes[0,:]).dilate(R2)
        f3 = sdf.line_segment(left_centroid, left_screw_holes[1,:]).dilate(R2)
        f4 = sdf.line_segment(left_centroid, left_screw_holes[2,:]).dilate(R2)
        f5 = sdf.line_segment(left_centroid, left_screw_holes[3,:]).dilate(R2)
        for k in kvec:
            f = f1 | f2.k(k) | f3.k(k) | f4.k(k) | f5.k(k)
            outline = f.contour()
            circles = list(zip(left_screw_holes.tolist(), [screw_hole_diam/2]*4))
            circles.append((left_centroid, centroid_tap_diam/2))
            sdf.write_svg("uhk-sdf-left.svg", paths=outline, circles=circles)
            sdf.write_dxf("uhk-sdf-left.dxf", paths=outline, circles=circles)
            for p in outline:
                plt.plot(p[:,0], p[:,1], label=f'SDF cutout, k={k}')

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('UHK left side (from below)')

    # plt.subplot(121)
    if PLOT_RIGHT:
        for c in right_screw_holes:
            pth = complex(*c) + screw_hole_diam/2 * C
            plt.plot(pth.real, pth.imag, 'r')
        for c in right_foot_holes:
            pth = complex(*c) + foot_hole_diam/2 * C
            plt.plot(pth.real, pth.imag, 'g')
        plt.plot(perimeter_r[:,0], perimeter_r[:,1], 'm')

        pth = complex(*right_centroid) + centroid_diam/2 * C
        plt.plot(pth.real, pth.imag, 'b')

        pth = complex(*reset_switch) + reset_diam/2 * C
        plt.plot(pth.real, pth.imag, 'k')

        plt.title('right side (from below)')

    plt.gca().set_aspect('equal')

    if PLOT_RIGHT_SDF:
        # compute shape via sdf
        f1 = sdf.circle(R1).translate(right_centroid)
        f2 = sdf.line_segment(right_centroid, right_screw_holes[0,:]).dilate(R2)
        f3 = sdf.line_segment(right_centroid, right_screw_holes[1,:]).dilate(R2)
        f4 = sdf.line_segment(right_centroid, right_screw_holes[2,:]).dilate(R2)
        f5 = sdf.line_segment(right_centroid, right_screw_holes[3,:]).dilate(R2)
        for k in kvec:
            f = f1 | f2.k(k) | f3.k(k) | f4.k(k) | f5.k(k)
            outline = f.contour()
            circles = list(zip(right_screw_holes.tolist(), [screw_hole_diam/2]*4))
            circles.append((right_centroid, centroid_tap_diam/2))
            sdf.write_svg("uhk-sdf-right.svg", paths=outline, circles=circles)
            sdf.write_dxf("uhk-sdf-right.dxf", paths=outline, circles=circles)
            for p in outline:
                plt.plot(p[:,0], p[:,1])

    if PLOT_LEFT_SPLINE:
        # generate points at radii around screw holes
        L = left_screw_holes
        Lc = left_centroid
        d = L - Lc
        r1 = 7.5
        dn0 = normalize(d, axis=1)
        dn1 = dn0 @ np.array([[0, 1], [-1, 0]])
        dn2 = dn0 @ np.array([[0, -1], [1, 0]])

        # generate points evenly spaced between the "legs"
        p01 = normalize((dn0[0,:] + dn0[1,:])/2)
        p12 = normalize((dn0[1,:] + dn0[2,:])/2)
        p23 = normalize((dn0[2,:] + dn0[3,:])/2)
        p30 = normalize((dn0[3,:] + dn0[0,:])/2)
        # db()
        knots = np.array([
            [Lc[0]+15*p30[0], Lc[1]+15*p30[1], 0, 0],
            [L[0,0]+r1*dn2[0,0], L[0,1]+r1*dn2[0,1], 0, 0],
            [L[0,0]+r1*dn0[0,0], L[0,1]+r1*dn0[0,1], 0, 0],
            [L[0,0]+r1*dn1[0,0], L[0,1]+r1*dn1[0,1], 0, 0],
            [Lc[0]+15*p01[0], Lc[1]+15*p01[1], 0, 0],
            [L[1,0]+r1*dn2[1,0], L[1,1]+r1*dn2[1,1], 0, 0],
            [L[1,0]+r1*dn0[1,0], L[1,1]+r1*dn0[1,1], 0, 0],
            [L[1,0]+r1*dn1[1,0], L[1,1]+r1*dn1[1,1], 0, 0],
            [Lc[0]+15*p12[0], Lc[1]+15*p12[1], 0, 0],
            [L[2,0]+r1*dn2[2,0], L[2,1]+r1*dn2[2,1], 0, 0],
            [L[2,0]+r1*dn0[2,0], L[2,1]+r1*dn0[2,1], 0, 0],
            [L[2,0]+r1*dn1[2,0], L[2,1]+r1*dn1[2,1], 0, 0],
            [Lc[0]+15*p23[0], Lc[1]+15*p23[1], 0, 0],
            [L[3,0]+r1*dn2[3,0], L[3,1]+r1*dn2[3,1], 0, 0],
            [L[3,0]+r1*dn0[3,0], L[3,1]+r1*dn0[3,1], 0, 0],
            [L[3,0]+r1*dn1[3,0], L[3,1]+r1*dn1[3,1], 0, 0],
            [Lc[0]+15*p30[0], Lc[1]+15*p30[1], 0, 0],
        ])
        path = slope_controlled_bezier_curve(knots)
        plt.plot(path[:,0], path[:,1], 'k-', label='spline cutout')
        ak = {'head_width': 0.025, 'head_length': 0.05}
        for (x, y, mx, my) in knots:
            plt.plot(x, y, 'k.')
            plt.arrow(x, y, mx, my, color='k', **ak)

    plt.show()



def minimum_spanning_tree(points):
    """Computes the MST using Kruskal's algorithm."""
    dist_matrix = distance_matrix(points, points)
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            G.add_edge(i, j, weight=dist_matrix[i, j])

    mst = nx.minimum_spanning_tree(G)
    return mst

def fermat_point(triangle):
    """Finds the Fermat-Steiner point for a given triangle (3 points)."""
    def total_distance(S):
        return sum(np.linalg.norm(S - p) for p in triangle)

    centroid = np.mean(triangle, axis=0)
    result = minimize(total_distance, centroid, method='Nelder-Mead')
    return result.x if result.success else centroid

def add_steiner_points(points, mst):
    """Adds Steiner points to the MST where beneficial."""
    new_points = list(points)
    edges = list(mst.edges())

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u, v = edges[i]
            x, y = edges[j]
            if len(set([u, v, x, y])) == 3:  # Ensure three unique points
                triangle = np.array([points[u], points[v], points[x]])
                steiner = fermat_point(triangle)
                new_points.append(steiner)

    return np.array(new_points)

def plot_graph(points, graph, steiner_points=[]):
    """Plots the graph with terminals and Steiner points."""

    for u, v in graph.edges():
        p1, p2 = points[u], points[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2)

    plt.scatter(points[:,0], points[:,1], c='r', s=50, label="Terminals")

    if len(steiner_points) > 0:
        plt.scatter(steiner_points[:,0], steiner_points[:,1], c='g', s=50, label="Steiner Points")

    plt.legend()




def from_file():
    #input_file = 'uhk-hole-template.svg'
    #paths, attributes = svg2paths(input_file)
    #print(attributes)

    input_file = 'uhk-hole-template.dxf'
    # doc = ezdxf.readfile(input_file)

    plt.figure(figsize=(10, 5))
    plot_dxf(input_file)
    #plot_paths(paths, title="hole template", subplot=111)
    plt.show()


def plot_dxf(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()

    fig, ax = plt.subplots()

    for entity in msp:
        if entity.dxftype() == 'LINE':
            x_values = [entity.dxf.start.x, entity.dxf.end.x]
            y_values = [entity.dxf.start.y, entity.dxf.end.y]
            ax.plot(x_values, y_values, 'b')

        elif entity.dxftype() == 'CIRCLE':
            circle = plt.Circle((entity.dxf.center.x, entity.dxf.center.y), 
                                entity.dxf.radius, color='r', fill=False)
            ax.add_patch(circle)

        elif entity.dxftype() == 'ARC':
            from numpy import linspace, cos, sin, radians
            angles = linspace(entity.dxf.start_angle, entity.dxf.end_angle, num=100)
            x = entity.dxf.center.x + entity.dxf.radius * cos(radians(angles))
            y = entity.dxf.center.y + entity.dxf.radius * sin(radians(angles))
            ax.plot(x, y, 'g')

    ax.set_aspect('equal')
    plt.show()


def plot_paths(paths, title="SVG Paths", subplot=111):
    """Plot SVG paths using Matplotlib for debugging."""
    plt.subplot(subplot)
    for path in paths:
        x_vals, y_vals = [], []
        for seg in path:
            x_vals.extend([seg.start.real, seg.end.real])
            y_vals.extend([seg.start.imag, seg.end.imag])

        plt.plot(x_vals, y_vals, marker=".")

    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()  # SVG has (0,0) at the top-left, like images
    plt.title(title)


main()
