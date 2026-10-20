"""Mathematical, geometric, coordinate-conversion, and VTK polydata utilities."""

import math
import operator
import re
import numpy as np
import slicer
import vtk


def extract_model_id_number(node_id_or_name):
    """Extract numeric ID from a MRML model node ID or needle name."""
    if isinstance(node_id_or_name, int):
        return node_id_or_name
    text = str(node_id_or_name)
    match = re.search(r"vtkMRMLModelNode(\d+)", text)
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else 0


def factorial(n):
    """Return the factorial of integer `n` (preserving sign for negative `n`)."""
    result = math.factorial(abs(int(n)))
    return result if n >= 0 else -result


def binomial(n, k):
    """Return the binomial coefficient C(n, k)."""
    if not 0 <= k <= n:
        return 0
    return math.comb(int(n), int(k))


def fibonacci(n):
    """Return Fibonacci sequence up to index `n + 1`."""
    seq = [0, 1]
    for i in range(1, int(n) + 1):
        seq.append(seq[i - 1] + seq[i])
    return seq


def step_size(k, l):
    """Compute normalized Fibonacci step size for step `k` out of `l`."""
    fib = fibonacci(l)
    return fib[k + 1] / float(sum(fib))


def step_size_andre(k, l):
    """Compute alternative normalized Fibonacci step size."""
    fib = fibonacci(l)
    return fib[k + 1] / float(sum(fib) - 1)


def step_size_13(k, l):
    """Compute MICCAI 2013 cumulative Fibonacci step size."""
    fib = fibonacci(l + 1)
    return (sum(fibonacci(k + 1), -1) + fib[k + 1]) / float(sum(fib, -1))


def sort_table(table, cols, reverse=False):
    """Sort a list of sequences by multiple column indices."""
    result = list(table)
    for col in reversed(cols):
        result = sorted(result, key=operator.itemgetter(col), reverse=reverse)
    return result


def get_active_background_volume():
    """Return the active background volume node from the Red slice viewer."""
    lm = slicer.app.layoutManager()
    if lm and lm.sliceWidget("Red"):
        return lm.sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    return None


def ijk_to_ras(ijk, volume_node=None):
    """Convert IJK coordinates to RAS coordinates using the volume's IJKToRAS matrix."""
    if volume_node is None:
        volume_node = get_active_background_volume()
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    pt = mat.MultiplyDoublePoint((float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0))
    return [pt[0], pt[1], pt[2]]


def ras_to_ijk(ras, volume_node=None):
    """Convert RAS coordinates to IJK coordinates using the inverted IJKToRAS matrix."""
    if volume_node is None:
        volume_node = get_active_background_volume()
    mat = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(mat)
    mat.Invert()
    pt = mat.MultiplyDoublePoint((float(ras[0]), float(ras[1]), float(ras[2]), 1.0))
    return [pt[0], pt[1], pt[2]]


def distance_3d(pt1, pt2):
    """Return the 3D Euclidean distance between `pt1` and `pt2`."""
    return math.sqrt(
        (float(pt1[0]) - float(pt2[0])) ** 2
        + (float(pt1[1]) - float(pt2[1])) ** 2
        + (float(pt1[2]) - float(pt2[2])) ** 2
    )


def norm(vec):
    """Return the Euclidean norm of vector `vec`."""
    return float(np.sqrt(np.dot(vec, vec)))


def normalized(vec):
    """Return `(unit_vector, length)` for vector `vec`."""
    length = norm(vec)
    if length == 0:
        return np.zeros_like(vec, dtype=float), 0.0
    return np.asarray(vec, dtype=float) / length, length


def angle_between_vectors(v1, v2):
    """Return the angle in radians between 3D vectors `v1` and `v2`."""
    cos_ang = np.dot(v1, v2)
    sin_ang = np.linalg.norm(np.cross(v1, v2))
    return float(np.arctan2(sin_ang, cos_ang))


def unique_points(points_array):
    """Remove duplicate rows in a 2D NumPy array of 3D coordinates."""
    arr = np.asarray(points_array)
    if arr.ndim != 2 or len(arr) <= 1:
        return arr
    order = np.lexsort(arr.T)
    arr = arr[order]
    diff = np.diff(arr, axis=0)
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] = (diff != 0).any(axis=1)
    return arr[mask]


def trilinear_interp(image_data, x, y, z, interpolate=True):
    """Perform trilinear or nearest-neighbor scalar lookup on a `vtkImageData`."""
    if not interpolate:
        return image_data.GetScalarComponentAsDouble(
            int(round(x)), int(round(y)), int(round(z)), 0
        )
    xf, yf, zf = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    xc, yc, zc = int(math.ceil(x)), int(math.ceil(y)), int(math.ceil(z))
    xd, yd, zd = x - xf, y - yf, z - zf
    zdd = 1.0 - zd
    i1 = (
        image_data.GetScalarComponentAsDouble(xf, yf, zf, 0) * zdd
        + image_data.GetScalarComponentAsDouble(xf, yf, zc, 0) * zd
    )
    i2 = (
        image_data.GetScalarComponentAsDouble(xf, yc, zf, 0) * zdd
        + image_data.GetScalarComponentAsDouble(xf, yc, zc, 0) * zd
    )
    j1 = (
        image_data.GetScalarComponentAsDouble(xc, yf, zf, 0) * zdd
        + image_data.GetScalarComponentAsDouble(xc, yf, zc, 0) * zd
    )
    j2 = (
        image_data.GetScalarComponentAsDouble(xc, yc, zf, 0) * zdd
        + image_data.GetScalarComponentAsDouble(xc, yc, zc, 0) * zd
    )
    w1 = i1 * (1.0 - yd) + i2 * yd
    w2 = j1 * (1.0 - yd) + j2 * yd
    return w1 * (1.0 - xd) + w2 * xd


def compute_bezier_curve(control_points, num_eval_points=50):
    """Evaluate a 3D Bezier curve defined by `control_points` at `num_eval_points + 1` points."""
    n = len(control_points) - 1
    curve_points = [[0.0, 0.0, 0.0] for _ in range(num_eval_points + 1)]
    coeffs = [binomial(n, i) for i in range(n + 1)]
    for t_idx in range(num_eval_points + 1):
        tt = float(t_idx) / float(num_eval_points)
        one_minus_tt = 1.0 - tt
        for i in range(n + 1):
            weight = coeffs[i] * (one_minus_tt ** (n - i)) * (tt ** i)
            curve_points[t_idx][0] += weight * control_points[i][0]
            curve_points[t_idx][1] += weight * control_points[i][1]
            curve_points[t_idx][2] += weight * control_points[i][2]
    return curve_points


def build_polyline_polydata(points_list):
    """Build a `vtkPolyData` containing a single polyline through `points_list` (VTK 9 compatible)."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    num_pts = len(points_list)
    lines.InsertNextCell(num_pts)
    for pt in points_list:
        pt_idx = points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        lines.InsertCellPoint(pt_idx)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    return poly_data


def build_tube_polydata(poly_data, radius=1.0, num_sides=50):
    """Generate a tubular surface `vtkPolyData` around an input polyline `vtkPolyData`."""
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly_data)
    tube.SetRadius(float(radius))
    tube.SetNumberOfSides(int(num_sides))
    tube.Update()
    return tube.GetOutput()


def build_cspline_polydata(control_points, num_output_points=50):
    """Build a cardinal spline polyline `vtkPolyData` interpolating `control_points`."""
    spline_x = vtk.vtkCardinalSpline()
    spline_y = vtk.vtkCardinalSpline()
    spline_z = vtk.vtkCardinalSpline()
    num_input = len(control_points)
    for i, pt in enumerate(control_points):
        spline_x.AddPoint(i, float(pt[0]))
        spline_y.AddPoint(i, float(pt[1]))
        spline_z.AddPoint(i, float(pt[2]))

    sampled = []
    denom = float(max(1, num_output_points - 1))
    scale = float(max(0, num_input - 1))
    for i in range(num_output_points):
        t = (scale / denom) * i
        sampled.append([spline_x.Evaluate(t), spline_y.Evaluate(t), spline_z.Evaluate(t)])
    return build_polyline_polydata(sampled)


def extract_centerline(polydata, num_points=50, num_sides=50):
    """Extract `num_points` centerline coordinates from a tubular needle `vtkPolyData`."""
    centerline = []
    half_ring = num_sides // 2
    total_pts = polydata.GetNumberOfPoints()
    for i in range(num_points):
        idx1 = min(i * num_sides, total_pts - 1)
        idx2 = min(i * num_sides + half_ring, total_pts - 1)
        p1 = polydata.GetPoint(idx1)
        p2 = polydata.GetPoint(idx2)
        centerline.append((np.array(p1) + np.array(p2)) / 2.0)
    return centerline


def compute_tip_distance(polydata1, polydata2):
    """Compute the minimum Euclidean distance (mm) between the tips of two needle models."""
    nb1 = polydata1.GetNumberOfPoints()
    nb2 = polydata2.GetNumberOfPoints()
    if nb1 == 0 or nb2 == 0:
        return float("inf")
    distances = []
    p, pbis = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    p2, p2bis = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    limit = min(100, nb1, nb2)
    for i in range(limit):
        polydata1.GetPoint(i, p)
        polydata1.GetPoint(nb1 - 1 - i, pbis)
        pt1 = pbis if pbis[2] > p[2] else p

        polydata2.GetPoint(i, p2)
        polydata2.GetPoint(nb2 - 1 - i, p2bis)
        pt2 = p2bis if p2bis[2] > p2[2] else p2

        distances.append(distance_3d(pt1, pt2))
    return min(distances)


def compute_base_distance(polydata1, polydata2):
    """Compute the minimum Euclidean distance (mm) between the bases of two needle models."""
    nb1 = polydata1.GetNumberOfPoints()
    nb2 = polydata2.GetNumberOfPoints()
    if nb1 == 0 or nb2 == 0:
        return float("inf")
    distances = []
    p, pbis = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    p2, p2bis = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    limit = min(100, nb1, nb2)
    for i in range(limit):
        polydata1.GetPoint(i, p)
        polydata1.GetPoint(nb1 - 1 - i, pbis)
        pt1 = p if pbis[2] > p[2] else pbis

        polydata2.GetPoint(i, p2)
        polydata2.GetPoint(nb2 - 1 - i, p2bis)
        pt2 = p2 if p2bis[2] > p2[2] else p2bis

        distances.append(distance_3d(pt1, pt2))
    return min(distances)


def clip_polydata_axial(polydata, z_value):
    """Clip `polydata` above axial Z coordinate `z_value`."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(0.0, 0.0, 1.0)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    clipper.GenerateClipScalarsOn()
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(float(z_value))
    clipper.Update()
    return clipper.GetOutput()


def compute_hausdorff_distance(
    polydata1, polydata2, truncate=True, step1=10, step2=10
):
    """Compute the bidirectional axial-constrained Hausdorff distance (mm) between two needle models."""
    nb1 = polydata1.GetNumberOfPoints()
    nb2 = polydata2.GetNumberOfPoints()
    if nb1 < 2 or nb2 < 2:
        return 0.0

    if truncate:
        pt1, pt2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        polydata1.GetPoint(1, pt1)
        polydata1.GetPoint(nb1 - 1, pt2)
        min_val1 = min(pt1[2], pt2[2])

        polydata2.GetPoint(1, pt1)
        polydata2.GetPoint(nb2 - 1, pt2)
        min_val2 = min(pt1[2], pt2[2])
        value_base = max(min_val1, min_val2)

        target_pd1 = clip_polydata_axial(polydata1, value_base)
        target_pd2 = clip_polydata_axial(polydata2, value_base)
        if target_pd1.GetNumberOfCells() == 0:
            target_pd1 = polydata1
        if target_pd2.GetNumberOfCells() == 0:
            target_pd2 = polydata2
    else:
        target_pd1 = polydata1
        target_pd2 = polydata2

    cell_id = vtk.mutable(1)
    sub_id = vtk.mutable(1)
    dist = vtk.mutable(1.0)

    cl2 = vtk.vtkCellLocator()
    cl2.SetDataSet(target_pd2)
    cl2.BuildLocator()
    minima_12 = [0.0]
    for i in range(max(1, int(nb1 / float(step1)))):
        pt = [0.0, 0.0, 0.0]
        polydata1.GetPoint(min(step1 * i, nb1 - 1), pt)
        closest = [0.0, 0.0, 0.0]
        cl2.FindClosestPoint(pt, closest, cell_id, sub_id, dist)
        if abs(closest[2] - pt[2]) <= 1.0:
            minima_12.append(distance_3d(pt, closest))
    hd12 = max(minima_12)

    cl1 = vtk.vtkCellLocator()
    cl1.SetDataSet(target_pd1)
    cl1.BuildLocator()
    minima_21 = [0.0]
    for i in range(max(1, int(nb2 / float(step2)))):
        pt = [0.0, 0.0, 0.0]
        polydata2.GetPoint(min(step2 * i, nb2 - 1), pt)
        closest = [0.0, 0.0, 0.0]
        cl1.FindClosestPoint(pt, closest, cell_id, sub_id, dist)
        if abs(closest[2] - pt[2]) <= 1.0:
            minima_21.append(distance_3d(pt, closest))
    hd21 = max(minima_21)

    return max(hd12, hd21)
