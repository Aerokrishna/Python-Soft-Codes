"""

Path Planner with B-Spline

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# from utils.plot import plot_curvature


def approximate_b_spline_path(x: list,
                              y: list,
                              n_path_points: int,
                              degree: int = 2,
                              s=None,
                              ) -> tuple:
    """
    Approximate points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x position list of approximated points
    y : array_like
        y position list of approximated points
    n_path_points : int
        number of path points
    degree : int, optional
        B Spline curve degree. Must be 2<= k <= 5. Default: 3.
    s : int, optional
        smoothing parameter. If this value is bigger, the path will be
        smoother, but it will be less accurate. If this value is smaller,
        the path will be more accurate, but it will be less smooth.
        When `s` is 0, it is equivalent to the interpolation. Default is None,
        in this case `s` will be `len(x)`.

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    # distances is the indpendent variable in the spline generation
    # k is the degree of spline cureve
    # s is the smoothness of the curve
    distances = _calc_distance_vector(x, y)

    spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
    spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)

    sampled = np.linspace(0.0, distances[-1], n_path_points)
    return _evaluate_spline(sampled, spl_i_x, spl_i_y)


def _calc_distance_vector(x, y):
    dx, dy = np.diff(x), np.diff(y)
    
    distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)]) # cumsum is sum [1,2,3,4] = [1,3,6,10]
    distances = np.concatenate(([0.0], distances)) # adds [0.0] to the array
    distances /= distances[-1] # divide all elements with last element
    return distances


def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)

    # dx = spl_i_x.derivative(1)(sampled)
    # dy = spl_i_y.derivative(1)(sampled)
    # heading = np.arctan2(dy, dx)
    # ddx = spl_i_x.derivative(2)(sampled)
    # ddy = spl_i_y.derivative(2)(sampled)
    # curvature = (ddy * dx - ddx * dy) / np.power(dx * dx + dy * dy, 2.0 / 3.0)
    return np.array(x), y


def main():
    print(__file__ + " start!!")
    # way points
    way_point_x = [-1.0, 3.0, 4.0]
    way_point_y = [0.0, -3.0, 1.0]
    # way_point_x = [1.0, 4.0, 6.0]
    # way_point_y = [1.0, 3.0, 6.0]
    # way_point_x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    # way_point_y = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10]
    n_course_point = 50 # sampling number

    plt.subplots()
    rax, ray = approximate_b_spline_path(way_point_x, way_point_y, n_course_point, s=0.1)
    print("X = ",rax,"Y = ",ray)
    plt.plot(rax, ray, '-r', label="Approximated B-Spline path")
    # plot_curvature(rax, ray)

    plt.title("B-Spline approximation")
    plt.plot(way_point_x, way_point_y, '-og', label="way points")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    # plt.subplots()
    # rix, riy = interpolate_b_spline_path(
    #     way_point_x, way_point_y, n_course_point)
    # plt.plot(rix, riy, '-b', label="Interpolated B-Spline path")
    # # plot_curvature(rix, riy, heading, curvature)

    # plt.title("B-Spline interpolation")
    # plt.plot(way_point_x, way_point_y, '-og', label="way points")
    # plt.grid(True)
    # plt.legend()
    # plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()