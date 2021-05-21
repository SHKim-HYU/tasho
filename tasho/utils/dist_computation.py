""" A library for distance computations between simple geometric primitives """

import casadi as cs
from tasho.utils import geometry
import numpy as np


def dist_spheres(sphere1, sphere2):

    """ A function to compute the signed-distance between two spheres """

    dist = cs.sumsqr(sphere1["center"] - sphere2["center"]) ** 0.5 - (
        sphere1["radius"] + sphere2["radius"]
    )

    # dist_vector = dist/cs.norm_1(dist)

    return dist


def softmax(d, alpha=50):
    """A function to compute the softmax of a vector. To be used to implement
    the union operation to compute the minimum distance from the obstacle

    :param d: A list of numbers whose softmax is to be computed.
    :type d: list of floats.

    :param overflow: Takes measures to avoid overflow if the vector is too long if set to true.
    :type overflow: boolean.
    """

    max_element = cs.mmax(d)
    return (
        max_element
        + 1 / alpha * cs.log(cs.sum1(cs.exp(alpha * (d - max_element))))
        - cs.log(d.shape[0]) / alpha
    )


def dist_sphere_box(sphere, box, vector=False, do_softmax=False):

    """ A function to compute the distance between a sphere and a box """

    # convert the sphere center to the box coordinates
    sphere_box_coord = geometry.inv_T_matrix(box["tf"]) @ cs.vertcat(
        sphere["center"], cs.DM.ones(1)
    )
    diff = sphere_box_coord[0:3]

    # diff = box["tf"][0:3,3] - sphere["center"]

    mins = diff - (box["xyz_len"] + sphere["radius"])
    maxs = -(box["xyz_len"] + sphere["radius"]) - diff
    dist_surfaces = cs.vertcat(mins, maxs)
    if do_softmax:
        dist = softmax(dist_surfaces, alpha=200)
    else:
        dist = -100  # a ridiculously small number
        for i in range(6):
            dist = cs.fmax(dist_surfaces[i], dist)

    if not vector:
        return dist
    else:
        for i in range(6):
            if dist_surfaces[i] == dist:
                index = i
                if i == 0:
                    dist_vector = cs.vcat([-1, 0, 0])
                elif i == 1:
                    dist_vector = cs.vcat([0, -1, 0])
                elif i == 2:
                    dist_vector = cs.vcat([0, 0, -1])
                elif i == 3:
                    dist_vector = cs.vcat([1, 0, 0])
                elif i == 4:
                    dist_vector = cs.vcat([0, 1, 0])
                else:
                    dist_vector = cs.vcat([0, 0, 1])

                return dist, dist_vector


if __name__ == "__main__":

    print("no syntax errors")

    # Adding some tests
    cube = {}
    cube["tf"] = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]])
    cube["xyz_len"] = np.array([0.15, 0.15, 0.15])
    ball = {"center": np.array([0.5, 0.35, 0.15]), "radius": 0.1}
    assert cs.fabs(dist_sphere_box(ball, cube) - 0.1) <= 1e-12

    ball2 = {"center": np.array([0.5, 0.25, 0.15]), "radius": 0.1}
    assert cs.fabs(dist_spheres(ball, ball2) + 0.1) <= 1e-12
