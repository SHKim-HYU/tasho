""" A library for distance computations between simple geometric primitives """

import casadi as cs
from tasho.utils import geometry

def dist_spheres(sphere1, sphere2):

	""" A function to compute the signed-distance between two spheres """

	dist = sphere1['center'] - sphere2['center'] - (sphere1['radius'] + sphere2['radius'])

	dist_measure = cs.sumsqr(dist)**0.5
	dist_vector = dist/cs.norm_1(dist)

	return dist_measure, dist_vector


def dist_sphere_box(sphere, box, vector = False):

	""" A function to compute the distance between a sphere and a box """

	#convert the sphere center to the box coordinates
	sphere_box_coord = geometry.inv_T_matrix(box["tf"])@cs.vercat(sphere["center"], cs.DM.ones(1))
	diff = box['center'] - sphere['center']

	mins = diff - box['xyz_len']
	maxs = -box['xyz_len'] - diff
	dist_surfaces = cs.vercat(mins, maxs)

	dist = cs.mmax(dist_surfaces)

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


if __name__ == '__main__':

	print("no syntax errors")


