import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial import ConvexHull

def generate_vertices_from_box_constraints(box):
    # Computes the vertices corresponding to N dimensional box constraints.

    vertices_list = [list(np.array([min_val, max_val])) for min_val, max_val in box]
    vertices = list(itertools.product(*vertices_list))
    vertices = np.array(vertices)
    
    return vertices

def find_adjacent_vertices(vertices, given_vertex):
    # Computes the vertices adjacent to a given vertex in a polytope

    hull = ConvexHull(vertices)
    given_vertex_index = np.argwhere((hull.points == given_vertex).all(axis=1)).flatten()[0]
    indices = np.where((hull.points[hull.simplices] == given_vertex).all(axis=2))
    simplices = hull.simplices[indices[0]]
    adjacent_vertices = simplices[(simplices != given_vertex_index)]
    
    return np.unique(hull.points[adjacent_vertices],axis=0)

def estimate_cis(system,initial_cis,plotting=False):
    # Uses an iterative heuristic to estimate a control invariant polytope for a linear system

    A = system.A
    B = system.B
    state_dim = system.n_states
    control_dim = system.n_inputs

    u_min = system.control_bounds[:,0]
    u_max = system.control_bounds[:,1]

    polytope_vertices = initial_cis

    eps = 10**(-8)
    max_iter = 10
    max_length = 1000
    iter_count = 0
    converged = False
    length = len(polytope_vertices)
    coeff = 1/5

    while iter_count < max_iter and length < max_length and not converged:
        old_vertices = polytope_vertices.copy()
        old_length = len(old_vertices)
        for vertex in old_vertices:

            w = cp.Variable((old_length))
            x = cp.Variable((state_dim))
            u = cp.Variable((control_dim))
            d = cp.Variable((state_dim))

            constraints = [A@vertex + B@u == x,
                        u_min <= u,
                        u <= u_max,
                        w <= 1,
                        w >= 0,
                        cp.sum(w) == 1,
                        d == old_vertices.T@w]
            
            objective = cp.norm(x-d)

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve()

            if objective.value > eps:
                adjacent_vertices = find_adjacent_vertices(old_vertices,vertex)
                ind = np.argwhere((polytope_vertices == vertex).all(axis=1)).flatten()[0]
                polytope_vertices = np.delete(polytope_vertices,ind,axis=0)
                for adj_v in adjacent_vertices:
                    new_vertex = (adj_v * coeff + vertex * (1-coeff))
                    polytope_vertices = np.vstack((polytope_vertices,new_vertex))

        length = len(polytope_vertices)

        if length == old_length:
            converged = np.all(polytope_vertices == old_vertices)

        print(f"Iteration: {iter_count}")
        print(f"# vertices {length}")
        print(f"Converged: {converged}")

        iter_count += 1

        if state_dim == 2 and plotting:
            old_hull = ConvexHull(old_vertices)
            plt.title(f"# vertices {length}")
            plt.plot(old_vertices[old_hull.vertices,0],old_vertices[old_hull.vertices,1],'-ob')
            plt.plot(old_vertices[old_hull.vertices[[0,-1]],0],old_vertices[old_hull.vertices[[0,-1]],1],'-b')

            hull = ConvexHull(polytope_vertices)
            plt.plot(polytope_vertices[hull.vertices,0],polytope_vertices[hull.vertices,1],'-ok')
            plt.plot(polytope_vertices[hull.vertices[[0,-1]],0],polytope_vertices[hull.vertices[[0,-1]],1],'-k')

            plt.show()

    if converged:
        print(f"Converged to a polytope with {length} vertices")
        return polytope_vertices
    else:
        print(f"Did not converge")
        return None