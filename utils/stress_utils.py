import numpy as np

def compute_von_mises_stress(cauchy_tensor):

    """
    To compute the Von Mises stress from the Cauchy stress tensor, you can follow these steps:

    1. Compute the deviatoric stress tensor, denoted as S:

    - Subtract the mean stress component from the Cauchy stress tensor to obtain the deviatoric stress tensor.
    - The mean stress component is calculated as the average of the three principal stresses: σ_mean = (σ_xx + σ_yy + σ_zz) / 3.
    - The deviatoric stress tensor S is obtained by subtracting the mean stress component from the Cauchy stress tensor:
        S = σ - σ_mean * I, where σ is the Cauchy stress tensor, and I is the identity tensor.
    
    2. Compute the Von Mises stress (σ_VM) from the deviatoric stress tensor S:
    - Compute the second invariant of the deviatoric stress tensor, J2:
        J2 = (1/2) * (S_ij * S_ij), where S_ij represents the components of the deviatoric stress tensor.
    - Calculate the Von Mises stress as the square root of three times the second invariant:
        σ_VM = sqrt(3 * J2).    
    
    """

    # Compute the deviatoric stress tensor
    mean_stress = np.mean(cauchy_tensor)
    deviatoric_stress = cauchy_tensor - mean_stress * np.identity(3)

    # Compute the second invariant of the deviatoric stress tensor
    deviatoric_stress_squared = np.square(deviatoric_stress)
    deviatoric_stress_sum = np.sum(deviatoric_stress_squared)
    j2 = 0.5 * deviatoric_stress_sum

    # Compute the Von Mises stress
    von_mises_stress = np.sqrt(3 * j2)

    return von_mises_stress


def compute_stress_each_vertex(tet_stress, adjacent_tetrahedral_dict, vertex_idx):
    """
    Compute scalar stress value at each vertex by averaging stress values at all adjacent tetrahedral elements.  

    The stress tensor at each vertex is calculated by averaging
    the stress tensors at all adjacent tetrahedral elements. The average stress tensor is
    then converted to a scalar von Mises stress (i.e., the second
    invariant of the deviatoric stress), which is widely used to
    quantify whether a material has yielded.     

    tet_stress: shape (num_tetrahedra, 3, 3). Cauchy stress tensors (3,3) at each tetrahedron.
    von_mises_stress: scalar stress value at vertex_idx
    
    """
    
    # avg_cauchy_stress_tensor = np.zeros((3,3))
    # for tetrahedra_idx in adjacent_tetrahedral_dict[vertex_idx]:    # iterate through each of tetrahedron that contains vertex_idx    
        
    #     avg_cauchy_stress_tensor += tet_stress[tetrahedra_idx]

    # avg_cauchy_stress_tensor /= len(adjacent_tetrahedral_dict[vertex_idx])  # average out over all all adjacent tetrahedras
    
    avg_cauchy_stress_tensor = np.mean(tet_stress[adjacent_tetrahedral_dict[vertex_idx]], axis=0)   # average out over all all adjacent tetrahedras. Shape (3,3).     
    von_mises_stress = compute_von_mises_stress(avg_cauchy_stress_tensor)   # convert 3x3 mat to a scalar stress value
    
    return von_mises_stress


def compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, num_vertices):
    """ 
    return stresses at all vertices. Shape (full_pc.shape[0],) or (num_vertices,)
    """
    all_stresses = []
    for i in range(num_vertices):
        all_stresses.append(compute_stress_each_vertex(tet_stress, adjacent_tetrahedral_dict, vertex_idx=i))    
    
    return np.array(all_stresses)

def get_adjacent_tetrahedrals_of_vertex(tet_indices):
    """

    For each vertex in the tetrahedral mesh, find a list of tetrahedral elements that include it.
    ex: 

    tet_indices = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 4, 6]]
    Vertex 0 belongs to tetrahedra: [[0, 1, 2, 3], [0, 2, 4, 6]]
    Vertex 1 belongs to tetrahedra: [[0, 1, 2, 3]]
    Vertex 2 belongs to tetrahedra: [[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 4, 6]]
    Vertex 3 belongs to tetrahedra: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    Vertex 4 belongs to tetrahedra: [[1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 4, 6]]
    Vertex 5 belongs to tetrahedra: [[2, 3, 4, 5]]
    Vertex 6 belongs to tetrahedra: [[0, 2, 4, 6]]

    Return: a dict. Key is the vertex index. Value is the list of tetrahedra that this vertex belongs to.
    ex: {0: [[0, 1, 2, 3], [0, 2, 4, 6]]}
    """
    
    
    adjacent_tetrahedral_dict = {}

    for idx, tetrahedron in enumerate(tet_indices):
        v1, v2, v3, v4 = tetrahedron

        # Add the tetrahedron to the vertex's list
        adjacent_tetrahedral_dict.setdefault(v1, []).append(idx)    # setdefault(key, default) is a dictionary method in Python that returns the value of a specified key if it exists in the dictionary. If the key does not exist, it inserts the key with a specified default value and returns the default value.
        adjacent_tetrahedral_dict.setdefault(v2, []).append(idx)
        adjacent_tetrahedral_dict.setdefault(v3, []).append(idx)
        adjacent_tetrahedral_dict.setdefault(v4, []).append(idx)

    return adjacent_tetrahedral_dict


