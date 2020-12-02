"""
Metropolis-Hastings algorithm implementation
"""

import numba as nb
import numpy as np


@nb.njit(fastmath=True, parallel=True)
def get_allowed_neighborhs(microstate):
    """
    This funcion gets the allowed neighbors for a given
    polymer matrix

    Parameters:
    -----------
    aminoacid_matrix : ndarray
        Numpy array containing the polymer matrix

    Returns:
    --------
    allowed_neighbors : ndarray
        Array with the selected neigborhs
    """

    allowed_neighborhs = []
    for i in range(microstate.shape[0]):
        complete_matrix = np.vstack((np.array([[0, 0], [1, 0]]), microstate[i]))
        coord_matrix = np.empty(complete_matrix.shape)
        for j in range(complete_matrix.shape[1]):
            coord_matrix[:, j] = np.cumsum(complete_matrix[:, j])
        x_mat = np.random.rand(coord_matrix.shape[1])
        y_mat = coord_matrix.dot(x_mat)
        unique = np.unique(y_mat)
        if unique.shape[0] == coord_matrix.shape[0]:
            allowed_neighborhs.append(microstate[i])

    return allowed_neighborhs


@nb.njit
def num_adjacent_h(seq_matrix):
    """
    Function to count occurences where two H are adjacent

    Parameters:
    -----------
    seq_matrix : ndarray
        Array containing the sequence

    Returns:
    --------
    counter : int
        Counter of occurences of HH
    """

    counter = 0
    for i in range(len(seq_matrix)-1):
        if seq_matrix[i] == "H" and seq_matrix[i+1] == "H":
            counter += 1

    return counter


@nb.njit
def num_contacts(microstate, seq_matrix):
    """
    Function to count the number of contacts

    Parameters:
    -----------
    microstate : ndarray
        Matrix of microstates
    seq_matrix : ndarray
        Matrix with sequence

    Returns:
    --------
    acum : int
        Number of contacts
    """

    acum = 0
    right_up = np.array([[1,0], [0, 1]])
    complete_matrix = np.vstack((np.array([[0, 0], [1, 0]]), microstate))
    #calculation of absolute coordinates in 2D
    #i.e. adding two first rows to current matrix and cumsum on it
    coord_matrix = np.empty(complete_matrix.shape)
    for j in range(complete_matrix.shape[1]):
        coord_matrix[:, j] = np.cumsum(complete_matrix[:, j])

    m_logic = [idx for idx, element in enumerate(seq_matrix) if element == 'H']
    m_hydro = np.empty((len(m_logic), 2))
    for i, idx in enumerate(m_logic):
        m_hydro[i] = coord_matrix[idx, :]

    for i in range(right_up.shape[0]):
        m_moved = m_hydro + right_up[i]
        aux = np.vstack((m_hydro, m_moved))
        x_mat = np.random.rand(aux.shape[1])
        y_mat = aux.dot(x_mat)
        m_wspolna = np.unique(y_mat)
        print(m_hydro.shape[0], m_moved.shape[0], m_wspolna.shape[0])
        acum += (m_hydro.shape[0] + m_moved.shape[0]) - m_wspolna.shape[0]
    #print(acum)
    return acum

@nb.njit
def calc_energy(microstate, delta, seq_matrix):
    """
    Function to compute the energy of a given microstate

    Parameters:
    -----------
    microstate : ndarray
        Matrix of microstates
    delta : float
        Current value of delta
    seq_matrix : ndarray
        Matrix with sequence

    Returns:
    --------
    energy : ndarray
        Energy of the given microstate
    """

    n_contact = num_adjacent_h(seq_matrix)-num_contacts(microstate, seq_matrix)
    energy = -1*delta*n_contact

    return n_contact, energy


@nb.njit
def create_new_microstate(microstate, rot_matrices):
    """
    Function to create a new random microstate
    It finds the neigborhs from a given microstate to then
    select the subset containing the non-overlaped microstates
    finally, a random microstate from the subset is selected

    Parameters:
    -----------
    microstate : ndarray
        A matrix with the current microstate
    rot_matrices : ndarrray
        A multidimensional array containing the rotation matrices

    Returns:
    --------
    int
        Length of the subset of allowed neigborhs
    ndarray
        An array with the alloed neighbor
    """

    neighbors = find_neighbors(microstate, rot_matrices)
    neighbors_allowed = get_allowed_neighborhs(neighbors)
    idx = np.random.randint(0,len(neighbors_allowed))

    return len(neighbors_allowed), neighbors_allowed[int(idx)]


@nb.njit
def find_neighbors(microstate, rot_matrices):
    """
    Find the neighbors for a given microstate

    Parameters:
    -----------
    microstate : ndarray
        A matrix with the current microstate
    rot_matrices : list
        A list with the rotation matrices
    """

    neighbors_set = np.empty((microstate.shape[0]*len(rot_matrices), microstate.shape[0],
                         microstate.shape[1]), dtype=np.float32)
    for i in range(rot_matrices.shape[0]):
        micro_rot = np.dot(microstate.astype(np.float32), rot_matrices[i].astype(np.float32))
        for j in range(microstate.shape[0]):
            neighbors_set[i*microstate.shape[0]+j, :, :] = np.vstack((microstate[:j, :],
                                                                      micro_rot[j:, :]))
    return neighbors_set


# @nb.njit
def metropolis_hasting(steps, temperature, microstate_x, rotate_matrices, delta, matrix_l):
    """
    X-old microstate, Y-new microstate
    """

    microstates = np.empty((steps, microstate_x.shape[0], microstate_x.shape[1]))
    n_of_contacts = np.empty(steps)
    energy = np.empty(steps)
    i = 0
    while i < steps:

        n_contacts_x, energy_x = calc_energy(microstate_x, delta, matrix_l)

        n_neighbors_x, microstate_y = create_new_microstate(microstate_x, rotate_matrices)

        n_contacts_y, energy_y = calc_energy(microstate_y, delta, matrix_l)

        n_neighbors_y, _ = create_new_microstate(microstate_y, rotate_matrices)

        #microstate Z is not needed to further calculations
        prob_accept_y = (-energy_y/temperature)-np.math.log(n_neighbors_y)-\
                        (-energy_x/temperature)+np.math.log(n_neighbors_x)
        random_number = np.math.log(np.random.random_sample(1)[0]) #e.g.array([ 0.25290701])

        if random_number < prob_accept_y: # both numbers are logarithmized
            #creating new state
            microstates[i] = microstate_y
            n_of_contacts[i] = n_contacts_y
            energy[i] = energy_y
            microstate_x = microstate_y
        else:
            #old state stays
            microstates[i] = microstate_x
            n_of_contacts[i] = n_contacts_x
            energy[i] = energy_x
        i += 1

    return microstates, n_of_contacts, energy
