"""
Initialization and saving functions
"""

import numpy as np


def rotation_matrix(alfa):
    """
    Function to create the rotation matrix

    Paramters:
    ---------
    alpha : float
        Angle in radians

    Returns:
    --------
    ndarray
        Rotation matrix
    """

    return np.array([[round(np.math.cos(alfa)), round(-np.math.sin(alfa))],
		              [round(np.math.sin(alfa)), round(np.math.cos(alfa))]])


def initialization(amino_seq):
    """
    Function to initialize

    Parameters:
    -----------
    amino_seq : str
        Aminiacid sequence as a string

    Returns:
    --------
    init_microstate : ndarray
        Array with the initial microstate
    ndarray
        Array with the rotations at 180, 90 and 270 degrees
    matrix_polymer : ndarray
        Aminoacid sequence as matrix
    """

    #Parse string in uppercase letters
    matrix_polymer = np.array(list(amino_seq.upper()))

    #Creating initial microstate using the aminoacid sequence
    init_microstate = np.array((1, 0)*(len(amino_seq)-2)).reshape((len(amino_seq)-2, 2))

    #Generating rotations matrices for 90, 180 and 270 degrees
    m90 = rotation_matrix(np.math.pi/2.0)
    m180 = rotation_matrix(np.math.pi)
    m270 = rotation_matrix((3.0/2.0)*np.math.pi)

    return init_microstate, [m180, m90, m270], matrix_polymer


def matrix2coord(matrix):
    """
    Transform a matrix into coordinates

    Parameters:
    -----------
    matrix : ndarray
        Matrix with microstates

    Returns:
    ndarray
        Coordinates of the matrix of microstates
    """

    return matrix.cumsum(axis=0)


# def simulation2pdb(simulation, hp_polymer, output_path="output/trajectory_sa.pdb"):
#     """
#     Converts results from simulation into a pdb file

#     Parameters:
#     -----------
#     simulation : list
#         Results from the simulation (simulated annealing)
#     hp_polymer : ndarray
#         Aminoacid sequence
#     output_path : str (optional)
#         Path to save the output file
#     """

#     fid = open(output_path, "w")

#     aa_type = ["LYS" if x=="P" else "ALA" for x in hp_polymer]

#     for i, _ in enumerate(simulation):
#         _, matrix, number_of_contacts, _ = simulation[i][0], simulation[i][1],\
#                                            simulation[i][2], simulation[i][3]
#         fid.write("MODEL " + str(i + 1) + "\n")
#         fid.write("COMMENT:	nOfContacts=" + str(number_of_contacts)+"\n")

#         coords = matrix2coord(matrix)
#         for j in range(coords.shape[0]):
#             list_coords = coords[j].tolist()[0]
#             fid.write("ATOM      "+str(j)+"  CA  " + aa_type[j] + "   "+str(i)+"  A     "\
#                       + str(float(list_coords[0])) + "   " + str(float(list_coords[1])) +\
#                       "   0.000\n")
#         fid.write("ENDMDL\n")
#     fid.close()

def simulation2pdb(simulation, hp_polymer, output_path):
    
    f = open(output_path, "w")
    aa_type = ["LYS" if x=="P" else "ALA" for x in hp_polymer.tostring() ]
    
    for m in range(len(simulation)):
        temp, matrix, number_of_contacts, energy = simulation[m][0], simulation[m][1], simulation[m][2], simulation[m][3]
        f.write("MODEL " + str(m + 1) + "\n")
        f.write("COMMENT:	nOfContacts=" + str(number_of_contacts)+"\n")
      
        coords = matrix2coord(matrix)
        for i in range(coords.shape[0]):
            c = coords[i].tolist()[0]
            print(c)
            f.write("ATOM      "+str(i)+"  CA  " + aa_type[i] + "   "+str(i)+"  A     " + str(float(c[0])) + "   " + str(float(c[1])) + "   0.000\n")
        f.write("ENDMDL\n")
    f.close()
    