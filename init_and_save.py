"""
Initialization of everything
"""

import numpy as np


def rotation_matrix(alfa):
    """
    Define the rotation matrix. 
    This matrix represents the relative directives to predict the location 
    of CA in a lattice.
    """
    return np.matrix([[round(np.math.cos(alfa)), round(-np.math.sin(alfa))],
		  [round(np.math.sin(alfa)), round(np.math.cos(alfa))]])


# l is defined in "simulated_anneling.py. It is the aminoacid sequence.
def initialization(l):
    #l.upper: Capital letters. Matrix polimer is an array (1 column) with the aminoacids letters (H,P)
    matrix_polymer = np.array([i for i in l.upper()])
    #initialization
    #a = array((1,0)*len(l)).reshape((len(l),2))
    #but there is no need to take two first coordinates of amino acids:
    start_microstate = np.array((1,0)*(len(l)-2)).reshape((len(l)-2,2))

    m90 = rotation_matrix(np.math.pi/2.0)
    m180 = rotation_matrix(np.math.pi)
    m270 = rotation_matrix((3.0/2.0)*np.math.pi)
    return start_microstate, [m180, m90, m270], matrix_polymer

    
def matrix2coord(m):
    return m.cumsum(axis=0)


def simulation2pdb(simulation, hp_polymer, output_path):
    
    f = open(output_path, "w")
    aa_type = ["LYS" if x=="P" else "ALA" for x in hp_polymer]
    
    for m in range(len(simulation)):
        #temp, matrix, number_of_contacts, energy = simulation[m][0], simulation[m][1], simulation[m][2], simulation[m][3]
        _, matrix, number_of_contacts, _ = simulation[m][0], simulation[m][1], simulation[m][2], simulation[m][3]
        f.write("MODEL " + str(m + 1) + "\n")
        f.write("COMMENT:	nOfContacts=" + str(number_of_contacts)+"\n")
      
        coords = matrix2coord(matrix)
        for i in range(coords.shape[0]):
            c = coords[i].tolist()[0]
            f.write("ATOM      "+str(i)+"  CA  " + aa_type[i] + "   "+str(i)+"  A     " + str(float(c[0])) + "   " + str(float(c[1])) + "   0.000\n")
        f.write("ENDMDL\n")
    f.close()
    