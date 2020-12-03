"""
Simulated annealing main code
This script initializes and call metropolis_hasting function to
compute the simmulated annealing code
Then it saves the resulta into a pdb file
"""

from time import time
import numpy as np
from metropolis_hastings import metropolis_hasting
from init_and_save import simulation2pdb, initialization


start = time()

## polymer based on HP model (hydrophobic-polar protein folding model)
SEQUENCE = "HPPPHHPPHPHHHHHH"

K = 10       #number of steps of the Metropolis-Hastings algorithm
KB = 1          #Boltzman constant
TMAX = 1        #initial temperature
TMIN = 0.95     #final temperature
DELTA = 1       #constant used during calculating energy for each microstate
ALPHA = 0.05     #temperature will decrease by alfa

start_microstate, rotation_matrices, matrix_polymer = initialization(SEQUENCE)

T = TMAX
simulation = []

while T >= TMIN:

    print("Temp:", T)

    aM, number_of_contacts, energy = metropolis_hasting(K, T, start_microstate, rotation_matrices,
                                                        DELTA, matrix_polymer)
    start_microstate = aM[-1] #last microstate from previous step will be the first of current step

    for j, _ in enumerate(aM):
        # temp, matrix, number_of_contacts, energy
        simulation.append((T, aM[j], number_of_contacts[j], energy[j]))

    T -= ALPHA #T = T - alfa
    T = np.round(T, 5)

simulation2pdb(simulation, matrix_polymer, "output/trajectory_sa_par.pdb")

end = time()
print(f'It took {end - start} seconds!')
