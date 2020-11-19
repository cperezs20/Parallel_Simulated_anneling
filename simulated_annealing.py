import numpy as np
from metropolis_hastings import Metropolis_Hastings
from init_and_save import simulation2pdb, initialization
from time import time

start = time()

## polymer based on HP model (hydrophobic-polar protein folding model)
#l="PHPPHPPHHPPHHPPHPPHP"
l = "HPPPHHPPHPHHHHHH"
#l= "HPPPHHPPHPHHHHHHPPHPPPHHPHPPPHPHPHHHPPHPHHPH"

K = 100       #number of steps of the Metropolis-Hastings algorithm
kb = 1          #Boltzman constant 
Tmax = 1        #initial temperature
Tmin = 0.15     #final temperature
delta = 1       #constant used during calculating energy for each microstate
alfa = 0.05     #temperature will decrease by alfa

start_microstate, rotation_matrices, matrix_polymer = initialization(l)

T = Tmax
simulation = []

while T>=Tmin:
    
    print("Temp:",T)

    aM, number_of_contacts, energy = Metropolis_Hastings(K, T, start_microstate, rotation_matrices,
                                                         delta,matrix_polymer)
    microstateX = aM[-1] #last microstate from previous step will be the first of current step

    for j in range(len(aM)):
        simulation.append((T, aM[j], number_of_contacts[j], energy[j])) #temp, matrix, number_of_contacts, energy

    T -= alfa #T = T - alfa
    T = np.round(T, 5)

simulation2pdb(simulation, matrix_polymer, "output/trajectory_sa.pdb")

end = time()
print(f'It took {end - start} seconds!')
