# Convergence.py                                                              #
# -*- coding: utf-8 -*-                                                       #
#                                                                             #
# This scprit plot several parameters related to the each step during the ion #
# relaxation and MD simulations in VASP. The source data is the file called   #
# <vaspout.h5> generated when the simulation ends correctly.                  #
#                                                                             #
# @Author: Marco A. Villena                                                   #
# @Version: 0.3                                                               #
###############################################################################

from operator import truediv
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')

###### Open the <vaspout.h5> file ######
try:
    f = h5py.File('vaspout.h5', 'r')
except FileNotFoundError:
    print('File not found')
    sys.exit()

###### Detection of MD/Ion relaxation by IBRION paramenter in INCAR ######
ibrion = np.array(f['input']['incar']['IBRION'])
if ibrion == 0:
    print('MD simulation detected')
    md_control = True
else:
    print('Ion relaxation detected')
    md_control = False

###### Energies ######
energy_names = list(f['intermediate']['ion_dynamics']['energies_tags'])
energy_names = [str(item) for item in energy_names]
energy_names = [item.replace("b'", "") for item in energy_names]
energy_names = [item.replace("'", "") for item in energy_names]

energies = np.array(f['intermediate']['ion_dynamics']['energies'])

###### Forces ######
forces = np.array(f['intermediate']['ion_dynamics']['forces'])

shape_aux = forces.shape  # [steps, atoms, xyz-axis]
print('Number of steps: ' + str(shape_aux[0]))

forces_result = np.zeros((shape_aux[0], 2))

forces_atoms = np.zeros((shape_aux[1], 1))

for i in range(shape_aux[0]):  # steps
    for j in range(shape_aux[1]):  # atoms
        aux = forces[i, j, :]
        forces_atoms[j, 0] = np.sqrt(aux.dot(aux))

    forces_result[i, 0] = np.mean(forces_atoms)
    forces_result[i, 1] = np.amax(forces_atoms)

###### Figures ######
if md_control:
    fig = plt.figure(figsize=(15, 10))
    sub1 = plt.subplot(2, 2, 1) # Energy plot
    sub2 = plt.subplot(2, 2, 2) # Temperature plot
    sub3 = plt.subplot(2, 2, 3) # Nose plot
    sub4 = plt.subplot(2, 2, 4) # Forces plot

    # Energy plot
    sub1.plot(energies[:,[0, 6]], '.-', label=[energy_names[0], energy_names[6]])
    sub1.set_xlim([0, shape_aux[0]])
    sub1.set_xlabel('# Step')
    sub1.set_ylabel('Energy (eV)')
    sub1.legend()

    # Temperature plot
    sub2.plot(energies[:,3], '.-', label=energy_names[3])
    sub2.set_xlim([0, shape_aux[0]])
    sub2.set_xlabel('# Step')
    sub2.set_ylabel('Temperature (K)')
    sub2.legend()

    # Nose potential/kinetic plot
    sub3.plot(energies[:,[4, 5]], '.-', label=[energy_names[4], energy_names[5]])
    sub3.set_xlim([0, shape_aux[0]])
    sub3.set_xlabel('# Step')
    sub3.set_ylabel('Energy (eV)')
    sub3.legend()

    # Forces plot
    sub4.plot(forces_result, '.-', label=['Mean', 'Max'])
    sub4.set_xlim([0, shape_aux[0]])
    sub4.set_ylim([0, np.amax(forces_result[:, 1])])
    sub4.set_ylabel('Force (eV/A)')
    sub4.set_xlabel('# Step')
    sub4.legend()

else:
    fig = plt.figure(figsize=(15, 10))
    sub1 = plt.subplot(2, 1, 1) # Energy plot
    sub2 = plt.subplot(2, 1, 2) # Forces plot

    # Energy plot
    sub1.plot(energies, '.-', label=energy_names)
    if shape_aux[0] > 1:
        sub2.set_xlim([0, shape_aux[0]])
    else:
        sub2.set_xlim([-1, 1])

    sub1.set_ylabel('Energy (eV)')
    sub1.legend()

    # Forces plot
    sub2.plot(forces_result, '.-', label=['Mean', 'Max'])
    if shape_aux[0] > 1:
        sub2.set_xlim([0, shape_aux[0]])
    else:
        sub2.set_xlim([-1, 1])

    sub2.set_ylim([0, np.amax(forces_result[:, 1])])
    sub2.set_ylabel('Force (eV/A)')
    sub2.set_xlabel('# Step')
    sub2.legend()

plt.show()

print('END')
