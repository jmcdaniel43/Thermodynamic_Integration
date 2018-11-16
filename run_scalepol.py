from __future__ import print_function
import logging
from simtk.openmm.app import *
from simtk.openmm import *
import simtk.openmm as mm
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
#******** this is module that goes with sapt force field files to generate exclusions
from sapt_exclusions import *
#******** this contains the Thermodynamic Integration subroutines 
from TI_classes import * 


#******************* Change this for CUDA/OpenCl/CPU kernels ******************
platform = Platform.getPlatformByName('CUDA')
properties = {'Precision': 'mixed', 'DeviceIndex':'0'}
# this is platform used for simulation object that computes numerical derivative
platform_dx = Platform.getPlatformByName('CPU')
#**********************************************************************************

#************************************************************************
#  set True if we want to switch to SCF routine for Drude optimization for numerical derivatives
#  this is more rigorous, as we assume dE/dx_drude = 0 when computing charge derivatives, which is
#  only true at the SCF limit
use_SCF_lambda_pol = True

#********* set true if we want to print total energies when computing numerical derivative (debugging)
print_energies = True

#********** this is the solute molecule for TI
solutename='NO3'

energyfile='energies_SCF.log'
derivativefile='dE_dlambda_SCF.log'

temperature=300*kelvin
pdb = PDBFile('md_npt.pdb')
ffdir = './ffdir/'
strdir = ''

integrator = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)

pdb.topology.loadBondDefinitions( ffdir + 'sapt_residues.xml')
pdb.topology.createStandardBonds();
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField( ffdir + 'sapt.xml')
modeller.addExtraParticles(forcefield)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, constraints=None, rigidWater=True)

#************************************************
#   Create TI object for Thermodynamic Integration
#
TI_system = TI(solutename, system, modeller, forcefield, integrator, platform, properties, use_SCF_lambda_pol )
#************************************************

logger_energy = construct_logger('energies', energyfile )
logger_derivative = construct_logger('derivatives', derivativefile )


# create simulation object for numerical derivative
TI_system.lambda_derivative = TI_system.numerical_lambda_derivative( TI_system, platform_dx )

# if we're doing SCF for polarization, need another simulation context...
if use_SCF_lambda_pol == True:
    TI_system.lambda_SCF = TI_system.numerical_lambda_derivative( TI_system, platform_dx )


logger_derivative.info('Thermodynamic Integration: Scaling down polarization...')
if print_energies :
    logger_energy.info('Thermodynamic Integration: Scaling down polarization...')

#************************************************
#    TI step 1:  Turn off polarization
#************************************************
# Equilibration after each change of Hamiltonian
n_equil=10000

for lambda_i in [ 1.0 , 0.8 , 0.6 , 0.4 , 0.2 , 0.01 ]:  # scale factor of zero is numerically unstable, so use 0.01 instead...
    # scale polarization in main simulation object for this value of lambda
    flag = simulation_scale_polarization( TI_system, lambda_i )
    # scale polarization in numerical derivative
    TI_system.lambda_derivative.create_numerical_derivative( lambda_pol=True , lambda_elec=False, scalefactor=lambda_i )

    logger_derivative.info('%(str1)s %(lambda)r', { 'str1' : 'derivatives at lambda = ', 'lambda' : lambda_i })
    logger_derivative.info('%(str1)s %(nstep)d %(str2)s' , { 'str1' : 'Equilibrating for  ', 'nstep' : n_equil , 'str2' : ' steps'})
    if print_energies :
        logger_energy.info('%(str1)s %(lambda)r', { 'str1' : 'energies at lambda = ', 'lambda' : lambda_i })
        logger_energy.info('%(str1)s %(nstep)d %(str2)s' , { 'str1' : 'Equilibrating for  ', 'nstep' : n_equil , 'str2' : ' steps'})

    # equilibrate
    TI_system.simmd.step(n_equil)

    #*********************** Sample numerical derivatives for polarization energy ****************************
    for i in range(100):
        TI_system.simmd.step(1000)

        if print_energies :
            logger_energy.info( 'Energies from default simulation context' )
            flag = log_system_energy( TI_system, logger_energy )

        if use_SCF_lambda_pol == True:
            # use SCF to compute numerical derivative
            state = TI_system.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
            position = state.getPositions()
            TI_system.lambda_SCF.simmd.context.setPositions(position)
            TI_system.lambda_SCF.simmd.step(1)
            if print_energies:
                logger_energy.info( 'Energies from SCF Drude simulation context' )
                flag=log_system_energy( TI_system.lambda_SCF, logger_energy )
            dH_dlambda = TI_system.lambda_derivative.compute_numerical_derivative( TI_system.lambda_SCF.simmd, print_energies, logger_energy )
        else:
            #dE_dlambda evaluated from Thermal Drudes, derivative will not be rigorous (since Hellman-Feynman doesn't hold)
            dH_dlambda = TI_system.lambda_derivative.compute_numerical_derivative( TI_system.simmd, print_energies, logger_energy )

        logger_derivative.info('%(str1)s  %(i)d %(dH)r', { 'str1': 'step', 'i' : i , 'dH' : dH_dlambda })

exit()
