from __future__ import print_function
import logging
import os
from simtk.openmm.app import *
from simtk.openmm import *
import simtk.openmm as mm
from simtk.unit import *
import sys
from time import gmtime, strftime
from datetime import datetime
#******** this is module that goes with sapt force field files to generate exclusions
from sapt_exclusions import *
#******** this contains the Thermodynamic Integration subroutines 
from TI_classes import * 


#**************************************************************
#    This code runs Thermodynamic Integration for SAPT-FF force fields
#    currently, we use it to compute solvation free energies of ions in water,
#    but code is general.
#    there are 4 steps to the Thermodynamics Integration:
#
#    1) scale polarization:  polarization on solute molecule/ion is scaled from 1 to 0.01 (0.0 has numerical problems with Drude oscillators)
#       for this part, derivative dE_dlambda is computed numerically, using SCF integrator (switch from Langevin ==> SCF in between sampling)
#       we need to switch to SCF so that Hellman-Feynmann derivatives are exact (with Langevin, we observe positive (unphysical derivatives in some cases)
#
#    2) scale electrostatics:  electrostatics on solute molecule/ion is scaled from 1 to 0.0.  For this, we should load non-polarizable force field
#       for solute molecule/ion (polarization still on solvent), so that simulation is numerically stable for pol=0.0 on solute
#       derivative dE_dlambda is computed numerically
#  
#    3) scale SAPT-FF to LJ-like potential:  In order to use existing soft-core functional forms, we first scale SAPT-FF to LJ-like potential.  
#       Derivative dE_dlambda is computed analytically, using compute global derivative method for custom nonbonded forces within OpenMM
#
#    4) scale soft-core LJ repulsion:  This is the trickiest part, requiring "soft-core potentials".  At this stage we have turned off electrostatics
#       (and exluded these interactions in nonbonded force), and have scaled the SAPT-FF potential to an LJ-like potential (with an extra C8/R^8 term)
#       we then scale off this interaction (which includes both repulsion and dispersion) using a soft-core functional form similar to that given in the
#       Gromacs manual.  Note that it's important to scale repulsion/dispersion simultaneously for equal sampling.
#       We also have coded in the option of scaling molecule/ion by atom shells, defined by user.  The idea here is to treat the molecule
#       like an "onion", and remove atoms layer by layer.  However, we don't think this is generally a good idea ...
#
#
#    IMPORTANT NOTE:  We cannot switch between GPU/CPU contexts for numerical derivatives
#        This is because these involve different kernels, with slightly different numerical output
#        while these numerical differences may seem small (e.g. 0.5 kJ/mol for total system electrostatics)
#        they are not small compared to the delta_E from small delta_lambda derivatives, and cause severe artifacts!
#        for example, using both a GPU/CPU kernel for electrostatic numerical derivatives for methane, we get a value of
#        50 kJ/mol contribution to the solvation free energy!  this is just an artifact of the 0.5 kJ/mol difference between kernels,
#        and the use of a 0.01 delta lambda value ... so be careful, and switch from GPU to CPU/CPU kernels for numerical derivatives
#        (unless we have multiple GPU contexts)
#
#*******************************************************************

#********** Set this for the functional form of the solute potential.  If SAPT_FF = False, assume standard LJ without polarization
SAPT_FF_potential = True

#***********************************  Fill in these names/files for each simulation *******************
solutename='BF4'       # this is the solute molecule for TI
# for repulsion, we scale by atom shells, so this should be a list of lists, each inner list corresponding to atoms within
# the same atom shell.  For example for NO3, we scale repulsion off oxygen atoms first, then turn off nitrogen
atomshells=[]
atomshells.append( [ 'B', 'F1' , 'F2' , 'F3' , 'F4' ] )

energyfile_name='energies.log'
derivativefile_name='dE_dlambda.log'

# names of force field and residue files, one set with polarization on solute, one set without polarization on solute
ffxml_pol='sapt.xml'
resxml_pol='sapt_residues.xml'
ffxml_nopol='sapt_noionpol.xml'
resxml_nopol='sapt_residues_noionpol.xml'

pdbstart='md_npt.pdb'
ffdir = './ffdir/'

temperature=300*kelvin
pressure = 1.0*atmosphere
barofreq = 100

lambda_range = [ 1.0 , 0.9, 0.8 , 0.7, 0.6 , 0.5, 0.4 , 0.3, 0.2 , 0.1, 0.0 ]  # lambda values for TI

NPT_simulation=True  # NPT or NVT simulation??  if False, pressure/barofreq will be ignored...

n_equil=10000 # Equilibration after each change of Hamiltonian
n_deriv=500 # number of derivatives to sample for each lambda value
n_step=500  # number of MD steps between derivative sampling
trjsteps = 10000 # steps for saving trajectory...
chksteps =  1000 # steps for saving checkpoint file
 
#*********************************** End Input section

# platform type, use same for both contexts of numerical derivative, otherwise
# will give significant artifacts even with small differences!
platform_type = 'CUDA'  # e.g. 'CPU' or 'CUDA'

#******************* Change this for CUDA/OpenCl/CPU kernels ******************
platform = Platform.getPlatformByName(platform_type)
properties = {'Precision': 'mixed', 'DeviceIndex':'2'}
# this is platform used for simulation object that computes numerical derivative
platform_dx = Platform.getPlatformByName(platform_type)
#**********************************************************************************

#************************************************************************
#  set True if we want to switch to SCF routine for Drude optimization for numerical derivatives
#  this is more rigorous, as we assume dE/dx_drude = 0 when computing charge derivatives, which is
#  only true at the SCF limit
use_SCF_lambda_pol = True

#********* set true if we want to print total energies when computing numerical derivative (debugging)
print_energies = True

# NOTE: Don't change these string names!! they are hard-coded in as key-words in other parts of the code...
#       SAPT_FF_LJ scales the SAPT_FF potential to an LJ-like potential, and "repulsion" scales off the full LJ interaction
#       using a soft-core function.  We use the terminology "repulsion" even though we scale of attractive interactions here as well,
#       so as not to be confused that the original force field was LJ.
TI_jobs = [ "polarization" , "electrostatic" , "SAPT_FF_LJ" , "repulsion" ]

# first see if any of these directory exists (they shouldn't! if they do, then exit...)
for interaction_type in TI_jobs:
    directory="./"+interaction_type
    if os.path.exists(directory):
        print(" Directory ", directory , " already exists!")
        sys.exit()

# set initial positions, for first interaction type we will use these, otherwise we'll use positions from previous loop
pdb = PDBFile(pdbstart)
start_positions=pdb.positions


#********************************
#  Loop over 4 stages of lambda scaling:  1st) polarization, 2nd) electrostatic, 3rd) SAPT_FF_LJ, 4th) repulsion
#  treat these essentially as "separate simulations"
#********************************

for interaction_type in TI_jobs:    
    # directory for output
    directory="./"+interaction_type
    os.makedirs(directory)
    strdir=directory+"/"


    # see which force field files to use, whether polarizable or non-polarizable for solute....
    if interaction_type == "polarization":
        ffxml_local = ffxml_pol
        resxml_local = resxml_pol
    else:
        ffxml_local = ffxml_nopol
        resxml_local = resxml_nopol

    pdb = PDBFile(pdbstart)

    #****** Drude integrator for SAPT-FF
    if SAPT_FF_potential :
        integrator = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
    else :
        integrator = LangevinIntegrator(temperature, 1/picosecond, 0.001*picoseconds)

    pdb.topology.loadBondDefinitions( ffdir + resxml_local )
    pdb.topology.createStandardBonds();

    # use positions object, this is either initial positions or positions from previous loop
    modeller = Modeller(pdb.topology, start_positions)
    forcefield = ForceField( ffdir + ffxml_local )
    modeller.addExtraParticles(forcefield)

    # create system
    system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, constraints=None, rigidWater=True)

    # if NPT simulation
    if NPT_simulation:
        barostat = MonteCarloBarostat(pressure,temperature,barofreq)
        system.addForce(barostat)

    #************************************************
    #   Create TI object for Thermodynamic Integration
    #
    TI_system = TI(solutename, atomshells, system, modeller, forcefield, integrator, platform, properties, use_SCF_lambda_pol, interaction_type, SAPT_FF_potential, NPT_simulation )
    #************************************************

    energyfile = strdir + energyfile_name
    derivativefile = strdir + derivativefile_name


    logger_energy = construct_logger('energies', energyfile )
    logger_derivative = construct_logger('derivatives', derivativefile )

    logger_derivative.info('%(str1)s %(str2)s', { 'str1' : 'Thermodynamic Integration: Scaling down Interaction type: ', 'str2' : interaction_type })
    if print_energies :
        logger_energy.info('%(str1)s %(str2)s', { 'str1' : 'Thermodynamic Integration: Scaling down Interaction type: ', 'str2' : interaction_type })

    #********************** special setups for the different interaction types ...
    #  1) polariation or electrostatics, create new simulation context for numerical derivative
    if interaction_type == "polarization" or interaction_type == "electrostatic" :   
        # create simulation object for numerical derivative
        TI_system.lambda_derivative = TI_system.numerical_lambda_derivative( TI_system, platform_dx )

        # if this interaction type is polarization and we're using SCF, need another simulation context...
        if interaction_type == "polarization" and use_SCF_lambda_pol == True:
             TI_system.lambda_SCF = TI_system.numerical_lambda_derivative( TI_system, platform_dx )


    # write initial positions
    state = TI_system.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
    position = state.getPositions()
    PDBFile.writeFile(TI_system.simmd.topology, position, open(strdir+'start_drudes.pdb', 'w'))
    # log initial energy
    flag = log_system_energy( TI_system, logger_energy )
    # set up reporters
    TI_system.simmd.reporters = []
   
    # name files nvt/npt ...
    if NPT_simulation :
        name_file = 'md_npt'
    else:
        name_file = 'md_nvt'
    TI_system.simmd.reporters.append(DCDReporter(strdir+name_file+'.dcd', trjsteps ))
    TI_system.simmd.reporters.append(CheckpointReporter(strdir+name_file+'.chk', chksteps ))
    TI_system.simmd.reporters[1].report(TI_system.simmd,state)


    #************************************************
    #    TI sampling:  At this point, the simulation should be setup
    #       and we sample dE_dlambda derivatives for the particular interaction type
    #************************************************

    # we need this outer loop here for repulsive interactions in order to loop over atom shells.  For other interactions, just one iteration
    if interaction_type == "repulsion" :
        loop_val = len(atomshells)
    else:
        loop_val = 1

    for force_index in range(loop_val):    # loop over atom shells if repulsion interaction ....
      # ********** loop over lambda values

      for lambda_i1 in lambda_range:

        # if polarization, don't use lambda =0.0, use lambda=0.01 instead...
        lambda_i = lambda_i1
        if interaction_type == "polarization" and lambda_i1 == 0.0:
            lambda_i = 0.01

        # scale interaction type for this lambda value ...
        if interaction_type == "polarization" :
            # scale polarization in main simulation object for this value of lambda
            flag = simulation_scale_polarization( TI_system, lambda_i )
            # scale polarization in SCF simulation object if we're using it...
            if use_SCF_lambda_pol == True:
                flag = simulation_scale_polarization( TI_system.lambda_SCF, lambda_i )
            # scale polarization in numerical derivative
            TI_system.lambda_derivative.create_numerical_derivative( scalefactor=lambda_i )

        elif interaction_type == "electrostatic" :
            # scale electrostatics in main simulation object for this value of lambda
            flag = simulation_scale_electrostatic( TI_system, lambda_i )
            # scale electrostatics in numerical derivative
            TI_system.lambda_derivative.create_numerical_derivative( scalefactor=lambda_i )
 
        else : 
            # here we're either scaling SAPT_FF to LJ, or scaling off LJ with soft-core.  Either way, analytic derivatives...
            # set Global parameter
            TI_system.simmd.context.setParameter(TI_system.lambda_string[force_index],lambda_i)


        logger_derivative.info('%(str1)s %(lambda)r', { 'str1' : 'derivatives at lambda = ', 'lambda' : lambda_i })
        logger_derivative.info('%(str1)s %(nstep)d %(str2)s' , { 'str1' : 'Equilibrating for  ', 'nstep' : n_equil , 'str2' : ' steps'})
        if print_energies :
            logger_energy.info('%(str1)s %(lambda)r', { 'str1' : 'energies at lambda = ', 'lambda' : lambda_i })
            logger_energy.info('%(str1)s %(nstep)d %(str2)s' , { 'str1' : 'Equilibrating for  ', 'nstep' : n_equil , 'str2' : ' steps'})

        # equilibrate
        TI_system.simmd.step(n_equil)

        #*********************** Sample derivatives for energy ****************************
        for i in range(n_deriv):
            TI_system.simmd.step(n_step)

            if print_energies :
                logger_energy.info( 'Energies from default simulation context' )
                flag = log_system_energy( TI_system, logger_energy )

            # get derivatives, depending on interaction type
            if interaction_type == "polarization" :
                if use_SCF_lambda_pol == True:
                    # use SCF to compute numerical derivative
                    state = TI_system.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
                    # if NPT set box length
                    if NPT_simulation:
                        box = state.getPeriodicBoxVectors()
                        TI_system.lambda_SCF.simmd.context.setPeriodicBoxVectors(box[0], box[1], box[2])
                    # set positions
                    position = state.getPositions()
                    TI_system.lambda_SCF.simmd.context.setPositions(position)
                    # converge Drude positions
                    TI_system.lambda_SCF.simmd.step(1)
                    dH_dlambda = TI_system.lambda_derivative.compute_numerical_derivative( TI_system.lambda_SCF.simmd, print_energies, logger_energy )
                    if print_energies:
                        logger_energy.info( 'Energies from SCF Drude simulation context' )
                        flag=log_system_energy( TI_system.lambda_SCF, logger_energy )
                else:
                    #dE_dlambda evaluated from Thermal Drudes, derivative will not be rigorous (since Hellman-Feynman doesn't hold)
                    dH_dlambda = TI_system.lambda_derivative.compute_numerical_derivative( TI_system.simmd, print_energies, logger_energy )

            elif interaction_type == "electrostatic" :
                dH_dlambda = TI_system.lambda_derivative.compute_numerical_derivative( TI_system.simmd, print_energies, logger_energy )

            else :  # SAPT_FF_LJ and repulsion
                # get analytic derivative
                state = TI_system.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True,getParameterDerivatives=True)
                dp = state.getEnergyParameterDerivatives()
                dH_dlambda = dp[TI_system.lambda_string[force_index]]

            # now store derivative
            logger_derivative.info('%(str1)s  %(i)d %(dH)r', { 'str1': 'step', 'i' : i , 'dH' : dH_dlambda })



    #******************************************************************
    # ********************** end interation over interaction type
    #******************************************************************

    # delete loggers, will create new ones
    for h in list(logger_energy.handlers):
        logger_energy.removeHandler(h)
    del logger_energy

    for h in list(logger_derivative.handlers):
        logger_derivative.removeHandler(h)
    del logger_derivative

    # delete OpenMM objects (need to do this to free up GPU, and store positions for next
    del pdb
    del integrator
    del modeller
    del forcefield
    del system
    del TI_system

exit()
