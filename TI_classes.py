from simtk.openmm.app import *
from simtk.openmm import *
import simtk.openmm as mm
from simtk.unit import *
from sys import stdout
import subprocess
import logging
import copy
#******** this is module that goes with sapt force field files to generate exclusions
from sapt_exclusions import *

#*************************************************************************************
# This Module defines a new class and method that we use to run a Thermodynamic Integration
#
#*************************************************************************************



#************************************************************************************************
# this is definition of energy function for aa and na interactions modeled by CustomNonbondedForce
# short-range and VDWs interactions can be scaled with global parameters 'lambda_rep', 'lambda_attrac'
# currently hard-coded in as SAPT-FF functional form
#************************************************************************************************

# this is full repulsive and attractive energy function
def define_energy_function_full( lambda_rep_string , lambda_attrac_string ):
    # lambda_rep scales repulsive part, lambda_attrac scales attractive part
    string='{}*A*exBr - {}*(f6*C6/(r^6) + f8*C8/(r^8) + f10*C10/(r^10) + f12*C12/(r^12));     A=Aex-Ael-Ain-Adh;     Aex=sqrt(Aexch1*Aexch2); Ael=sqrt(Aelec1*Aelec2); Ain=sqrt(Aind1*Aind2); Adh=sqrt(Adhf1*Adhf2);     f12 = f10 - exBr*( (1/39916800)*(Br^11)*(1 + Br/12) );     f10 = f8 - exBr*( (1/362880)*(Br^9)*(1 + Br/10 ) );     f8 = f6 - exBr*( (1/5040)*(Br^7)*(1 + Br/8 ) );     f6 = 1 - exBr*(1 + Br * (1 + (1/2)*Br*(1 + (1/3)*Br*(1 + (1/4)*Br*(1 + (1/5)*Br*(1 + (1/6)*Br ) ) )  ) ) );     exBr = exp(-Br);     Br = B*r;     B=(Bexp1+Bexp2)*Bexp1*Bexp2/(Bexp1^2 + Bexp2^2);     C6=sqrt(C61*C62); C8=sqrt(C81*C82); C10=sqrt(C101*C102); C12=sqrt(C121*C122)'.format(lambda_rep_string, lambda_attrac_string)
    return string

# this is special energy function for scaling repulsive interations.  VDWs has been turned off, and we scale the exponent to smooth out the repulsion
def define_energy_function_rep( lambda_rep_string , lambda_attrac_string ):
    # lambda_rep scales repulsive part, lambda_attrac scales attractive part
    string='{0}*A*exBr - {1}; A=Aex-Ael-Ain-Adh;   Aex=sqrt(Aexch1*Aexch2); Ael=sqrt(Aelec1*Aelec2); Ain=sqrt(Aind1*Aind2); Adh=sqrt(Adhf1*Adhf2);   exBr = exp(-Br);  Br = B*(r+0.2*(1-{0})) ;  B=(Bexp1+Bexp2)*Bexp1*Bexp2/(Bexp1^2 + Bexp2^2);'.format(lambda_rep_string, lambda_attrac_string)
    return string


#**********************************************************************************************
# set up logger
#**********************************************************************************************
def construct_logger( logname, outputfile ):
     logger = logging.getLogger(logname)
     logger.setLevel(logging.INFO)
     fh = logging.FileHandler(outputfile)
     fh.setLevel(logging.INFO)
     logger.addHandler(fh)
    
     return logger 
     

#**********************************************************************************************
# this method logs the total energy and energy components for an input simulation object
#*********************************************************************************************
def log_system_energy( simobject, logger ):
     state = simobject.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
     logger.info(str(state.getKineticEnergy()))
     logger.info(str(state.getPotentialEnergy()))
     for j in range(simobject.system.getNumForces()):
         f = simobject.system.getForce(j)
         logger.info('%(str1)s  %(str2)s', { 'str1': type(f), 'str2' : str(simobject.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()) })

     return True

#*************************************************************************************************
# this class defines an atom-drude pair.  If the atom doesn't have a corresponding Drude oscillator, that portion of the data structure remains empty
#*************************************************************************************************
class atom_drude_pair(object):
      # initialization variables are aindex_g : global index of atom ; dindex_g : global index of drude ; dindex_d : index in DrudeForce class
      # q_a : default charge of atom ; q_d : default charge of Drude ; pol : polarizability  
      def __init__(self, aindex_g , dindex_g , dindex_d , q_a , q_d , pol ):
          self.aindex_g = aindex_g
          self.dindex_g = dindex_g
          self.dindex_d = dindex_d
          self.q_a      = q_a
          self.q_d      = q_d
          self.pol      = pol

      # this returns the static charge of the atom/drude pair
      def static_charge(self):
          q_static = self.q_a + self.q_d
          return q_static



#************************************************
#  this method creates a OpenMM simulation object
#  to be contained within our more abstract simulation
#  objects such as TI_simulation, etc.
#  
#  Input:
#  1) simobject should have an integrator,
#  and system class associated with it. 
#  2) modeller is an OpenMM modeller object from which to get the topology
#  3) platform is simulation platform ('CUDA','CPU','OpenCl', etc.)
#  4) properties is platform properties
#************************************************
def construct_OpenMM_simulation_object( simobject, modeller, platform, properties=False ): 

    # make sure integrator has hard wall on Drudes
    # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
    if isinstance( simobject.integrator, mm.openmm.DrudeLangevinIntegrator ):
         simobject.integrator.setMaxDrudeDistance(0.02)

    # store force class objects
    simobject.nbondedForce = [f for f in [simobject.system.getForce(i) for i in range(simobject.system.getNumForces())] if type(f) == NonbondedForce][0]
    simobject.customNonbondedForce = [f for f in [simobject.system.getForce(i) for i in range(simobject.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
    simobject.custombond = [f for f in [simobject.system.getForce(i) for i in range(simobject.system.getNumForces())] if type(f) == CustomBondForce][0]
    simobject.drudeForce = [f for f in [simobject.system.getForce(i) for i in range(simobject.system.getNumForces())] if type(f) == DrudeForce][0]

    # PME for electrostatics, Cutoff for custom. Probably fine to hard-code these in,
    # as I don't envision ever using anything else...
    simobject.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
    simobject.customNonbondedForce.setNonbondedMethod(NonbondedForce.CutoffPeriodic)

    # create OpenMM simulation object
    if  properties == False:
        simobject.simmd = Simulation(modeller.topology, simobject.system, simobject.integrator, platform)
    else:
        simobject.simmd = Simulation(modeller.topology, simobject.system, simobject.integrator, platform, properties)

    #************************************************
    #         IMPORTANT: generate exclusions for SAPT-FF
    sapt_exclusions = sapt_generate_exclusions(simobject.simmd, simobject.system, modeller.positions)
    #************************************************
 
    # set force groups
    for i in range(simobject.system.getNumForces()):
        f = simobject.system.getForce(i)
        type(f)
        f.setForceGroup(i)

    # reinitialize context, positions after setting exclusions, groups
    simobject.simmd.context.reinitialize()
    simobject.simmd.context.setPositions(modeller.positions)

    return True


#*************************************************
#  Scale polarization of force field
#
#  we need to do two things here.  First we need to 'add back' part of the drude oscillator charge
#  to the parent atom, and then we need to scale the parameters in 'drudeForce'
#
#*************************************************
def simulation_scale_polarization( simulation_object, scalefactor ):
      #*** use local variables here for nbondedForce, drudeForce as corresponding variables
      #*** in simulation_object class could be private "_*"
      nbondedForce = [f for f in [simulation_object.system.getForce(i) for i in range(simulation_object.system.getNumForces())] if type(f) == NonbondedForce][0]
      drudeForce = [f for f in [simulation_object.system.getForce(i) for i in range(simulation_object.system.getNumForces())] if type(f) == DrudeForce][0]

      # first, create the scaled Hamiltonian
      for atom_drude_pair_i in simulation_object.solute_atoms:
           # see if this atom has a Drude oscillator, otherwise we don't have to do anything
           if atom_drude_pair_i.dindex_g > 0:
                 # new scaled charges on atom and Drude
                 q_static = atom_drude_pair_i.static_charge()
                 q_i_scaled = q_static - scalefactor * atom_drude_pair_i.q_d
                 q_d_scaled = scalefactor * atom_drude_pair_i.q_d

                 # assume no LJ interaction, set sigma=1, epsilon=0
                 # Atom
                 nbondedForce.setParticleParameters(atom_drude_pair_i.aindex_g, q_i_scaled, 1.0, 0.0)
                 # Drude
                 nbondedForce.setParticleParameters(atom_drude_pair_i.dindex_g, q_d_scaled, 1.0, 0.0)

                 # now adjust polarizability.
                 # since alpha = q^2/k, alpha_scaled = scalefactor^2 * alpha
                 pol_scaled = scalefactor**2*atom_drude_pair_i.pol
                 # remember that drudeForce.getParticleParameters(i) returns list that looks like
                 # [27, 2, -1, -1, -1, Quantity(value=-1.1478, unit=elementary charge), Quantity(value=0.00195233, unit=nanometer**3), 0.0, 0.0]
                 drudeForce.setParticleParameters( atom_drude_pair_i.dindex_d , atom_drude_pair_i.dindex_g, atom_drude_pair_i.aindex_g , -1 , -1, -1, q_d_scaled , pol_scaled , 0.0 , 0.0 )
      # now update new parameters in context
      nbondedForce.updateParametersInContext(simulation_object.simmd.context)
      drudeForce.updateParametersInContext(simulation_object.simmd.context)

      return True


#*************************************************
#  Scale (static) electrostatic interactions of simulation
#
#  this method adjusts the charges on solute atoms within the simulation
#  note, we should already have turned polarization off, by scaling drudes to zero!
#
#*************************************************
def simulation_scale_electrostatic( simulation_object, scalefactor ):

      #*** use local variables here for nbondedForce as corresponding variable in simulation_object class could be private "_*"
      nbondedForce = [f for f in [simulation_object.system.getForce(i) for i in range(simulation_object.system.getNumForces())] if type(f) == NonbondedForce][0]
      for atom_drude_pair_i in simulation_object.solute_atoms:
           q_static = atom_drude_pair_i.static_charge()
           q_static_scaled = scalefactor * q_static
           nbondedForce.setParticleParameters(atom_drude_pair_i.aindex_g, q_static_scaled, 1.0, 0.0)
      # now update new parameters in context
      nbondedForce.updateParametersInContext(simulation_object.simmd.context)

      return True


#*****************************************************
# This class controls the Thermodynamic Integration
#
# we make use of some of the code format in Chodera's alchemy code, found at
# https://github.com/choderalab/alchemy/blob/master/alchemy/alchemy.py
# we take the nomenclature 'aa' for alchemical/alchemical interaction, 'na' for non-alchemical/alchemical interaction, etc.
# this is the same nomenclature as referenced code
# we assume that a 'solute' molecules is the alchemical species, so we use these names interchangably
#******************************************************
class TI(object):
    def __init__(self, solutename, atomshells, system, modeller, forcefield, integrator, platform, properties, use_SCF_lambda_pol, interaction_type, NPT_simulation):
          self.solutename = solutename
          self.atomshells = atomshells
          self.system = system
          self.modeller = modeller
          self.forcefield = forcefield
          self.integrator = integrator
          self.platform = platform
          self.properties = properties
          self.use_SCF_lambda_pol =  use_SCF_lambda_pol
          self.interaction_type = interaction_type
          self.NPT_simulation = NPT_simulation


          #******   create simulation object
          #******   self.simmd will be created.
          # also, this will assign the following attributes
          # self.nbondedForce, self.customNonbondedForce, self.custombond, self.drudeForce
          flag = construct_OpenMM_simulation_object( self , self.modeller , self.platform , self.properties )
          
          # get Cutoff: Be consistent for all defined force classes...
          self.Cutoff = self.customNonbondedForce.getCutoffDistance()

          # setup Custom energy functions.  If scaling VDWs, we only need one of these.  If scaling repulsion, we need one for
          # each shell of atoms
          self.setup_energy_functions_alchemical( self.atomshells , self.interaction_type )

          # this stores indices of all atoms and drudes in solute molecule
          # note that this list is bigger than self.solute_atoms list as the latter only runs over 'real atoms'
          self.solute_total_atoms_list=[]

          # this is data structure that contains atom data for solute.  We will fill this list with 'atom_drude_pair' objects
          self.solute_atoms=[]
          # fill this data structure
          flag = self.create_solute_atoms_data()
          # if return -1, couldn't find solute!
          if flag < 0 :
             print( "couldn't find solute ", solutename, " in pdb file, are you sure this is correct???" )
             sys.exit()

          # this stores indices of atoms listed by atomshell
          if interaction_type == "repulsion":
              self.solute_atomshells_list=[]
              self.create_solute_atomshells_data()    


          # create set of non-alchemical (rest of system) atoms to use in OpenMM interaction groups.
          self.alchemical_atomset = set( self.solute_total_atoms_list )
          self.all_atomset = set(range(system.getNumParticles()))  # all atoms
          self.nonalchemical_atomset = self.all_atomset.difference(self.alchemical_atomset)         

 
          #***********************************************
          #  Create CustomNonbondedforce interactions for alchemical/non-alchemical interactions
          #      if we're scaling repulsion, we need a different force class for each atom shell
          #      otherwise, we only need one force class for the alchemical solute molecule
          #***********************************************
          self.customNonbondedForce_alchemical=[]
          self.alchemical_force_index=[]
          if interaction_type == "repulsion":
              # create unique CustomNonbondedforce class for every shell of atoms of solute
              # this shell will interact with all non-solute atoms
              for force_index in range(len(self.atomshells)):
                  self.create_customNonbondedForce_alchemical( force_index , self.solute_atomshells_list[force_index] )
          else:
              force_index=0
              # only one new CustomNonbondedforce class
              self.create_customNonbondedForce_alchemical( force_index , self.solute_total_atoms_list ) 


          #*********************************************
          # now remove alchemical interactions from original CustomNonbondedforce
          #
          #*********************************************
          # turn off alchemical-alchemical (aa) and alchemical-nonalchemical interactions in original force class
          #for index in self.solute_total_atoms_list:
              # set CustomNonbonded interactions to zero
              # these are Aexch, Aelec, Aind, Adhf, B, C6 , C8 , C10 , C12
              #self.customNonbondedForce.setParticleParameters(index, [ 0.0 , 0.0 , 0.0 , 0.0 , 10.0 , 0.0 , 0.0 , 0.0 , 0.0 ] )

          # don't zero parameters as in commented above code, because we want to keep intra-molecular alchemical interactions within this force class ...
          for i_index in self.alchemical_atomset:
              for j_index in self.nonalchemical_atomset:
                  self.customNonbondedForce.addExclusion(i_index,j_index)
                  if interaction_type == "repulsion":
                      # need to add exceptions for electrostatics, even though charges are zero, have numerical problems in electrostatics when repulsion is scaled down
                      self.nbondedForce.addException(i_index,j_index,0,1,0,True)

          # we need to reset force groups after adding new customNonbondedForce_alchemical force object
          for i in range(self.system.getNumForces()):
              f = self.system.getForce(i)
              type(f)
              f.setForceGroup(i)


          # lastly, reinitialize context to incorporate new alchemical force class
          state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
          positions = state.getPositions()
          self.simmd.context.reinitialize()
          self.simmd.context.setPositions(positions)



    #***********************************************
    #  Create CustomNonbondedforce interactions for alchemical/non-alchemical interactions
    #  the input atom list defines the group of solute atoms that we put in this force class
    #***********************************************
    def create_customNonbondedForce_alchemical(self, force_index , solute_atom_list ):
          # set up new CustomNonbondedforce with scaling factors for alchemical/alchemical and non-alchemical/alchemical interactions.
          self.customNonbondedForce_alchemical.append(openmm.CustomNonbondedForce( self.energy_function_alchemical[force_index] ))
          # add particle parameters
          self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("Aexch")
          self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("Aelec")
          self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("Aind")
          self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("Adhf")
          self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("Bexp")

          # add VDWs parameters if not repulsion
          if self.interaction_type != "repulsion":
              self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("C6")
              self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("C8")
              self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("C10")
              self.customNonbondedForce_alchemical[force_index].addPerParticleParameter("C12")

          # set methods
          self.customNonbondedForce_alchemical[force_index].setNonbondedMethod(NonbondedForce.CutoffPeriodic)
          self.customNonbondedForce_alchemical[force_index].setCutoffDistance(self.Cutoff)
          # I don't think LRC works with scaled interactions...
          #self.customNonbondedForce_alchemical[force_index].setUseLongRangeCorrection(True)

          # add force to system
          self.alchemical_force_index.append(self.system.addForce(self.customNonbondedForce_alchemical[force_index]))

          # Distribute Particles to this new CustomNonbondedforce
          for index in range(self.customNonbondedForce.getNumParticles()):
              [ Aexch , Aelec , Aind , Adhf , Bexp , C6 , C8 , C10 , C12 ] =  self.customNonbondedForce.getParticleParameters(index)
              # if repulsion, only add Born-Mayer parameters...
              if self.interaction_type == "repulsion":
                  self.customNonbondedForce_alchemical[force_index].addParticle([ Aexch , Aelec , Aind , Adhf , Bexp ])
              else :
                  self.customNonbondedForce_alchemical[force_index].addParticle([ Aexch , Aelec , Aind , Adhf , Bexp , C6 , C8 , C10 , C12 ])

          # add interaction between specified solute atoms and rest of system
          self.alchemical_local = set( solute_atom_list )
          self.customNonbondedForce_alchemical[force_index].addInteractionGroup( self.alchemical_local , self.nonalchemical_atomset )

          # add scale factors for force class as Global parameters
          self.customNonbondedForce_alchemical[force_index].addGlobalParameter( self.lambda_rep_string[force_index], self.lambda_rep[force_index] )
          self.customNonbondedForce_alchemical[force_index].addGlobalParameter( self.lambda_attrac_string[force_index], self.lambda_attrac[force_index] )
          # add energy derivatives for these scale factors
          self.customNonbondedForce_alchemical[force_index].addEnergyParameterDerivative(self.lambda_rep_string[force_index])
          self.customNonbondedForce_alchemical[force_index].addEnergyParameterDerivative(self.lambda_attrac_string[force_index])

        

    
    #**********************************
    # this function sets up energy expressions for the custom nonbonded interaction
    # if we're scaling repulsion, we need as many of these as there are atomshells
    def setup_energy_functions_alchemical( self, atomshells , interaction_type ):
        # repulsion vs attraction scaling, see how many energy expressions we need to set up
        if interaction_type == "repulsion" :
           numberterms = len(atomshells)           
        else:
           numberterms = 1

        # initialize lists
        self.lambda_rep_string=[]
        self.lambda_attrac_string=[]
        self.energy_function_alchemical=[]
        self.lambda_rep=[]
        self.lambda_attrac=[]
        for i in range(numberterms):
            string1 = 'lambda_rep' + str(i)
            string2 = 'lambda_attrac' + str(i)
            self.lambda_rep_string.append( string1 )
            self.lambda_attrac_string.append( string2 )
            # energy function depends on whether we're scaling repulsive interactions
            if interaction_type == "repulsion" :          
                self.energy_function_alchemical.append( define_energy_function_rep( self.lambda_rep_string[i] , self.lambda_attrac_string[i] ) )
            else:
                self.energy_function_alchemical.append( define_energy_function_full( self.lambda_rep_string[i] , self.lambda_attrac_string[i] ) )

            self.lambda_rep.append( 1.0 )
            self.lambda_attrac.append( 1.0 )



    #**********************************
    # this function fills in the solute_atoms_data list by creating
    # 'atom_drude_pair' objects for each atom.
    #**********************************
    def create_solute_atoms_data(self):
          flag=-1
          # store indices of drudes so we avoid adding them to data structure
          solute_drude_indices=[]
          # get atom indices of solute
          for res in self.simmd.topology.residues():
              if res.name == self.solutename:
                  flag=1
                  for i in range(len(res._atoms)):
                      # global index of atom
                      index = res._atoms[i].index

                      # add to solute_total_atoms_list no matter what (even if Drude oscillator)
                      self.solute_total_atoms_list.append(index)
                      
                      # make sure this isn't a Drude oscillator
                      if index not in solute_drude_indices:                          
                          #solute_atom_indices.append(index)
                          # get charge of atom
                          (q_i, sig, eps) = self.nbondedForce.getParticleParameters(index)

                          drude_flag=0
                          # see if this atom has drude oscillator attached
                          # note that self.drudeForce.getParticleParameters(i) returns list that looks like
                          # [27, 2, -1, -1, -1, Quantity(value=-1.1478, unit=elementary charge), Quantity(value=0.00195233, unit=nanometer**3), 0.0, 0.0]                          
                          for i in range(self.drudeForce.getNumParticles()):
                              if self.drudeForce.getParticleParameters(i)[1] == index:
                                  drude_flag=1
                                  dindex_d = i
                                  dindex_g = self.drudeForce.getParticleParameters(i)[0]
                                  # get charge of Drude
                                  (q_d, sig, eps) = self.nbondedForce.getParticleParameters(dindex_g)  
                                  # get polarizability
                                  pol = self.drudeForce.getParticleParameters(i)[6]._value
                                  break
                          #*****************
                          #  create 'atom_drude_pair' data structure for atom with or without drude oscillator
                          #*****************
                          if drude_flag == 1: 
                              atom_drude_pair_i = atom_drude_pair( index , dindex_g , dindex_d , q_i , q_d , pol )
                              # add drude oscillator to solute_drude_indices list
                              solute_drude_indices.append( dindex_g )
                          else:
                              atom_drude_pair_i = atom_drude_pair( index , -1 , -1 , q_i , 0.0*elementary_charge , 0.0 )      
           
 
                          #*******************
                          # append atom object to self.solute_atoms list
                          #*******************
                          self.solute_atoms.append( atom_drude_pair_i )
          
          return flag


    #**********************************
    # this function finds indices of atoms
    # in each atom shell
    #**********************************
    def create_solute_atomshells_data(self):
          # first create name/index list pair for residue
          atomname=[]
          atomindex=[]
          for res in self.simmd.topology.residues():
              if res.name == self.solutename:
                 for i in range(len(res._atoms)):
                      # global index of atom
                      atomindex.append(res._atoms[i].index)
                      # name of atom
                      atomname.append(res._atoms[i].name)
                     

          # now loop over atom shells and add atom indices to list
          for i in range(len(self.atomshells)):
              # initialize new list for this shell
              shelllist=[]
              for j in range(len(self.atomshells[i])):                      
                  i_index = atomname.index( self.atomshells[i][j] )              
                  shelllist.append( atomindex[i_index] )
              self.solute_atomshells_list.append( shelllist )



    #**********************************
    # Note: this class should only be utilized with lambda scaling of
    # polarization and electrostatic interactions.  Other interaction types
    # (e.g. repulsion/dispersion), are scaled using analytic derivatives computed
    # using global parameter derivatives as implemented in CustomNonbondedForce
    #
    # the differential used to compute numerical derivatives is set in
    # __init__ to dlambda, which can be modified as wanted
    #
    # input in __init__ :  platform_dx determines the platform to be used for the new simulation object
    #  probably a good choice is 'CPU', since this object will be called much less frequently...
    #**********************************     
    # this class contains data structures necessarity for computing numerical lambda derivatives from thermodynamic integration
    class numerical_lambda_derivative(object):
         # input to init is outerclass (TI_system) to access simulation data structures, and platform_dx which is platform to use for simulation object
         def __init__(self, outerclass, platform_dx ):

             # copy interaction type
             self.interaction_type = outerclass.interaction_type
             # copy NPT_simulation info
             self.NPT_simulation = outerclass.NPT_simulation

             #*****************************************
             #   controls size of differential for computing numerical derivatives
             if self.interaction_type == "polarization":
                 self.dlambda = 0.05
             elif self.interaction_type == "electrostatic":
                 self.dlambda = 0.01
             else:
                 print( "Couldn't recognize interaction_type in numerical_lambda_derivative.__init__ ")
                 sys.exit()
             #*****************************************

             # copy solute atoms datastructure
             self.solute_atoms = outerclass.solute_atoms
             # create new integrator for this simulation object

             if self.interaction_type == "polarization" and outerclass.use_SCF_lambda_pol == True:
                  self.integrator = DrudeSCFIntegrator(0.00001*picoseconds)
             else:
                  self.integrator = copy.deepcopy( outerclass.integrator )
             
             self.system = outerclass.forcefield.createSystem(outerclass.modeller.topology, nonbondedCutoff=outerclass.Cutoff, constraints=None, rigidWater=True)

             #******   create simulation object
             #******   self.simmd will be created.
             # also, this will assign the following attributes
             # self.nbondedForce, self.customNonbondedForce, self.custombond, self.drudeForce
             flag = construct_OpenMM_simulation_object( self , outerclass.modeller , platform_dx )

             # get positions from outerclass
             state = outerclass.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
             positions = state.getPositions()
             self.simmd.context.setPositions(positions)


         # this method sets up numerical derivative
         def create_numerical_derivative(self, scalefactor=1.0):
             if self.interaction_type == "polarization" :
                # change simulation object to utilize scalefactor_local = scalefactor + dlambda
                self.scalefactor_local = scalefactor + self.dlambda
                flag = simulation_scale_polarization( self, self.scalefactor_local )
                if not flag:
                    print( 'error in simulation_scale_polarization')
                    sys.exit()

             elif self.interaction_type == "electrostatic" :
                # change simulation object to utilize scalefactor_local = scalefactor + dlambda
                self.scalefactor_local = scalefactor + self.dlambda
                flag = simulation_scale_electrostatic( self, self.scalefactor_local )
                if not flag:
                    print( 'error in simulation_scale_electrostatic')
                    sys.exit()


         #*************************************************
         # this method computes the numerical derivative dE/dlambda for the specified interaction type
         # input `simmd' object should be from main simulation, to get positions, energies
         #*************************************************
         def compute_numerical_derivative(self, simmd, print_energy, logger ): 
             state = simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)   
             # if NPT set box length
             if self.NPT_simulation:
                 box = state.getPeriodicBoxVectors()
                 self.simmd.context.setPeriodicBoxVectors(box[0], box[1], box[2])

             # set positions and get energies from input state
             position = state.getPositions()
             self.simmd.context.setPositions(position)
             state2 = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)

             if print_energy:
                 logger.info('Energies from perturbed (numerical derivative) simulation context')
                 flag = log_system_energy( self, logger )

             if self.interaction_type == "polarization" :
                 # polarization derivative
                 flag=0
                 # get electrostatic + polarization energy from actual force field
                 for j in range(self.system.getNumForces()):
                     f = self.system.getForce(j)
                     if isinstance(f, mm.openmm.NonbondedForce):
                         Elec0 = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         Elec1 = self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         flag+=1
                     elif isinstance(f, mm.openmm.DrudeForce):
                         Drude0 = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         Drude1 = self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         flag+=1
                 if flag != 2:
                     print("couldn't find NonbondedForce and DrudeForce force objects in compute_numerical_derivative!")
                     sys.exit()
                 # take derivative
                 dH_dlambda = ( Elec1 + Drude1 - ( Elec0 + Drude0 ) ) / self.dlambda

                 #print( '  test ' )
                 #print( 'Elec0 , Drude0 ' , Elec0 , Drude0 )
                 #print( 'Elec1 , Drude1 ' , Elec1 , Drude1 )

             elif self.interaction_type == "electrostatic" :
                 # electrostatic derivative
                 flag=0
                 # get electrostatic energy from actual force field
                 for j in range(self.system.getNumForces()):
                     f = self.system.getForce(j)
                     if isinstance(f, mm.openmm.NonbondedForce):
                         Elec0 = simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         Elec1 = self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()._value
                         flag+=1
                 if flag != 1:
                     print("couldn't find NonbondedForce force object in compute_numerical_derivative!")
                     sys.exit()
                 # take derivative
                 dH_dlambda = ( Elec1 - Elec0  ) / self.dlambda

             return dH_dlambda  

              

