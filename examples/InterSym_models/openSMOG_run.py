from OpenSMOG import SBM
import sys

#setting MD params
simul_prefix = "Output"
dt = 0.0005 #ps
collision_rate = 1.0 #ps-1
r_cutoff = 3.0 #nm 
T_in_K =120 #K
T = float(T_in_K)*0.008314 #reduced units RT

sbm_CA = SBM(name=simul_prefix, time_step=dt, collision_rate=collision_rate, r_cutoff=r_cutoff, temperature=T,pbc=True)

#platform="cuda"` or ="HIP" or ="opencl" or ="cpu"
#running using cuda toolkit
sbm_CA.setup_openmm(platform='opencl',GPUindex='default')


sbm_CA.saveFolder("output_01")

sbm_CA_grofile = [x.strip() for x in sys.argv if x.endswith(".gro")][0]
sbm_CA_topfile = [x.strip() for x in sys.argv if x.endswith(".top")][0]
sbm_CA_xmlfile = [x.strip() for x in sys.argv if x.endswith(".xml")][0]


sbm_CA.loadSystem(Grofile=sbm_CA_grofile, Topfile=sbm_CA_topfile, Xmlfile=sbm_CA_xmlfile)

sbm_CA.createSimulation()

trjformat = "xtc"
sbm_CA.createReporters(trajectory=True,trajectoryFormat=trjformat, energies=True, energy_components=True, interval=1000)
#(checkpoint=True, checkpointInterval=1000) #not supported in default version



#report is for verbose
#for appending/extending simulations
#sbm_CA.simulation.loadCheckpoint("output_01/Output_checkpoint.chk")      
#sbm_CA.simulation.loadCheckpoint("endfile.chk")
#sbm_CA.simulation.loadState("endfile.state")
#mdrun
sbm_CA.run(nsteps=5000000, report=True, interval=1)
sbm_CA.simulation.saveState("endfile.state")
sbm_CA.simulation.saveCheckpoint("endfile.chk")

