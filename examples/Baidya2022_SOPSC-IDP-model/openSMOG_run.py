from OpenSMOG import SBM
import sys

#setting MD params
simul_prefix = "Output"
dt = 0.005 #0.010 #ps
collision_rate = 0.05 #ps-1
r_cutoff = 1.6 #nm 
T_in_K = 371.5 #K
T = float(T_in_K)*0.008314 #reduced units RT

sbm = SBM(name=simul_prefix, time_step=dt, collision_rate=collision_rate, r_cutoff=r_cutoff, temperature=T,pbc=True)

#platform="cuda"` or ="HIP" or ="opencl" or ="cpu"
#running using cuda toolkit
sbm.setup_openmm(platform='opencl',GPUindex='default')


sbm.saveFolder("%.2fK_RUN"%float(T_in_K))

sbm_grofile = [x.strip() for x in sys.argv if x.endswith(".gro")][0]
sbm_topfile = [x.strip() for x in sys.argv if x.endswith(".top")][0]
sbm_xmlfile = [x.strip() for x in sys.argv if x.endswith(".xml")][0]


sbm.loadSystem(Grofile=sbm_grofile, Topfile=sbm_topfile, Xmlfile=sbm_xmlfile)

sbm.createSimulation()

trjformat = "dcd"
sbm.createReporters(trajectory=True,trajectoryFormat=trjformat, energies=True, energy_components=True, interval=25000)



#report is for verbose
#mdrun
sbm.run(nsteps=1000000000, report=True, interval=25000)
sbm.simulation.saveState("endfile.state")
sbm.simulation.saveCheckpoint("endfile.chk")

