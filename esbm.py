#!/usr/bin/env python

"""
	Enhanced SBN (esbm)
	Copyright (C) <2024>  <Digvijay Lalwani Prakash>

	A toolkit for generating input files for performing Molecular Dynamics
	Simulations (MD) of Coarse-Grain Structure Based Models (CG-SBM) on 
	GROMACS (4.5.4/5.1.4) and/or OpenSMOG (v1.1.1)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

	To add a new model, 1) add a new class (to a new .py file or to topology.py)
	and inherit Topology from topology.py. 2) See available functions in Class 
	Topology and re-write those which require changes based on your model.
	3) Predefine available arguments in (esbm.py).

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
	<http://www.gnu.org/licenses/>.

    A copy of the GNU General Public License is included along with
	this program.

usage: python esbm.py --help
(Author: Digvijay L. Prakash)
"""

import argparse
import numpy as np
from typing import NamedTuple, Dict
from pathlib import Path
from PDB_IO import PDB_IO,Nucl_Data,Prot_Data
from topology import *

class Options(Dict):
	sopsc=False
	uniqtype = False
	btparams=False
	mjmap=False
	btmap=False
	dswap=False
	interactions="interactions.dat"
	hphobic=False
	dsb=False

class Constants(Dict):
	Kb_prot = 200.0
	Ka_prot = 40.0
	Kd_prot = {"bb":1.0,"sc":Ka_prot,"mf":2.0}
	Kr_prot = 1.0
	Kb_nucl= 200				
	Ka_nucl = 40				
	Kd_nucl = {"bb":0.7,"sc":0.5,"mf":1.0}				
	Kr_nucl = 1.0
	Kboltz = 8.314462618E-3 #KJ mol-1 nm-1
	caltoj=4.184 #kcal mol-1 A-2 to kcal mol-1 A-2
	permol = 6.022E+23          #n/mol  		#Avogadro's number

class ContactMap(Dict):
	cutoff = 4.5 	#A
	type = 1 		#all-atom mapped to CG
	scale = 1.0 
	func = 2 		# LJ 10-12
	W = False 		#Equal weights 
	file = str()	# no cmap file
	scsc_custom=False

class Charge(Dict):
	CA = False
	CB = False
	P = False
	debye = False
	dielec = 78
	iconc = 0.1 #M L-1               
	irad = 1.4 #nm (for NaCl)
	debye_temp = 298  #K
	inv_dl = 0
	Kboltz = 8.314462618E-3 #KJ mol-1 nm-1
	caltoj=4.184 #kcal mol-1 A-2 to kcal mol-1 A-2
	permol = 6.022E+23          #n/mol  		#Avogadro's number

class ModelDir:
	def __init__(self,file) -> str:
		self.path = "/".join(str(Path(__file__)).split("/")[:-1]+["models"]+file.split("/"))
		return 

	def copy2(self,copyfile):
		with open(copyfile,"w+") as fout:
			fout.write(open(self.path).read())
		return 1

def main():
	
	""" loading arguments here """
	parser = argparse.ArgumentParser(description="Generate GROMACS and OPTIM potential files for Protein + Nucleic Acids enhanced SBM models.")
	
	#Predefined Models
	parser.add_argument("--clementi2000","-clementi2000","--calpha_go2000","-calpha_go2000",action="store_true",help="Clementi et. al. 2000 CA-only model")
	parser.add_argument("--azia2009","-azia2009",action="store_true",help="Azia 2009 CB-CA + Debye-Huckel model")
	parser.add_argument("--pal2019","-pal2019","--levy2019","-levy2019",action="store_true",help="Pal & Levy 2019 Protein CB-CA & RNA/DNA P-S-B model")
	parser.add_argument("--reddy2017","-reddy2017","--sopsc2017","-sopsc2017",action="store_true",help="Reddy. 2017 SOP-SC CA-CB")
	parser.add_argument("--baidya2022","-baidya2022","--sopsc_idp","-sopsc_idp",action="store_true",help="SOP-SC-IDP CA-CB")
	#input options for protein
	parser.add_argument("--CA_rad","-CA_rad",type=float, help="User defined radius for C-alpha (same for all beads) in Angstrom. Default: 4.0A")
	parser.add_argument("--CA_com","-CA_com",action='store_true',help="Place C-alpha at COM of backbone. Default: False")
	parser.add_argument("--CB_rad","-CB_rad",type=float, help="User defined radius for C-beta (same for all beads) in Angstrom for prot_cg 2. Default: Statistically Derived for each AA-residue")
	parser.add_argument('--CB_radii',"-CB_radii",action='store_true', help='User defined C-beta radii from radii.dat (AA-3-letter-code       radius-in-Angsrtom)')
	parser.add_argument("--Kb_prot","-Kb_prot","--Kb","-Kb", help="User defined force constant K_bond for Proteins")
	parser.add_argument("--Ka_prot","-Ka_prot","--Ka","-Ka", help="User defined force constant K_angle for Proteins")
	parser.add_argument("--Kd_bb_prot","-Kd_bb_prot","--Kd","-Kd", help="User defined force constant K_dihedral for Proteins")
	parser.add_argument("--Kd_sc_prot","-Kd_sc_prot","--Kd_chiral","-Kd_chiral", help="User defined force constant K_dihedral for Proteins")
	parser.add_argument("--mulfac_prot","-mulfac_prot", help="User defined Multiplicity scale factor of K_dihedral/mulfac_prot for Proteins")
	parser.add_argument("--CB_chiral","-CB_chiral",action='store_true',help="Improper dihedral for CB sidechain chirality. Default: False")
	parser.add_argument("--uniqtype","-uniqtype",action="store_true",help="Each atom has unique atom type (only use for large systems)")
	parser.add_argument("--bfunc","-bfunc",help="Bond function 1: harnomic, 7: FENE. Default: 1 (Harmonic)")
	#native  determining contacts parameters
	parser.add_argument("--cutoff","-cutoff",type=float,help="User defined Cut-off (in Angstrom) for contact-map generation. Default: 4.5A")
	parser.add_argument("--cutofftype","-cutofftype",type=int,help="-1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1")
	parser.add_argument("--W_cont","-W_cont",action="store_true",help="Weight (and normalize) CG contacts based on all atom contacts")
	parser.add_argument("--cmap","-cmap",help="User defined cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)")
	parser.add_argument("--scaling","-scaling", help="User defined scaling for mapping to all-atom contact-map.")
	parser.add_argument("--contfunc","-contfunc",type=int,help="1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss  . Default: 2")
	
	#atom type 1: CA only. 2: Ca+Cb
	parser.add_argument("--prot_cg", "-prot_cg", type=int, help="Level of Amino-acid coarse-graining 1 for CA-only, 2 for CA+CB. Dafault: 2 (CA+CB)")

	#CB position #if prot_cg = 2
	parser.add_argument("--CB_com","-CB_com", action='store_true', default=False,help='Put C-beta at side-chain COM (no hydrogens). Default: False')
	parser.add_argument("--CB_far", "-CB_far", action='store_true', help="Place C-beta on farthest non-hydrogen atom. Default: False")

	#
	parser.add_argument("--dsb", "-dsb",action='store_true', help="Use desolvation barrier potential for contacts. Default: False")
	parser.add_argument("--native_ca","-native_ca", help='Native file with only C-alphas. Just grep pdb. ')
	
	#input
	parser.add_argument("--aa_pdb","-aa_pdb", help='User input all-atom pdbfile/gro/mmCIF e.g. 1qys.pdb')
	parser.add_argument("--cg_pdb","-cg_pdb", help='User input coarse grained pdbfile')

	#output
	parser.add_argument("--grotop","-grotop",help='Gromacs topology file output name (tool adds prefix nucl_  and prot_ for independednt file). Default: gromacs.top')
	parser.add_argument("--pdbgro","-pdbgro", help='Name for output .gro file.(tool adds prefix nucl_  and prot_ for independednt file). Default: gromacs.gro')
	parser.add_argument("--smogxml","-smogxml", help='Name for output .xml (openSMOG) file.(tool adds prefix nucl_  and prot_ for independednt file). Default: opensmog.xml (and opensmog.top)')
	parser.add_argument("--opensmog", "-opensmog",action='store_true', help="Generate files ,xml and .top files for openSMOG. Default: False")

	#file parameters
	parser.add_argument("--CB_gly","--CB_GLY","-CB_gly","-CB_GLY",action='store_true',default=False,help='Add C-beta for glycine (pdb-file must have H-atoms). Default: Flase ')
	parser.add_argument('--btparams',"-btparams", action='store_true', help='Use Betancourt-Thirumalai interaction matrix.')
	parser.add_argument('--mjparams',"-mjparams", action='store_true', help='Use Miyazawa-Jernighan interaction matrix.')

	#For Nucleotide
	#radius for P,B,S
	parser.add_argument("--P_rad","-P_rad", help="User defined radius for Backbone Phosphate bead. Default=3.7A")
	parser.add_argument("--S_rad","-S_rad", help="User defined radius for Backbone Sugar bead. Default=3.7A")
	parser.add_argument("--Bpu_rad","-Bpu_rad", help="User defined radius for N-Base Purine bead. Default=1.5A")
	parser.add_argument("--Bpy_rad","-Bpy_rad", help="User defined radius for N-Base Pyrimidine bead. Default=1.5A")
	
	#atom type 1: P only. 3: P-S-B. 5: P-S-3B
	parser.add_argument("--nucl_cg", "-nucl_cg", type=int, help="Level of Amino-acid coarse-graining 1 for P-only, 3 for P-S-B, 5 for P-S-3B. Dafault: 3 (P-S-B)")

	#force constants
	parser.add_argument("--Kb_nucl","-Kb_nucl","--nKb","-nKb", help="User defined force constant K_bond for RNA/DNA")
	parser.add_argument("--Ka_nucl","-Ka_nucl","--nKa","-nKa", help="User defined force constant K_angle for RNA/DNA. Default=20")
	parser.add_argument("--Kd_sc_nucl","-Kd_sc_nucl","--nKd","-nKd", help="User defined force constant K_dihedral for Bi-Si-Si+1-Bi+1. Default=0.5")
	parser.add_argument("--Kd_bb_nucl","-Kd_bb_nucl","--P_nKd","-P_nKd", help="User defined force constant K_dihedral for Backbone Pi-Pi+1-Pi+2-Pi+3. Default=0.7")
	parser.add_argument("--P_stretch","-P_stretch",help="Stretch the backbone dihedral to 180 degrees. Default = Use native  backbone dihedral")
	parser.add_argument("--mulfac_nucl","-mulfac_nucl", help="User defined Multiplicity scale factor of K_dihedral for Nucleic Acids")

	#positions
	parser.add_argument("--Bpu_pos","-Bpu_pos", help="Put input atom of Purine [N1,C2,H2-N2,N3,C4,C5,C6,O6-N6,N7,C8,N9,COM] as position of B. Default=COM(Center_of_Mass)")
	parser.add_argument("--Bpy_pos","-Bpy_pos", help="Put input atom of Pyrimidine [N1,C2,O2,N3,C4,O4-N4,C5,C6,COM] as position of B. Default=COM(Center_of_Mass)")
	parser.add_argument("--S_pos"  ,"-S_pos"  , help="Put input atom of Sugar [C1',C2',C3',C4',C5',H2'-O2',O3',O4',O5',COM] as position of S. Default=COM(Center_of_Mass)")
	parser.add_argument("--P_pos"  ,"-P_pos"  , help="Put input atom of Phosphate [P,OP1,OP2,O5',COM] group as position of P. Default=COM(Center_of_Mass)")
	
	#common
	parser.add_argument("--pistacklen", help="pi-pi stacking length. Default=3.6A")

	#electrostatic
	parser.add_argument("--debye","-debye",action='store_true', help="Use Debye-Huckel electrostatic term.")
	parser.add_argument("--debye_temp","-debye_temp", help="Temperature for Debye length calculation. Default = 298K")
	parser.add_argument("--debye_length","-debye_length", help="Debye length. in (A)")
	parser.add_argument("--CA_charge","-CA_charge", action='store_true', default=False, help='Put charges on CA for K,L,H,D,E. Default: False')
	parser.add_argument("--CB_charge","-CB_charge", action='store_true', default=False, help='Put charges on CB for K,L,H,D,E. Default: False')
	parser.add_argument("--P_charge","-P_charge", action='store_true', default=False, help='Negative charge on Phosphate bead. Default: False')
	parser.add_argument("--iconc","-iconc", help="Solvant ion conc.(N) for Debye length calcluation. Default=0.1M")  
	parser.add_argument("--irad","-irad", help="Solvant ion rad for Debye length calcluation. Default=1.4A")  
	parser.add_argument("--dielec","-dielec", help="Dielectric constant of solvant. Default=70")
	
	#disabled for now
	parser.add_argument('--hpstrength',"-hpstrength",help='Strength with which hydrophobic contacts interact.')
	parser.add_argument('--ext_conmap',"-ext_conmap",help='External contact map in format chain res chain res')
	parser.add_argument("--interaction","-interaction",action='store_true', default=False, help='User defined interactions in file interactions.dat.')
	parser.add_argument("--dswap","-dswap", action='store_true', default=False, help='For domain swapping runs. Symmetrised SBM is generated.')
	parser.add_argument('--hphobic',"-hphobic",action='store_true',help='Generate hydrophobic contacts.')
	parser.add_argument('--hpdist', "-hpdist", help='Equilibrium distance for hydrophobic contacts.')

	#extras
	parser.add_argument("--interface","-interface", action='store_true', default=False, help='Takes input for Nucleiotide_Protein interface from file nucpro_interface.input.')
	parser.add_argument("--custom_nuc","-custom_nuc", help='Use custom non native DNA/RNA structure Eg.: polyT.pdb. Default: Use from native structure')
	parser.add_argument("--control", action='store_true', help='Use the native system as control. Use DNA/RNA bound to native protein site. --custom_nuc will be disabled. Default: False (Move DNA/RNA away from native binding site)')
	#exclusion volume
	parser.add_argument("--excl_rule",help="Use 1: Geometric mean. 2: Arithmatic mean")
	parser.add_argument("--Kr_prot", help="Krepulsion. Default=1.0")
	parser.add_argument("--Kr_nucl", help="Krepulsion. Default=1.0")

	#presets 
	args = parser.parse_args()

	#defualt potoein-NA parameters
	interface = False
	custom_nuc = False
	control_run = False


	#Set default parameters for proteins
	#For preteins
	opt = Options()
	fconst = Constants()
	CGlevel = {"prot":2,"nucl":3}
	contmap = ContactMap()
	rad = dict()	 
	bond_function = 1
	rad["CA"] = 1.9
	rad["CB"] = 1.5
	CA_com = False
	CB_far=False
	CB_com=False
	CB_chiral=False
	#Set default parameters for nucleotides
	rad["P"] = 3.7					#A
	rad["S"] = 3.7					#A
	rad["Bpy"] = 1.5				#A
	rad["Bpu"] = 1.5				#A
	rad["stack"] = 3.6					#A
	P_Stretch = False

	charge = Charge()
	charge.CA = False
	charge.CB = False
	charge.P = False

	#default position
	nucl_pos = dict()
	nucl_pos["Bpu"] = "COM"		#Center of Mass for purine
	nucl_pos["Bpy"] = "COM"		#Center of Mass for pyrimidine
	nucl_pos["P"] = "COM"			#Center of Mass for phosphate group
	nucl_pos["S"] = "COM"			#Center of Mass for sugar

	"""" Defining inputs for preset models """

	if args.clementi2000:
		print (">>> Using Clementi et. al. 2000 CA-only model. 10.1006/jmbi.2000.3693")
		assert args.aa_pdb, "Error no pdb input --aa_pdb"
		#fixed params can't be overwritten
		args.prot_cg = 1	# CA_only	
		args.nucl_cg = 0	# No RNA/DNA
		rad["CA"] = 2.0	# 4.0 A excl vol rad
		contmap.W = 1		# not weighted 
		contmap.cutoff = 4.5	# 4.5 A
		contmap.cutofftype = 1	# all-atom contacts mapped to CG
		contmap.contfunc = 2	# LJ 10-12

	if args.pal2019:
		print (">>> Using Pal & Levy 2019 model. 10.1371/journal.pcbi.1006768")
		assert args.aa_pdb, "Error no pdb input --aa_pdb."
		args.prot_cg = 2	# CB-CA
		args.nucl_cg = 3	# P-S-B
		args.CA_rad = 1.9	# 3.8 A excl vol rad
		args.CB_rad = 1.5	# 3.0 A excl vol rad
		args.CB_far = True	# CB at farthest SC atom 
		args.CB_chiral = True	# improp dihed for CAi-1 CAi+1 CAi CBi
		args.CB_charge = True	# Charge on CB
		args.P_charge = True	#charge on P
		args.excl_rule = 2		# Excl volume Arith. Mean
		args.mulfac_prot = 1.0	#factor to divide 3 multiplicity dihed term
		args.W_cont = 1			# contact weight
		args.cutoff = 4.5		# cutoff
		args.cutofftype = 1		# Calculate from all-atom structure
		args.contfunc = 2		# Use LJ 10-12 pairs
		args.debye = True		# Use DH-electrostatics
		args.dielec = 70		# dielectric constant
		args.iconc = 0.01		# concentration

	if args.azia2009:
		print (">>> Using Azia & Levy 2009 CA-CB model. 10.1006/jmbi.2000.3693")
		args.uniqtype = True
		args.Kr_prot = 0.7**12

	if args.reddy2017:
		print (">>> Using Reddy & Thirumalai 2017 SOP-SCP model. 10.1021/acs.jpcb.6b13100")
		args.prot_cg = 2
		args.uncl_cg = 3
		args.bfunc = 8
		args.cutoff = 8.0
		args.cutofftype = 2
		args.contfunc = 1
		args.excl_rule = 2
		args.btparams = True
		opt.sopsc = True
		args.CB_charge = True
		args.CB_gly = True
		args.Kb_prot = 20.0*fconst.caltoj
		args.Kr_prot = 1.0*fconst.caltoj
		args.CB_radii = True
		args.debye = True
		args.dielec = 10
		args.iconc = 0.01		# M
		ModelDir("reddy2017/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2017/sopsc.btparams.dat").copy2("interactions.dat")

	if args.baidya2022:
		print (">>> Using Reddy SOP-SCP-IDP model.")
		args.prot_cg = 2
		args.nuck_cg = 0
		args.bfunc = 8
		args.cutoff = 8.0
		args.cutofftype = -1
		args.contfunc = 1
		args.excl_rule = 2
		args.btparams = True
		opt.sopsc = True
		args.CA_rad = 1.9 #A
		args.CB_radii = True
		args.CB_charge = True
		args.CB_gly = True
		args.CB_charge = True
		args.Kb_prot = 20.0*fconst.caltoj
		args.Kr_prot = 1.0*fconst.caltoj
		args.debye = True
		args.dielec = 78
		args.iconc = 0.15	#M
		ModelDir("reddy2017/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2017/sopsc.btparams.dat").copy2("interactions.dat")

	""" presets end here """

	if args.excl_rule: 
		excl_rule = int(args.excl_rule)
		assert excl_rule in (1,2), "Error: Choose correct exclusion rule. Use 1: Geometric mean or 2: Arithmatic mean"
	else: excl_rule = 1
	
	if args.uniqtype: uniqtype = True
	else: uniqtype = False
	
	if args.prot_cg: 
		CGlevel["prot"] = int(args.prot_cg)
		if CGlevel["prot"] == 1: print (">>> Using CA-only model for protein. All other CB parameters will be ingnored.")
		elif CGlevel["prot"] == 2: print (">>> Using CB-CA model for protein.")
	else: CGlevel["prot"] = 2
	if args.nucl_cg:
		CGlevel["nucl"] = int(args.nucl_cg)
		assert CGlevel["nucl"] in (1,3,5), "Error. RNA/DNA only supports 1,3 or 5 as atom types"
		if CGlevel["nucl"] == 1: print (">>> Using P-only model for protein. All other parameters will be ingnored.")
		elif CGlevel["nucl"] == 3: print (">>> Using 3-bead P-S-B model for RNA/DNA.")
		elif CGlevel["nucl"] == 5: print (">>> Using 2-bead P-S and 3 beads per Base for RNA/DNA.")
	else: CGlevel["nucl"] = 3

	if args.interface: interface = True
	if args.Kb_prot:fconst.Kb_prot=float(args.Kb_prot)
	if args.Ka_prot:fconst.Ka_prot=float(args.Ka_prot)
	if args.Kd_bb_prot:fconst.Kd_prot["bb"]=float(args.Kd_bb_prot)
	if args.Kd_sc_prot:fconst.Kd_prot["Sc"]=float(args.Kd_sc_prot)
	else: fconst.Kd_prot["Sc"]=fconst.Ka_prot
	if args.mulfac_prot:fconst.Kd_prot["mf"]=float(args.mulfac_prot)
	if args.Kr_prot:fconst.Kr_prot = float(args.Kr_prot)
	if args.cutoff:contmap.cutoff=float(args.cutoff)

	if args.cmap: 
		args.cutofftype = 0
		contmap.type = 0
		contmap.file = args.cmap
	if args.cutofftype:
		contmap.type = int(args.cutofftype)
		if len(contmap.file) > 0:
			assert contmap.type == 0, "Error, Use type 0 if giving cmap file"
	if args.scaling: contmap.scale = float(args.scaling)
	if args.W_cont: contmap.W = True
	if args.contfunc: 
		contmap.func = int(args.contfunc)
		assert (contmap.func in range(0,5))
	else: contmap.func = 2

	if args.bfunc: 
		bond_function = int(args.bfunc)
		assert bond_function in (1,7,8), "Only Harmonic (1) and FENE (7) are supported bond length potentials"

	if args.interaction: contmap.scsc_custom = True
	if args.btparams:
		ModelDir("reddy2017/sopsc.btparams.dat").copy2("interactions.dat")
		opt.btparams=True
		contmap.scsc_custom=True
	if args.mjparams:
		ModelDir("reddy2017/sopsc.mjparams.dat").copy2("interactions.dat")
		opt.mjparams = True
		contmap.scsc_custom=True
	
	if args.CB_radii:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		CBradii=True
	else: CBradii=False

	if args.CB_gly:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		skip_glycine=False
		CB_gly = True
		print ("WARNING: Using CB for Glycines!! Make sure the all-atom pdb contains H-atom (HB)")
	else: CB_gly = False

	if args.CB_chiral:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		CB_chiral = True

	if args.CB_charge:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		charge.CB = True

	if args.P_charge: charge.P = True
	else: charge.P = False

	if args.CA_com:
		CA_com=True

	if args.dswap:
		dswap=True
		print ('This one assumes both chains are identical.')
		print ("Setting --all_chains True")
		all_chains = True
	
	if args.CA_rad: rad["CA"]=float(args.CA_rad)
	if args.CB_rad:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		rad["CB"] = float(args.CB_rad)
		aa_resi = Prot_Data().amino_acid_dict
		with open("radii.dat","w+") as fout:
			print (">>> C-beta radius given via user input. Storing in radii.dat")
			for i in aa_resi: fout.write('%s%4.2f\n' % (i.ljust(4),rad["CB"]))
		CBradii = True	#Read CB radius from radii.dat	

	if CBradii:
		aa_resi = Prot_Data().amino_acid_dict
		with open("radii.dat") as fin:
			rad.update({"CB"+aa_resi[x.split()[0]]:float(x.split()[1]) for x in fin})
	else:
		aa_resi = Prot_Data().amino_acid_dict
		rad.update({"CB"+aa_resi[x]:rad["CB"] for x in aa_resi})

	#converting A to nm
	for x in rad: rad[x] = np.round(0.1*rad[x],3)

	if args.dsb: dsb=True
	if args.hphobic: 
		hphobic=True
		if args.hpdist: hpdist=args.hpdist
		else: hpdist=5.5
		if args.hpstrength: hpstrength = args.hpstrength
		else: hpstrength=1
	
	if args.CB_far:
		if CGlevel["prot"] != 2: print ("WARNING: User opted from only-CA model. Ignoring all C-beta parameters.")
		CB_com=False
		CB_far=True
	if args.CB_com:
		if CGlevel["prot"] != 2: print ("WARNING: User opted from only-CA model. Ignoring all C-beta parameters.")
		CB_com=True
		assert not CB_far, "Conflicting input --CB_far and CB_com"


	#Replacing default paramteres with input paramters for nucleotide
	if args.Bpu_pos:
		nucl_pos["Bpu"] = str(args.Bpu_pos)
		assert nucl_pos["Bpu"] in Nucl_Data().pur_atom, "Error: Wrong Atom name entered for Purine. Use --help option to check allowed atom names"
	if args.Bpy_pos:
		nucl_pos["Bpy"] = str(args.Bpy_pos)
		assert nucl_pos["Bpy"] in Nucl_Data().pyr_atom, "Error: Wrong Atom name entered for Pyrimidine. Use --help option to check allowed atom names"
	if args.P_pos:
		nucl_pos["P"] = str(args.P_pos)
		assert nucl_pos["P"] in Nucl_Data().phos_atom, "Error: Wrong Atom name entered for Phosphate. Use --help option to check allowed atom names"
	if args.S_pos:
		nucl_pos["S"] = str(args.S_pos)
		assert nucl_pos["S"] in Nucl_Data().sug_atom, "Error: Wrong Atom name entered for Sugar. Use --help option to check allowed atom names"
	CG_pos = {"Bpu":nucl_pos["Bpu"],"Bpy":nucl_pos["Bpy"],"S":nucl_pos["S"],"P":nucl_pos["P"]}	

	if args.P_stretch: P_Stretch = True
	else: P_Stretch = False

	
	#Force constants
	if args.Kb_nucl: fconst.Kb_nucl = float(args.nKb)
	if args.Ka_nucl: fconst.Ka_nuck = float(args.nKa)
	if args.Kd_sc_nucl: fconst.Kd_nucl["sc"] = float(args.Kd_sc_nucl)
	if args.Kd_bb_nucl: fconst.Kd_nucl["bb"] = float(args.Kd_bb_nucl)
	if args.mulfac_nucl:fconst.Kd_nucl["mf"] = float(args.mulfac_nucl)
	if args.Kr_nucl:	fconst.Kr_nucl = float(args.Kr_nucl)
	if args.P_rad: rad["P"] = float(args.P_rad)
	if args.S_rad: rad["S"] = float(args.S_rad)
	if args.Bpu_rad: rad["Bpu"] = float(args.Bpu_rad)
	if args.Bpy_rad: rad["Bpy"] = float(args.Bpy_rad)
	if args.pistacklen: rad["stack"] = float(args.pistacklen)
	
	#solvant and ionic params
	if args.dielec:	charge.dielec = float(args.dielec)
	if args.iconc: charge.iconc = float(args.iconc)
	if args.irad: charge.irad = float(args.irad)
	if args.debye: charge.debye = True
	if args.debye_temp: charge.debye_temp = float(args.T)
	if args.debye_length: charge.inv_dl = 1.0/float(args.debye_length)

	#input structure file
	pdbdata = PDB_IO()
	if contmap.type == 1: assert args.aa_pdb, "Error. No all-atom pdb provided. --aa_pdb"
	assert args.aa_pdb or args.cg_pdb, "Error. Provide all-atom or coarse-grain pdb. --aa_pdb/--cg_pdb"
	if args.aa_pdb: pdbdata.loadfile(infile=args.aa_pdb,refine=True)
	elif args.cg_pdb: pdbdata.loadfile(infile=args.cg_pdb,refine=True)
	if args.control:	#Use Protein with DNA/RNA bound at natve site
		control_run = True
		assert not args.custom_nuc, "Error: --custom_nuc cannot be used with --control"
		custom_nuc = False
	else:
		control_run = False
		custom_nuc = True
		if args.custom_nuc:
			custom_nucl_file = PDB_IO()
			custom_nucl_file.loadfile(infile=args.custom_nuc,refine=True)
			print (">> Note: Using RNA/DNA from",custom_nucl_file.nucl.pdbfile)
			pdbdata.nucl = custom_nucl_file.nucl
			del(custom_nucl_file)
		else:
			if pdbdata.nucl.pdbfile != "":
				custom_nucl_file = pdbdata.nucl.pdbfile
				print (">> Note: custom_nuc option being used without input, will use unbound version of native RNA/DNA ")
				pdbdata.coordinateTransform()
	#output grofiles
	if args.pdbgro: grofile = str(args.pdbgro)
	else: grofile = "gromacs.gro"
	
	#defining individual group

	#write CG file
	pdbdata.write_CG_protfile(CGlevel=CGlevel,CAcom=CA_com,CBcom=CB_com,CBfar=CB_far,CBgly=CB_gly,nucl_pos=nucl_pos,outgro=grofile)

	#write .top file
	if args.grotop: ttopfile = str(args.grotop)
	else:topfile = 'gromacs.top'
	

	if args.clementi2000:
		top = Clementi2000(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,cmap=contmap,opt=opt)
		topdata = top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
	elif args.pal2019:
		top = Pal2019(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,cmap=contmap,opt=opt)
		topdata = top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
	elif args.reddy2017:
		top = Reddy2017(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,cmap=contmap,opt=opt)
		topdata = top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
	elif args.baidya2022:
		top = Baidya2022(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,cmap=contmap,opt=opt)
		topdata = top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
	else:
		top = Topology(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,cmap=contmap,opt=opt)
		topdata = top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)

if __name__ == '__main__':
    main()