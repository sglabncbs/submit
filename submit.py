#!/usr/bin/env python

"""
	SuBMIT: Structure Based Model(s) Input Toolkit
	Copyright (C) <2024>  <Digvijay Lalwani Prakash>

	A toolkit for generating input files for performing Molecular Dynamics
	Simulations (MD) of Coarse-Grain Structure Based Models (CG-SBM) on 
	GROMACS (4.5.4/4.6.7/5.1.4) and/or OpenSMOG (v1.1.1)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

	To add a new model, 1) add a new class (to a new .py file or to topology.py)
	and inherit Topology from topology.py. 2) See available functions in Class 
	Topology and re-write those which require changes based on your model.
	3) Predefine available arguments in (submit.py).

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
	<http://www.gnu.org/licenses/>.

    A copy of the GNU General Public License is included along with
	this program.

usage: python submit.py --help

Citation:-
	Publication:
	Authors: Digvijay L. Prakash, Arkadeep Banerjee & Shachi Gosavi
"""

import os
import argparse
import numpy as np
from typing import NamedTuple, Dict
from pathlib import Path
from PDB_IO import PDB_IO,Nucl_Data,Prot_Data,Fill_Box
from topology import *

class Options(Dict):
	opensmog=False
	xmlfile=str()
	dihed2xml=False
	nbshift=False
	uniqtype=False
	btparams=False
	mjmap=False
	btmap=False
	intra_symmetrize=False
	inter_symmetrize=False
	P_stretch=False
	codon_pairs=False
	interactions="interactions.dat"
	CAX=False
	hphobic=False
	CA_hp=False
	CB_hp=False
	nonbond=False
	interface=False
	dsb=False
	custom_nuc=False
	control_run=False
	CB_gly=False
	mass=dict()

class Constants(Dict):
	Kb_prot=200.0
	Ka_prot=40.0
	Kd_prot={"bb":1.0,"sc":Ka_prot,"mf":2.0}
	Kr_prot=1.0
	Kb_nucl= 200				
	Ka_nucl=40				
	Kd_nucl={"bb":0.7,"sc":0.5,"mf":1.0}				
	Kr_nucl=1.0
	Kboltz=8.314462618E-3 #KJ mol-1 nm-1
	caltoj=4.184 #kcal mol-1 A-2 to kcal mol-1 A-2
	permol=6.022E+23          #n/mol  		#Avogadro's number

class ContactMap(Dict):
	cutoff=4.5 	#A
	type=1 		#all-atom mapped to CG
	scale=1.0 
	func=2 		# LJ 10-12
	nbfunc=-1	# LJ 10-12
	W=False 		#Equal weights 
	file=list()	# no cmap file
	custom_pairs=False

class Charge(Dict):
	CA=False
	CB=False
	P=False
	PP=False
	debye=False
	dielec=78
	iconc=0.1 				#M L-1               
	irad=1.4 				#nm (for NaCl)
	debye_temp=298  		#K
	inv_dl=0
	Kboltz=8.314462618E-3	#KJ mol-1 nm-1
	kcal=False				# use Kcal/mol value 
	caltoj=4.184			#kcal mol-1 A-2 to kcal mol-1 A-2
	inv_4pieps=138.935485	#KJ mol-1 e-2
	permol=6.022E+23		#n/mol  		#Avogadro's number

class ModelDir:
	def __init__(self,file) -> str:
		self.path="/".join(str(Path(__file__)).split("/")[:-1]+["models"]+file.split("/"))
		return 

	def copy2(self,copyfile):
		with open(copyfile,"w+") as fout:
			fout.write(open(self.path).read())
		return 1

class CleanUP:
	def __init__(self,grosuffix=str(),topsuffix=str(),xmlsuffix=str(),coulomb=Charge(),enrgrps=[],box_width=500.0,fillstatus=False,gen_cg=False):
		self.coulomb=coulomb
		self.enrgrps=enrgrps
		self.createDir()
		#self.delFiles(f_suffix="renum.pdb")
		self.delFiles(f_suffix="renum.nucl.pdb")
		self.delFiles(f_suffix="renum.prot.pdb")
		self.moveFiles(f_suffix=".pdb",f_middle=".renum.",out_subdir="RenumberedPDB_CMap")
		self.moveFiles(f_suffix=".fa.pdb",out_subdir="RenumberedPDB_CMap")
		self.moveFiles(f_suffix="cont",out_subdir="RenumberedPDB_CMap")
		self.moveFiles(f_suffix=grosuffix,out_subdir="GRO_TOP_XML")
		if not gen_cg:
			self.moveFiles(f_suffix=topsuffix,out_subdir="GRO_TOP_XML")
			self.moveFiles(f_middle=topsuffix)
		self.moveFiles(f_middle=grosuffix)
		if len(xmlsuffix)>0: 
			self.moveFiles(f_suffix=xmlsuffix,out_subdir="GRO_TOP_XML")
			self.moveFiles(f_middle=xmlsuffix)
		self.moveFiles(f_prefix="table",f_suffix=".xvg",out_subdir="Tables")
		self.moveFiles(f_prefix="interactions",f_suffix=".dat",out_subdir="model_params")
		self.moveFiles(f_prefix="rad",f_suffix=".dat",out_subdir="model_params")
		self.moveFiles(f_prefix="cgmass",f_suffix=".dat",out_subdir="model_params")
		self.moveFiles(f_middle="molecule_order.list")
		self.renameTables()
		if not fillstatus:
			self.genbox(grosuffix=grosuffix,topsuffix=topsuffix,box_width=box_width,gen_cg=gen_cg)	

	def createDir(self):
		os.makedirs("SuBMIT_Output/RenumberedPDB_CMap",exist_ok=True)
		os.makedirs("SuBMIT_Output/GRO_TOP_XML",exist_ok=True)
		os.makedirs("SuBMIT_Output/model_params",exist_ok=True)
		os.makedirs("SuBMIT_Output/Tables",exist_ok=True)
		return

	def moveFiles(self,f_suffix=str(),f_prefix=str(),f_middle=str(),out_subdir=str()):
		common_dir="SuBMIT_Output"
		for filename in os.listdir():
			if filename.startswith(f_prefix) and filename.endswith(f_suffix):
				if f_middle in filename and filename!=f_suffix and filename!=f_prefix:
					os.replace(filename,"%s/%s/%s"%(common_dir,out_subdir,filename))
		return
	
	def renameTables(self):
		tabledir="SuBMIT_Output/Tables"
		tablelist=[x for x in os.listdir(tabledir) if "lj" in x.lower()]
		if len(tablelist)==0: return
		if len(self.enrgrps)<=1: return
		self.enrgrps.remove("Protein")
		fgrp=open("%s/%s"%(tabledir,"energy_groups.list"),"w+")
		for t1 in tablelist:
			if "Prot" in t1 or "NA" in t1: continue
			t="%s/%s"%(tabledir,t1)
			if self.coulomb.CA or self.coulomb.CB:
				if "_coul_" in t:
					fgrp.write("Protein_Protein\t:\t%s\n"%t1)
					outfile=str().join(t.split(".")[:-1]+["_Protein_Protein.xvg"]).split("_coul_")
					with open("_".join(outfile),"w+") as fout: fout.write(open(t).read())
					if self.coulomb.P:
						for g1 in self.enrgrps:
							fgrp.write("%s_Protein\t:\t%s\n"%(g1,t1))
							outfile=str().join(t.split(".")[:-1]+["_%s_Protein.xvg"%g1]).split("_coul_")
							with open("_".join(outfile),"w+") as fout: fout.write(open(t).read())
			if self.coulomb.P and self.coulomb.PP:
				if "_coul_" in t:
					g1=self.enrgrps[0]
					for g2 in self.enrgrps:
						fgrp.write("%s_%s\t:\t%s\n"%(g1,g2,t1))
						outfile=str().join(t.split(".")[:-1]+["_%s_%s.xvg"%(g1,g2)]).split("_coul_")
						with open("_".join(outfile),"w+") as fout: fout.write(open(t).read())
			if self.coulomb.P and not self.coulomb.PP:
				if "_coul_" not in t:
					g1=self.enrgrps[0]
					for g2 in self.enrgrps:
						fgrp.write("%s_%s\t:\t%s\n"%(g1,g2,t1))
						outfile=str().join(t.split(".")[:-1]+["_%s_%s.xvg"%(g1,g2)])
						with open(outfile,"w+") as fout: fout.write(open(t).read())
		fgrp.close()	
		return

	def delFiles(self,f_suffix=str(),f_prefix=str(),f_middle=str()):
		if len(f_prefix)+len(f_middle)+len(f_suffix)==0: return
		for filename in os.listdir():
			if filename.startswith(f_prefix) and filename.endswith(f_suffix):
				if f_middle in filename and filename!=f_suffix and filename!=f_prefix:
					os.remove(filename)
		return

	def genbox(self,grosuffix,topsuffix,box_width,gen_cg):
		if "molecule_order.list" in os.listdir("SuBMIT_Output"):
			mol_list=[tuple(line.split()) \
				 	for line in open("SuBMIT_Output/molecule_order.list")\
					if not line.startswith(("#","@",":"))]
			if len(mol_list)==1 and int(mol_list[0][-1])==1:
				open("SuBMIT_Output/%s"%grosuffix,"w+").write(\
					open("SuBMIT_Output/GRO_TOP_XML/%s_%s"%(mol_list[0][1],grosuffix)).read())
			else:			
				assert len(mol_list)!=0
				if np.sum(np.int_(np.transpose(mol_list)[0]))==0:
					mol_list=[("%s_%s"%(x[1],grosuffix),int(x[2])) for x in mol_list]
				else:
					mol_list=[("%s_%s"%(x[1],grosuffix),int(x[2])) for x in mol_list]
				with open('SuBMIT_Output/genbox_commands.sh','w+') as fout:
					for i in "xyz":fout.write("box_%s=%.3f\n"%(i,0.1*box_width))
					fout.write("seed=1997 #default for gromacs 4.5.4\n")
					fout.write('echo -e "EMPTY GROFILE\\n0\\n$box_x $box_y $box_z" > _temp_0.gro\n')
					fout.write('echo -e "pbc = xyz" > pbcBox.mdp\n')
					infile="_temp_0.gro"
					fout.write('#for GROMACS v4 (<v5)\n')
					for i in range(len(mol_list)):
						outfile="_temp_%d.gro"%(i+1)
						addfile="GRO_TOP_XML/%s"%mol_list[i][0]
						#if i+1==len(mol_list): outfile=grosuffix
						command1='genbox -seed ${seed} -cp %s -ci %s -nmol %d -try 100 -o %s\n'%\
									(infile,addfile,mol_list[i][1],outfile)
						fout.write(command1)
						infile=outfile
					fout.write('#for GROMACS v5 (>=v5)\n')
					for i in range(len(mol_list)):
						outfile="_temp_%d.gro"%(i+1)
						addfile="GRO_TOP_XML/%s"%mol_list[i][0]
						#if i+1==len(mol_list): outfile=grosuffix
						command1='#gmx insert-molecules -cp %s -ci %s -nmol %d -try 100 -o %s\n'%\
										(infile,addfile,mol_list[i][1],outfile)
						fout.write(command1)
						infile=outfile
					outfile=grosuffix
					if not gen_cg:
						command2='grompp -f pbcBox.mdp -c %s -p %s -o _temp_.tpr  -po _temp_.mdp\n'%(infile,topsuffix)
						command2+='echo 0 | trjconv -f %s -s _temp_.tpr -o %s -pbc mol -ur rect\n'%(infile,outfile)
					else: command2='mv %s %s\n'%(infile,outfile)
					fout.write(command2)
					fout.write('rm pbcBox.mdp _temp_*\n')
					fout.write('echo "NOTE: at higher nmol values in smaller box (high number density), beyond a certain nmol value, genbox outputs will be Identical. To avoid this, the order of molecules in genbox commands can be shuffled."')
		return

def main():
	
	""" loading arguments here """
	parser=argparse.ArgumentParser(description="Generate GROMACS and OpenSMOG potential files for Protein + Nucleic Acids SBM models.")
	
	#Predefined Models
	parser.add_argument("--clementi2000","-clementi2000","--calpha_go2000","-calpha_go2000",action="store_true",help="Clementi et. al. 2000 CA-only model. 10.1006/jmbi.2000.3693")
	parser.add_argument("--afsar2008","-afsar2008","--chan2008","-chan2008",action="store_true",help="Zarrine-Afsar et. al. 2008 CA-only + hydrophobic model with . 10.1073/pnas.0801874105")
	parser.add_argument("--azia2009","-azia2009","--levy2009","-levy2009",action="store_true",help="Azia 2009 CB-CA + Debye-Huckel model. 10.1016/j.jmb.2009.08.010")
	parser.add_argument("--pal2019","-pal2019","--levy2019","-levy2019",action="store_true",help="Pal & Levy 2019 Protein CB-CA & RNA/DNA P-S-B model. 10.1371/journal.pcbi.1006768")
	parser.add_argument("--reddy2016","-reddy2016","--maity2016","-maity2016","--sopsc2016","-sopsc2016",action="store_true",help="Maity & Reddy 2016 SOP-SC CA-CB. 10.1021/jacs.5b11300")
	parser.add_argument("--denesyuk2013","-denesyuk2013","--rna_tis2013","-rna_tis2013",action="store_true",help="Denesyuk & Thirumalai 2013 Three Interaction Site TIS P-S-B model. 10.1021/jp401087x")
	parser.add_argument("--chakraborty2018","-chakraborty2018","--dna_tis2018","-dna_tis2018",action="store_true",help="Chakraborty & Thirumalai 2018 Three Interaction Site TIS P-S-B model. 10.1021/acs.jctc.8b00091")
	parser.add_argument("--baul2019","-baul2019","--sop_idp2019","-sop_idp2019",action="store_true",help="Baul et. al. 2019 SOP-SC-IDP CA-CB. 10.1021/acs.jpcb.9b02575")
	parser.add_argument("--baidya2022","-baidya2022","--sop_idp2022","-sop_idp2022",action="store_true",help="Baidya & Reddy 2022 SOP-SC-IDP CA-CB. 10.1021/acs.jpclett.2c01972")
	parser.add_argument("--baratam2024","-baratam2024","--sop_multi","-sop_multi",action="store_true",help="Baratam & Srivastava 2024 SOP-MULTI CA-CB. 10.1101/2024.04.29.591764")
	parser.add_argument("--sop_idr","-sop_idr",action="store_true",help="Reddy-Thiruamalai(SOPSC) + Baidya-Reddy(SOPIDP) hybrid CA-CB")
	parser.add_argument("--banerjee2023","-banerjee2023","--selfpeptide","-selfpeptide",action="store_true",help="Banerjee & Gosavi 2023 Self-Peptide model. 10.1021/acs.jpcb.2c05917")
	parser.add_argument("--virusassembly","-virusassembly","--capsid","-capsid",action="store_true",help="Preset for structure based virus assembly (inter-Symmetrized)")
	parser.add_argument("--dlprakash","-dlprakash","--duplexpair","-duplexpair",action="store_true",help="Codon pairs (duplex based weight) for Pal2019")

	#Input structures or sequences
	parser.add_argument("--aa_pdb","-aa_pdb", nargs='+', help='User input all-atom pdbfile/gro/mmCIF e.g. 1qys.pdb')
	parser.add_argument("--cg_pdb","-cg_pdb", nargs='+', help='User input coarse grained pdbfile')
	parser.add_argument("--idp_seq","-idp_seq",help="User input sequence fasta file for building/extracting IDRs/segments etc.")
	parser.add_argument("--nmol","-nmol", nargs='+', help="Include nmol number of molecules in the topology. List of integers. Defatul1 1 per input pdbfile")

	#output
	parser.add_argument("--gen_cg","-gen_cg",action='store_true', help="Only Generate CG structure without generating topology .top/.xml files")
	parser.add_argument("--outtop","-outtop",help='Gromacs topology file output name (tool adds prefix nucl_  and prot_ for independednt files). Default: gromacs.top')
	parser.add_argument("--outgro","-outgro", help='Name for output .gro file.(tool adds prefix nucl_  and prot_ for independednt files). Default: gromacs.gro')
	parser.add_argument("--box","-box", help='Width of the cubic simulation box. Default: 500.0 Å. Use 0 for no box.')
	parser.add_argument("--voxel","-voxel","--box_cell","-box_cell", help='Width of the minimal cubic volume unit, used to fill the simulation box. Default: 1.618 Å')
	parser.add_argument("--outxml","-outxml", help='Name for output .xml (openSMOG) file.(tool adds prefix nucl_  and prot_ for independednt files). Default: opensmog.xml (and opensmog.top)')
	parser.add_argument("--opensmog", "-opensmog",action='store_true', help="Generate files ,xml and .top files for openSMOG. Default: False")
	parser.add_argument("--dihed2xml", "-dihed2xml",action='store_true', help="Write torsions to opensmog xml. Adds conditon for angle->n*pi. Only supported for OpensMOGmod:https://github.com/sglabncbs/OpenSMOGmod. Default: False")

	#level of coarse-graining
	parser.add_argument("--prot_cg", "-prot_cg", type=int, help="Level of Amino-acid coarse-graining 1 for CA-only, 2 for CA+CB. Dafault: 2 (CA+CB)")
	parser.add_argument("--nucl_cg", "-nucl_cg", type=int, help="Level of Amino-acid coarse-graining 1 for P-only, 3 for P-S-B, 5 for P-S-3B. Dafault: 3 (P-S-B)")

	#protein CG paramters
	parser.add_argument("--CA_rad","-CA_rad","--ca_rad","-ca_rad",type=float, help="User defined radius (0.5*excl-volume-rad) for C-alpha (same for all beads) in Angstrom. Default: 1.9 Å")
	parser.add_argument("--CA_com","-CA_com","--ca_com","-ca_com",action="store_true",help="Place C-alpha at COM of backbone. Default: False")
	parser.add_argument("--CB_rad","-CB_rad","--cb_rad","-cb_rad",type=float, help="User defined radius (0.5*excl-volume-rad) for C-beta (same for all beads) in Angstrom. Default: 1.5 Å")
	parser.add_argument("--cg_radii","-cg_radii","--cg_radii","-cg_radii",action="store_true", help="User defined C-beta radii from radii.dat (AA-3-letter-code   radius-in-Angsrtom). Default: False")
	parser.add_argument("--CB_com","-CB_com","--cb_com","-cb_com", action="store_true", default=False,help="Put C-beta at side-chain COM. Default: False")
	parser.add_argument("--CB_far", "-CB_far","--Cb_far", "-Cb_far", action="store_true", help="Place C-beta on farthest non-hydrogen atom. Default: False")
	parser.add_argument("--CB_chiral","-CB_chiral","--cb_chiral","-CB_chiral",action='store_true',help="Improper dihedral for CB sidechain chirality (CAi-1:CAi+1:CAi:CBi). Default: False")
	parser.add_argument("--CB_gly","--cb_gly","-CB_gly","-cb_gly",action="store_true",default=False,help="Add C-beta for glycine (pdb-file must have H-atoms). Default: Flase")
	#RNA/DNA CG paramters
	parser.add_argument("--P_rad","-P_rad","--p_rad","-p_rad", type=float, help="User defined radius for Backbone Phosphate bead. Default= 1.9 Å")
	parser.add_argument("--S_rad","-S_rad","--s_rad","-s_rad", type=float, help="User defined radius for Backbone Sugar bead. Default= 1.9 Å")
	parser.add_argument("--Bpu_rad","-Bpu_rad","--bpu_rad","-bpu_rad", type=float, help="User defined radius for N-Base Purine bead. Default=1.5 Å")
	parser.add_argument("--Bpy_rad","-Bpy_rad","--bpy_rad","-bpy_rad", type=float, help="User defined radius for N-Base Pyrimidine bead. Default=1.5 Å")
	parser.add_argument("--Bpu_pos","-Bpu_pos","--bpu_pos","-bpu_pos", help="Put input atom of Purine [N1,C2,H2-N2,N3,C4,C5,C6,O6-N6,N7,C8,N9,COM] as position of B. Default=COM(Center_of_Mass)")
	parser.add_argument("--Bpy_pos","-Bpy_pos","--bpy_pos","-bpy_pos", help="Put input atom of Pyrimidine [N1,C2,O2,N3,C4,O4-N4,C5,C6,COM] as position of B. Default=COM(Center_of_Mass)")
	parser.add_argument("--S_pos"  ,"-S_pos","--s_pos"  ,"-s_pos", help="Put input atom of Sugar [C1',C2',C3',C4',C5',H2'-O2',O3',O4',O5',COM] as position of S. Default=COM(Center_of_Mass)")
	parser.add_argument("--P_pos"  ,"-P_pos","--p_pos"  ,"-p_pos", help="Put input atom of Phosphate [P,OP1,OP2,O5',COM] group as position of P. Default=COM(Center_of_Mass)")

	#protein ff paramters
	parser.add_argument("--Kb_prot","-Kb_prot","--Kb","-Kb", type=float, help="User defined force constant K_bond for Proteins. Default: 200.0 ε/Å^2 (ε = 1KJ/mol)")
	parser.add_argument("--Ka_prot","-Ka_prot","--Ka","-Ka", type=float, help="User defined force constant K_angle for Proteins. Default: 40.0 ε/rad^2 (ε = 1KJ/mol)")
	parser.add_argument("--Kd_bb_prot","-Kd_bb_prot","--Kd","-Kd", type=float, help="User defined force constant K_dihedral for Proteins. Default: 1.0 ε (ε = 1KJ/mol)")
	parser.add_argument("--Kd_sc_prot","-Kd_sc_prot","--Kd_chiral","-Kd_chiral", type=float, help="User defined force constant K_dihedral for Proteins. Default: Use Ka_prot value")
	parser.add_argument("--mulfac_prot","-mulfac_prot", type=float, help="User defined Multiplicity scaling factor of K_dihedral/mulfac_prot for Proteins. Default: 2")
	parser.add_argument("--Kr_prot", "-Kr_prot", type=float, help="Krepulsion. Default=1.0 ε")
	parser.add_argument("--uniqtype","-uniqtype",action="store_true",help="Each atom has unique atom type (only use for small systems)")
	parser.add_argument("--bfunc","-bfunc", type=int, help="Bond function 1: harnomic. Default: 1 (Harmonic)")
	#RNA/DNA ff paramters
	parser.add_argument("--Kb_nucl","-Kb_nucl","--nKb","-nKb", type=float, help="User defined force constant K_bond for RNA/DNA. Default: 200.0 ε/Å^2 (ε = 1KJ/mol)")
	parser.add_argument("--Ka_nucl","-Ka_nucl","--nKa","-nKa", type=float, help="User defined force constant K_angle for RNA/DNA. Default: 40.0 ε/rad^2 (ε = 1KJ/mol)")
	parser.add_argument("--Kd_sc_nucl","-Kd_sc_nucl","--nKd","-nKd", type=float, help="User defined force constant K_dihedral for Bi-Si-Si+1-Bi+1. Default: 0.5 ε (ε = 1KJ/mol)")
	parser.add_argument("--Kd_bb_nucl","-Kd_bb_nucl","--P_nKd","-P_nKd", type=float, help="User defined force constant K_dihedral for Backbone Pi-Pi+1-Pi+2-Pi+3. Default: 0.7 ε (ε = 1KJ/mol)")
	parser.add_argument("--P_stretch","-P_stretch",help="Stretch the backbone dihedral to 180 degrees. Default=Use native  backbone dihedral")
	parser.add_argument("--mulfac_nucl","-mulfac_nucl", type=float, help="User defined Multiplicity scale factor of K_dihedral for Nucleic Acids. Default: 1")
	parser.add_argument("--Kr_nucl", "-Kr_nucl", type=float, help="Krepulsion. Default: 1.0 ε") 

	#native  determining contacts parameters
	parser.add_argument("--cutoff","-cutoff",type=float,help="User defined Cut-off (in Angstrom) for contact-map generation. Default: 4.5 Å (for all-atom) or 8.0 Å (for coarse-grained)")
	parser.add_argument("--cutofftype","-cutofftype",type=int,help="-1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1")
	parser.add_argument("--W_cont","-W_cont",action="store_true",help="Weight (and normalize) CG contacts based on all atom contact pairs")
	parser.add_argument("--cmap","-cmap",nargs='+',help="User defined cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)")
	parser.add_argument("--scaling","-scaling", help="User defined scaling for mapping to all-atom contact-map.")
	parser.add_argument("--contfunc","-contfunc",type=int,help="1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 6 Gauss + excl, 7 Multi Gauss  . Default: 2")
	#overwrite for proteins
	parser.add_argument("--cutoff_p","-cutoff_p",type=float,help="User defined Cut-off (in Angstrom) for Protein contact-map generation. Default: 4.5 Å (for all-atom) or 8.0 Å (for coarse-grained)")
	parser.add_argument("--cutofftype_p","-cutofftype_p",type=int,help="For Proteins: -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1")
	parser.add_argument("--W_cont_p","-W_cont_p",action="store_true",help="Weight (and normalize) Protein CG contacts based on all atom contacts")
	parser.add_argument("--cmap_p","-cmap_p",nargs='+',help="User defined Protein cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)")
	parser.add_argument("--scaling_p","-scaling_p", help="User defined scaling for mapping to all-atom contact-map.")
	parser.add_argument("--contfunc_p","-contfunc_p",type=int,help="Proteins. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18,  6 Gauss + excl, 7 Multi Gauss  . Default: 2")
	#overwrite for RNA/DNA
	parser.add_argument("--cutoff_n","-cutoff_n",type=float,help="User defined Cut-off (in Angstrom) for RNA/DNA contact-map generation. Default. Default: 4.5 Å (for all-atom) or 8.0 Å (for coarse-grained)")
	parser.add_argument("--cutofftype_n","-cutofftype_n",type=int,help="For RNA/DNA. -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1")
	parser.add_argument("--W_cont_n","-W_cont_n",action="store_true",help="Weight (and normalize) RNA/DNA CG contacts based on all atom contacts")
	parser.add_argument("--cmap_n","-cmap_n",nargs='+',help="User defined RNA/DNA cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)")
	parser.add_argument("--scaling_n","-scaling_n", help="User RNA/DNA defined scaling for mapping to all-atom contact-map.")
	parser.add_argument("--contfunc_n","-contfunc_n",type=int,help="RNA/DNA. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18,  6 Gauss + excl, 7 Multi Gauss  . Default: 2")
	#inter Protein-RNA/DNA
	parser.add_argument("--cutoff_i","-cutoff_i",type=float,help="User defined Cut-off (in Angstrom) for Protein RNA/DNA interface contact-map generation. Default: 4.5 Å (for all-atom) or 8.0 Å (for coarse-grained)")
	parser.add_argument("--cutofftype_i","-cutofftype_i",type=int,help="For Protein RNA/DNA interface. -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1")
	parser.add_argument("--W_cont_i","-W_cont_i",action="store_true",help="Weight (and normalize) Protein RNA/DNA interface CG contacts based on all atom contacts")
	parser.add_argument("--cmap_i","-cmap_i",help="User defined Protein RNA/DNA interface cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)")
	parser.add_argument("--scaling_i","-scaling_i", help="User Protein RNA/DNA interface defined scaling for mapping to all-atom contact-map.")
	parser.add_argument("--contfunc_i","-contfunc_i",type=int,help="Protein RNA/DNA interface. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18,  6 Gauss + excl, 7 Multi Gauss  . Default: 2")

	#nonbonded params
	parser.add_argument("--nbfunc","-nbfunc",type=int,help="1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18 (3: modified gmx5), (6&7: OpenSMOG)6 Gauss + excl, 7 Multi Gauss  . Default: 2")
	parser.add_argument("--excl_rule",type=int,help="Use 1: Geometric mean. 2: Arithmatic mean")
	parser.add_argument("--nbshift", "-nbshift",action='store_true', help="(with --opensmog) Shift the potential (V(r)) by a constant (V(r_c)) such that it is zero at cutoff (r_c). Default: False")
	#for interactions
	parser.add_argument("--interaction","-interaction",action='store_true', default=False, help='User defined pair interactions in file interactions.dat.')
	parser.add_argument('--btparams',"-btparams", action='store_true', help='Use Betancourt-Thirumalai interaction matrix.')
	parser.add_argument('--mjparams',"-mjparams", action='store_true', help='Use Miyazawa-Jernighan interaction matrix.')
	parser.add_argument("--interface","-interface", help='User defined multimer interface nonbonded params. Format atype1 atype2 eps sig(A)')
	parser.add_argument("--cg_mass","-cg_mass",action="store_true", help="User defined CG bead masses from cgmass.dat (atype mas in au). Default: False")
	#electrostatic
	parser.add_argument("--debye","-debye",action='store_true', help="Use Debye-Huckel electrostatic interactions.")
	parser.add_argument("--debye_length","-debye_length", type=float, help="Debye length. in (Å)")
	parser.add_argument("--debye_temp","-debye_temp", type=float, help="Temperature for Debye length calculation. Default: 298 K")
	parser.add_argument("--CA_charge","-CA_charge", action='store_true', default=False, help='Put charges on CA for K,L,H,D,E. Default: False')
	parser.add_argument("--CB_charge","-CB_charge", action='store_true', default=False, help='Put charges on CB for K,L,H,D,E. Default: False')
	parser.add_argument("--P_charge","-P_charge", action='store_true', default=False, help='Negative charge on Phosphate bead. Default: False')
	parser.add_argument("--hphobic","-hphobic", action='store_true', default=False, help='Nake CA or CB hydrophobic for A,V,I,L,M,W,F,Y. Default: False')
	parser.add_argument('--hpstrength',"-hpstrength",help='Strength with which hydrophobic contacts interact. Default: 0.8 ε')
	parser.add_argument('--hpdist', "-hpdist", help='Equilibrium distance for hydrophobic contacts. Default: 5.0 Å')
	parser.add_argument("--PPelec","-PPelec", action='store_true', default=False, help='Add electrostatic repulsions for  Phosphate-Phosphate beads. Default: False')
	parser.add_argument("--iconc","-iconc", type=float, help="Solvent ion conc.(N) for Debye length calcluation. Default: 0.1 M")  
	parser.add_argument("--irad","-irad", type=float, help="Solvent ion rad for Debye length calcluation. Default: 1.4 Å")  
	parser.add_argument("--dielec","-dielec", type=float, help="Dielectric constant of Solvent. Default: 78")
	parser.add_argument("--elec_kcal","---elec_kcal", action='store_true', help="Use inv_4.pi.eps0 33.206 (Kcal mol-1 e-2) value . Default False (138.935485 KJ mol-1 e-2)")

	#disabled for now
	parser.add_argument("--dswap","-dswap", action='store_true', default=False, help='For domain swapping runs. Symmetrised SBM is generated.')
	parser.add_argument("--sym_intra","--sym_intra", action='store_true', default=False, help='Intra-chain Symmetrised SBM is generated.')
	#parser.add_argument("--dsb", "-dsb",action='store_true', help="Use desolvation barrier potential for contacts. Default: False")
	parser.add_argument("--custom_nuc","-custom_nuc", help='Use custom non native DNA/RNA structure Eg.: polyT.pdb. Default: Use from native structure')
	parser.add_argument("--control", action='store_true', help='Use the native system as control. Use DNA/RNA bound to native protein site. --custom_nuc will be disabled. Default: False (Move DNA/RNA away from native binding site)')
	#exclusion volume

	args=parser.parse_args()

	#defualt potoein-NA parameters
	opt=Options()
	fconst=Constants()
	charge=Charge()
	prot_contmap=ContactMap()
	nucl_contmap=ContactMap()
	inter_contmap=ContactMap()
	inter_contmap.type=-1 #default type
	inter_contmap.func=-1 #buffer

	opt.nonbond=False
	opt.interface=False
	opt.custom_nuc=False
	opt.control_run=False

	CGlevel={"prot":2,"nucl":3}
	Nmol={"prot":[1],"nucl":[1]}
	rad=dict()	 


	#Set default parameters for proteins
	#For preteins
	
	bond_function=1
	nonbond_function=-1
	rad["CA"]=1.9
	rad["CB"]=1.5
	CA_com=False
	CB_far=False
	CB_com=False
	CB_chiral=False
	cg_radii=False
	CB_gly=False
	#Set default parameters for nucleotides
	rad["P"]=1.9					#A
	rad["S"]=1.9					#A
	rad["Bpy"]=1.5				#A
	rad["Bpu"]=1.5				#A
	opt.P_stretch=False

	charge.CA=False
	charge.CB=False
	charge.P=False
	excl_rule=1
	uniqtype=False
	CG_mass=False

	#default position
	nucl_pos=dict()
	nucl_pos["Bpu"]="COM"		#Center of Mass for purine
	nucl_pos["Bpy"]="COM"		#Center of Mass for pyrimidine
	nucl_pos["P"]="COM"			#Center of Mass for phosphate group
	nucl_pos["S"]="COM"			#Center of Mass for sugar

	"""" Defining inputs for preset models """

	list_of_protein_presets=[
		args.clementi2000, args.afsar2008, args.azia2009, args.reddy2016,\
		args.baul2019, args.baidya2022, args.baratam2024, args.sop_idr]
	list_of_nucleicacid_presets=[args.denesyuk2013, args.chakraborty2018,args.dlprakash]
	list_of_hybrid_presets=[args.banerjee2023, args.virusassembly,args.pal2019]

	assert (np.sum(np.int_(list_of_hybrid_presets))) <= 1,\
		"Error! Two hybrid (protein + nucleic acid) SBMs cannot be implemented togther."
	assert (np.sum(np.int_(list_of_protein_presets+list_of_hybrid_presets))) <= 1,\
		"Error! Two protein SBMs cannot be implemented togther."
	assert (np.sum(np.int_(list_of_nucleicacid_presets+list_of_hybrid_presets))) <= 1,\
		"Error! Two nucleic acid SBMs cannot be implemented togther."

	if args.clementi2000:
		print (">>> Using Clementi et. al. 2000 CA-only model. 10.1006/jmbi.2000.3693")
		assert args.aa_pdb, "Error no pdb input --aa_pdb"
		#fixed params can't be overwritten
		CGlevel["prot"]=1		# CA_only	
		CGlevel["nucl"]=0		# No RNA/DNA
		rad["CA"]=2.0			# 4.0 A excl vol rad
		prot_contmap.W=False			# not weighted 
		prot_contmap.cutoff=4.5	# 4.5 A
		prot_contmap.cutofftype=1	# all-atom contacts mapped to CG
		prot_contmap.contfunc=2	# LJ 10-12

	if args.afsar2008:
		print (">>> Using Zarrine-Afsar et. al. 2008 CA-only + hydrophobic model with . 10.1073/pnas.0801874105")
		assert args.aa_pdb, "Error no pdb input --aa_pdb"
		#fixed params can't be overwritten
		CGlevel["prot"]=1		# CA_only	
		CGlevel["nucl"]=0		# No RNA/DNA
		rad["CA"]=2.0			# 4.0 A excl vol rad
		prot_contmap.W=False			# not weighted 
		prot_contmap.cutoff=4.5	# 4.5 A
		prot_contmap.cutofftype=1	# all-atom contacts mapped to CG
		prot_contmap.contfunc=2	# LJ 10-12
		opt.hphobic=True
		nonbond_function=7
		args.opensmog=True

	if args.pal2019:
		print (">>> Using Pal & Levy 2019 model. 10.1371/journal.pcbi.1006768")
		assert args.aa_pdb, "Error no pdb input --aa_pdb."
		CGlevel["prot"]=2		# CB-CA
		CGlevel["nucl"]=3		# P-S-B
		rad["CA"]=1.9			# 3.8 A excl vol rad
		rad["CB"]=1.5			# 3.0 A excl vol rad
		rad["P"]=3.7					#A
		rad["S"]=3.7					#A
		rad["Bpy"]=1.5				#A
		rad["Bpu"]=1.5				#A
		CB_far=True			# CB at farthest SC atom 
		CB_chiral=False		# improp dihed for CAi-1 CAi+1 CAi CBi
		charge.CB=True		# Charge on CB
		charge.P=True			#charge on P
		excl_rule=2			# Excl volume Arith. Mean
		fconst.Kd_prot["mf"]=1.0	#factor to divide 3 multiplicity dihed term
		prot_contmap.W=False			# contact weight
		prot_contmap.cutoff=4.5	# cutoff
		prot_contmap.type=1		# Calculate from all-atom structure
		prot_contmap.func=2		# Use LJ 10-12 pairs
		nucl_contmap.type=-1		# Do not calculate
		inter_contmap.type=-1		# Do not calculate
		charge.debye=True		# Use DH-electrostatics
		charge.dielec=70		# dielectric constant
		charge.iconc=0.01		# concentration
		charge.kcal=True		# use Kcal mol-1 e-2 value
		opt.interface=True
		opt.P_stretch=True	# set P_P_P_P dihed to 180
		ModelDir("pal2019/adj_nbnb.stackparams.dat").copy2("interactions.pairs.dat")
		ModelDir("pal2019/inter_nbcb.stackparams.dat").copy2("interactions.interface.dat")

	if args.virusassembly:
		print (">>> Using template capsid assembly preset.")
		assert args.aa_pdb, "Error no pdb input --aa_pdb."
		CGlevel["prot"]=2		# CB-CA
		CGlevel["nucl"]=3		# P-S-B
		rad["CA"]=1.9			# 3.8 A excl vol rad
		rad["CB"]=1.5			# 3.0 A excl vol rad
		CB_far=True			# CB at farthest SC atom 
		#CB_chiral=True		# improp dihed for CAi-1 CAi+1 CAi CBi
		charge.CB=True		# Charge on CB
		charge.P=True			#charge on P
		excl_rule=2			# Excl volume Arith. Mean
		fconst.Kd_prot["mf"]=1.0	#factor to divide 3 multiplicity dihed term
		prot_contmap.W=False			# contact weight
		prot_contmap.cutoff=4.5	# cutoff
		prot_contmap.type=1		# Calculate from all-atom structure
		prot_contmap.func=2		# Use LJ 10-12 pairs
		prot_contmap.W=True
		nucl_contmap.cutoff=4.5
		nucl_contmap.type=1	
		nucl_contmap.func=2
		nucl_contmap.W=True
		inter_contmap.type=0
		charge.debye=True		# Use DH-electrostatics
		charge.iconc=0.1		# concentration
		charge.dielec=80		# dielec constant
		charge.kcal=True		# use Kcal mol-1 e-2 value
		opt.nbshift=True        # Use potential shift for nonbond interactions
		opt.P_stretch=False	
		fconst.Kd_nucl["bb"]=0.9
		fconst.Kd_nucl["sc"]=1.5

	if args.dlprakash:
		print (">>> Using Pal & Levy 2019 model. 10.1371/journal.pcbi.1006768")
		assert args.aa_pdb, "Error no pdb input --aa_pdb."
		CGlevel["prot"]=2		# CB-CA
		CGlevel["nucl"]=3		# P-S-B
		rad["CA"]=1.9			# 3.8 A excl vol rad
		rad["CB"]=1.5			# 3.0 A excl vol rad
		CB_far=True			# CB at farthest SC atom 
		CB_chiral=True		# improp dihed for CAi-1 CAi+1 CAi CBi
		charge.CB=True		# Charge on CB
		charge.P=True			#charge on P
		excl_rule=2			# Excl volume Arith. Mean
		fconst.Kd_prot["mf"]=1.0	#factor to divide 3 multiplicity dihed term
		prot_contmap.W=False			# contact weight
		prot_contmap.cutoff=4.5	# cutoff
		prot_contmap.type=1		# Calculate from all-atom structure
		prot_contmap.func=2		# Use LJ 10-12 pairs
		nucl_contmap.type=-1		# Do not calculate
		inter_contmap.type=-1		# Do not calculate
		charge.debye=True		# Use DH-electrostatics
		charge.dielec=78		# dielectric constant
		charge.iconc=0.1		# concentration
		charge.kcal=True		# use Kcal mol-1 e-2 value
		opt.nonbond=True	#write custom nonbond from file
		opt.interface=True
		opt.P_stretch=False	# set P_P_P_P dihed to 180
		ModelDir("dlprakash/adj_nbnb.stackparams.dat").copy2("interactions.pairs.dat")
		ModelDir("dlprakash/inter_nbcb.stackparams.dat").copy2("interactions.interface.dat")
		ModelDir("dlprakash/codonduplex.bpairparams.dat").copy2("interactions.nonbond.dat")
		opt.codon_pairs=True

	if args.azia2009:
		print (">>> Using Azia & Levy 2009 CA-CB model. 10.1006/jmbi.2000.3693")
		uniqtype=True
		fconst.Kr_prot=0.7**12

	if args.reddy2016:
		print (">>> Using Maity & Reddy 2016 SOP-SCP model. 10.1021/jacs.5b11300")
		if args.idp_seq: 
			print (">>> IDR-sequence given. Using Baidya-Reddy 2022 SOP-SCP-IDP model for IDRs")
			args.sop_idr=True
			pass
		CGlevel["prot"]=2
		#if args.opensmog: args.denesyuk2013=True
		#else: 
		CGlevel["nucl"]=0
		CB_com=True
		bond_function=8
		prot_contmap.cutoff=8.0
		prot_contmap.type=2
		prot_contmap.func=1
		prot_contmap.custom_pairs=True
		excl_rule=2
		opt.btparams=True
		charge.CB=True
		CB_gly=True
		fconst.Kb_prot=20.0*fconst.caltoj
		fconst.Kr_prot=1.0*fconst.caltoj
		cg_radii=True
		charge.debye=True
		charge.dielec=10
		charge.iconc=0.01		# M
		charge.debye_temp=300	#K
		CG_mass=True
		ModelDir("reddy2016/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2016/sopsc.cgmass.dat").copy2("cgmass.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.pairs.dat")

	if args.baul2019 or args.baidya2022:
		print (">>> Using Baul et. al. 2019/ Baidya-Reddy 2022 SOP-SCP-IDP model.")
		CGlevel["prot"]=2
		CGlevel["nucl"]=0
		bond_function=8
		prot_contmap.cutoff=8.0 #will not be used
		prot_contmap.type=-1	#contacts not used
		prot_contmap.func=1		#Use LJ 6-11 for nonbond interactions
		prot_contmap.custom_pairs=True	#use custom eps and/or signma values
		excl_rule=2
		opt.btparams=True
		rad["CA"]=1.9 #A
		cg_radii=True
		charge.CB=True
		CB_gly=False
		fconst.Kb_prot=20.0*fconst.caltoj
		fconst.Kr_prot=1.0*fconst.caltoj
		charge.debye=True
		charge.dielec=78
		charge.iconc=0.15	#M
		charge.debye_temp=300	#K
		opt.nonbond=True
		CG_mass=True
		ModelDir("reddy2016/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2016/sopsc.cgmass.dat").copy2("cgmass.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.nonbond.dat")

	if args.sop_idr:
		print (">>> Using Reddy-Thirumalai SOP-SC for ordered regions and Baidya-Reddy SOP-IDP for IDRs.")
		CGlevel["prot"]=2
		#if args.opensmog: args.denesyuk2013=True
		#else: 
		CGlevel["nucl"]=0
		bond_function=8
		prot_contmap.cutoff=8.0
		prot_contmap.type=2
		prot_contmap.func=1
		prot_contmap.custom_pairs=True
		excl_rule=2
		opt.btparams=True
		charge.CB=True
		CB_gly=False
		CB_atom=True
		fconst.Kb_prot=20.0*fconst.caltoj
		fconst.Kr_prot=1.0*fconst.caltoj
		cg_radii=True
		charge.debye=True
		charge.dielec=78
		charge.iconc=0.15	#M
		charge.debye_temp=300	#K
		CG_mass=True
		ModelDir("reddy2016/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2016/sopsc.cgmass.dat").copy2("cgmass.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.pairs.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.nonbond.dat")

	if args.baratam2024:
		print (">>> Using Baratam & Srivastava SOP-MULTI IDR model.")
		CGlevel["prot"]=2
		#if args.opensmog: args.denesyuk2013=True
		#else: 
		CGlevel["nucl"]=0
		bond_function=8
		prot_contmap.cutoff=8.0
		prot_contmap.type=2
		prot_contmap.func=1
		prot_contmap.custom_pairs=True
		excl_rule=2
		opt.btparams=True
		charge.CB=True
		CB_gly=False
		CB_atom=True
		fconst.Kb_prot=20.0*fconst.caltoj
		fconst.Kr_prot=1.0*fconst.caltoj
		cg_radii=True
		charge.debye=True
		charge.dielec=78
		charge.iconc=0.15	#M
		charge.debye_temp=300	#K
		ModelDir("reddy2016/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.pairs.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.nonbond.dat")

	if args.denesyuk2013 or args.chakraborty2018:
		print (">>> Using TIS model. \n\t 1) Denesyuk & Thirumalai 2013 for RNA. \n\t 2) Chakraborty & Thirumalai 2018 for DNA.")
		print ("Currently this model is work in progress");exit()
		CGlevel["prot"]=2
		args.denesyuk2013=True
		bond_function=8
		prot_contmap.cutoff=8.0
		prot_contmap.type=2
		prot_contmap.func=1
		prot_contmap.custom_pairs=True
		nucl_contmap.cutoff=8.0
		nucl_contmap.type=2
		nucl_contmap.func=1
		nucl_contmap.custom_pairs=True
		opt.codon_pairs=True
		excl_rule=2
		opt.btparams=True
		charge.CB=True
		charge.P=True
		fconst.Kb_prot=20.0*fconst.caltoj
		fconst.Kr_prot=1.0*fconst.caltoj
		fconst.Kr_nucl=1.0*fconst.caltoj
		cg_radii=True
		charge.debye=True
		charge.dielec=10
		charge.iconc=0.01		# M
		ModelDir("reddy2016/sopsc.radii.dat").copy2("radii.dat")
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.pairs.dat")

	if args.banerjee2023:
		print (">>> Using Banerjee & Gosavi 2023 self-peptide approach.")
		print ("Currently this models is work in progress");exit()
		assert args.aa_pdb, "Error no pdb input --aa_pdb"
		assert args.idp_seq, "Error no peptide seq/range given. See example/Banjerjee2023"
		#fixed params can't be overwritten
		CGlevel["prot"]=1		# CA_only	
		CGlevel["nucl"]=0		# No RNA/DNA
		rad["CA"]=2.0			# 4.0 A excl vol rad
		prot_contmap.W=True			# weighted 
		prot_contmap.cutoff=4.5	# 4.5 A
		prot_contmap.cutofftype=1	# all-atom contacts mapped to CG
		prot_contmap.contfunc=2	# LJ 10-12

	""" presets end here """

	""" Work in progress """

	assert not args.uniqtype,\
		"Sorry, these options are still not implemented"
	"""####"""

	if args.uniqtype: uniqtype=True
	if args.excl_rule: 
		excl_rule=int(args.excl_rule)
		assert excl_rule in (1,2), "Error: Choose correct exclusion rule. Use 1: Geometric mean or 2: Arithmatic mean"

	if args.prot_cg: 
		CGlevel["prot"]=int(args.prot_cg)
		if CGlevel["prot"] == 1: print (">>> Using CA-only model for protein. All other CB parameters will be ingnored.")
		elif CGlevel["prot"] == 2: print (">>> Using CB-CA model for protein.")
	if args.nucl_cg:
		CGlevel["nucl"]=int(args.nucl_cg)
		assert CGlevel["nucl"] in (1,3,5), "Error. RNA/DNA only supports 1,3 or 5 as atom types"
		if CGlevel["nucl"] == 1: print (">>> Using P-only model for protein. All other parameters will be ingnored.")
		elif CGlevel["nucl"] == 3: print (">>> Using 3-bead P-S-B model for RNA/DNA.")
		elif CGlevel["nucl"] == 5: print (">>> Using 2-bead P-S and 3 beads per Base for RNA/DNA.")

	if args.interface: 
		with open("interactions.interface.dat","w+") as fout:
			fout.write(open(args.interface).read())
		opt.interface=True

	if args.Kb_prot:fconst.Kb_prot=float(args.Kb_prot)
	if args.Ka_prot:fconst.Ka_prot=float(args.Ka_prot)
	if args.Kd_bb_prot:fconst.Kd_prot["bb"]=float(args.Kd_bb_prot)
	if args.Kd_sc_prot:fconst.Kd_prot["Sc"]=float(args.Kd_sc_prot)
	else: fconst.Kd_prot["Sc"]=fconst.Ka_prot
	if args.mulfac_prot:fconst.Kd_prot["mf"]=float(args.mulfac_prot)
	if args.Kr_prot:fconst.Kr_prot=float(args.Kr_prot)

	if args.cutoff:
		prot_contmap.cutoff=float(args.cutoff)
		nucl_contmap.cutoff=float(args.cutoff)
		inter_contmap.cutoff=float(args.cutoff)
	if args.cutoff_p: prot_contmap.cutoff=float(args.cutoff_p)
	if args.cutoff_n: nucl_contmap.cutoff=float(args.cutoff_n)
	if args.cutoff_i: inter_contmap.cutoff=float(args.cutoff_n)
	if args.cmap: 
		prot_contmap.type,prot_contmap.file=0,args.cmap.copy()
		nucl_contmap.type,nucl_contmap.file=0,args.cmap.copy()
		#inter_contmap.type,inter_contmap.file=0,args.cmap.copy()
	if args.cmap_p: 
		assert not args.cmap, "Error, --cmap and --cmap_p both cannot used simultanously"
		prot_contmap.type,prot_contmap.file=0,args.cmap_p
	if args.cmap_n:
		assert not args.cmap, "Error, --cmap and --cmap_n both cannot used simultanously"
		nucl_contmap.type,nucl_contmap.file=0,args.cmap_n
	if args.cmap_i: inter_contmap.type,inter_contmap.file=0,args.cmap_i
	if args.cutofftype:
		prot_contmap.type=int(args.cutofftype)
		nucl_contmap.type=int(args.cutofftype)
		#if args.control_run: inter_contmap.type=int(args.cutofftype)
	if args.cutofftype_p: prot_contmap.type=int(args.cutofftype_p)
	if args.cutofftype_n: nucl_contmap.type=int(args.cutofftype_n)
	if args.cutofftype_i: inter_contmap.type=int(args.cutofftype_i)
	if not args.cutoff:
		if prot_contmap.type==2 and not args.cutoff_p: prot_contmap.cutoff=8.0
		if nucl_contmap.type==2 and not args.cutoff_n: nucl_contmap.cutoff=8.0
		if inter_contmap.type==2 and not args.cutoff_i: inter_contmap.cutoff=8.0

	if len(prot_contmap.file) > 0: assert prot_contmap.type == 0, "Error, Use type 0 if giving cmap file"
	if len(nucl_contmap.file) > 0: assert nucl_contmap.type == 0, "Error, Use type 0 if giving cmap file"
	if len(inter_contmap.file) > 0: assert inter_contmap.type == 0, "Error, Use type 0 if giving cmap file"
	if args.W_cont: prot_contmap.W,nucl_contmap.W=True,True
	if args.W_cont_p: prot_contmap.W=True
	if args.W_cont_n: nucl_contmap.W=True
	if args.W_cont_i: inter_contmap.W=True
	if args.scaling: prot_contmap.scale,nucl_contmap.scale=float(args.scaling),float(args.scaling)
	if args.scaling_p: prot_contmap.scale=float(args.scaling_p)
	if args.scaling_n: nucl_contmap.scale=float(args.scaling_n)
	if args.scaling_i: inter_contmap.scale=float(args.scaling_i)
	if args.contfunc: prot_contmap.func,nucl_contmap.func=int(args.contfunc),int(args.contfunc)
	if args.contfunc_p: prot_contmap.func=int(args.contfunc_p)
	if args.contfunc_n: nucl_contmap.func=int(args.contfunc_n)
	if args.contfunc_i: inter_contmap.func=int(args.contfunc_i)
	if inter_contmap.func==-1: 
		if CGlevel["prot"]!=0: inter_contmap.func=prot_contmap.func
		elif CGlevel["nucl"]!=0: inter_contmap.func=nucl_contmap.func
	if prot_contmap.func==7 and prot_contmap.type!=0: prot_contmap.func=6
	if nucl_contmap.func==7 and nucl_contmap.type!=0: nucl_contmap.func=6
	if inter_contmap.func==7 and inter_contmap.type!=0: inter_contmap.func=6
	assert (prot_contmap.func in [1,2,6,7])
	assert (nucl_contmap.func in [1,2,6,7])
	assert (inter_contmap.func in [1,2,6,7])

	if args.nbfunc:
		nonbond_function=int(args.nbfunc)
		if args.opensmog: assert nonbond_function in [1,2,3,6,7], "Error, nbfunc input not supported"
		else: assert nonbond_function in [1,2,3],"Error, nbfunc input not supported by GROMACS"
	else:
		if nonbond_function==-1:
			if opt.opensmog: nonbond_function=prot_contmap.func
			else:
				if prot_contmap.func in (1,2): nonbond_function=prot_contmap.func
				elif nucl_contmap.func in (1,2): nonbond_function=nucl_contmap.func
				elif inter_contmap.func in (1,2): nonbond_function=inter_contmap.func
				else: nonbond_function=2
			print (">>non-bond function --nbfunc not given. Using %d."%nonbond_function)
	prot_contmap.nbfunc,nucl_contmap.nbfunc,inter_contmap.nbfunc=nonbond_function,nonbond_function,nonbond_function

	if args.bfunc: 
		bond_function=int(args.bfunc)
		assert bond_function in (1,7,8), "Only Harmonic (1) and FENE (7) are supported bond length potentials"

	if args.interaction: prot_contmap.custom_pairs=True
	if args.btparams:
		ModelDir("reddy2016/sopsc.btparams.dat").copy2("interactions.dat")
		opt.btparams=True
		prot_contmap.custom_pairs=True
	if args.mjparams:
		ModelDir("reddy2016/sopsc.mjparams.dat").copy2("interactions.dat")
		opt.mjparams=True
		prot_contmap.custom_pairs=True
	
	if args.cg_radii:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		cg_radii=True

	if args.CB_gly:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		skip_glycine=False
		CB_gly=True
		print ("WARNING: Using CB for Glycines!! Make sure the all-atom pdb contains H-atom (HB)")
	opt.CB_gly=CB_gly
	
	if args.CB_chiral:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		CB_chiral=True

	if args.CA_charge: 
		assert CGlevel["prot"] != 2, ("ERROR. charge on CA is ont supported for CA+CB graining")
		charge.CA=True
	if args.CB_charge:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		charge.CB=True
	if args.P_charge: 
		charge.P=True
		if args.PPelec: charge.PP=True

	
	if args.hphobic or args.hpdist or args.hpstrength:
		if not args.hphobic: print ("WARNING! Hydrophobic parameters given without using --hphobic. Turing --hphobic ON for the given CG-level")
		opt.hphobic=True
	if opt.hphobic:
		if CGlevel["prot"]==1: opt.CA_hp=True
		elif CGlevel["prot"]==2: opt.CB_hp=True
	if opt.hphobic: 
		print (">>> Using non-native hydrophobic interactions with --nbfunc %d"%nonbond_function)
		if nonbond_function!=7 or not args.opensmog:
			input("WARNING!. Using non-native hydrophobic interactions with LJ-like potential might cause problems. Try Gaussian non-bonded interactions with OpenMOG instead  (--nbfunc 7 --opensmog). Press [enter] to contuue or ctrl+C to abort.")
		if args.hpdist: hpdist=float(args.hpdist)
		else: hpdist=5.0	#A
		if args.hpstrength: hpstrength=float(args.hpstrength)
		else: hpstrength=0.8	
		opt.nonbond=True
		with open("interactions.nonbond.dat","w+") as fout:
			fout.write("#a1 a1 eps sig(A)\n")
			for a1 in "FAMILYVW":
				for a2 in "FAMILYVW":
					fout.write("CB%s CB%s %.2f %.2f\n"%(a1,a2,hpstrength,hpdist))

	if args.CA_com:CA_com=True

	if args.CA_rad: rad["CA"]=float(args.CA_rad)
	if args.CB_rad:
		if CGlevel["prot"] != 2: print ("WARNING: User opted for only-CA model. Ignoring all C-beta parameters.")
		rad["CB"]=float(args.CB_rad)
		aa_resi=Prot_Data().amino_acid_dict
		with open("radii.dat","w+") as fout:
			print (">>> C-beta radius given via user input. Storing in radii.dat")
			for i in aa_resi: fout.write('%s%4.2f\n' % (i.ljust(4),rad["CB"]))
		cg_radii=True	#Read CB radius from radii.dat	

	if charge.CA or opt.CA_hp:
		print (">>> Adding non-native potential for CA. Creating residue wise CA-types (refered as CBX)")
		opt.CAX=True		
		aa_resi=Prot_Data().amino_acid_dict
		with open("radii.dat","w+") as fout:
			for i in aa_resi: fout.write('%s%4.2f\n' % (i.ljust(4),rad["CA"]))
		cg_radii=True
		if charge.CA: charge.CB=True

	if cg_radii:
		with open("radii.dat") as fin:
			rad.update({x.split()[0]:float(x.split()[1]) for x in fin})
	else:
		aa_resi=Prot_Data().amino_acid_dict
		rad.update({"CB"+aa_resi[x]:rad["CB"] for x in aa_resi})

	if args.cg_mass: CG_mass=True
	if CG_mass:
		opt.mass.update({l.split()[0]:float(l.split()[1]) for l in open("cgmass.dat") if l.strip()!=str() and not l.strip().startswith("#")})

	#rad adding nucl rads
	rad.update({"B"+b:rad["Bpu"] for b in "AG" if "B"+b not in rad})
	rad.update({"B"+b:rad["Bpy"] for b in "UTC" if "B"+b not in rad})
	rad.update({"S"+b:rad["S"] for b in "AGUTC" if "S"+b not in rad})

	#converting A to nm
	for x in rad: rad[x]=np.round(0.1*rad[x],3)

	#if args.dsb: dsb=True

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
		nucl_pos["Bpu"]=str(args.Bpu_pos)
		assert nucl_pos["Bpu"] in Nucl_Data().pur_atom, "Error: Wrong Atom name entered for Purine. Use --help option to check allowed atom names"
	if args.Bpy_pos:
		nucl_pos["Bpy"]=str(args.Bpy_pos)
		assert nucl_pos["Bpy"] in Nucl_Data().pyr_atom, "Error: Wrong Atom name entered for Pyrimidine. Use --help option to check allowed atom names"
	if args.P_pos:
		nucl_pos["P"]=str(args.P_pos)
		assert nucl_pos["P"] in Nucl_Data().phos_atom, "Error: Wrong Atom name entered for Phosphate. Use --help option to check allowed atom names"
	if args.S_pos:
		nucl_pos["S"]=str(args.S_pos)
		assert nucl_pos["S"] in Nucl_Data().sug_atom, "Error: Wrong Atom name entered for Sugar. Use --help option to check allowed atom names"
	CG_pos={"Bpu":nucl_pos["Bpu"],"Bpy":nucl_pos["Bpy"],"S":nucl_pos["S"],"P":nucl_pos["P"]}	
	if args.P_stretch: opt.P_stretch=True

	#Force constants
	if args.Kb_nucl: fconst.Kb_nucl=float(args.Kb_nucl)
	if args.Ka_nucl: fconst.Ka_nucl=float(args.Ka_nucl)
	if args.Kd_sc_nucl: fconst.Kd_nucl["sc"]=float(args.Kd_sc_nucl)
	if args.Kd_bb_nucl: fconst.Kd_nucl["bb"]=float(args.Kd_bb_nucl)
	if args.mulfac_nucl:fconst.Kd_nucl["mf"]=float(args.mulfac_nucl)
	if args.Kr_nucl:	fconst.Kr_nucl=float(args.Kr_nucl)
	if args.P_rad: rad["P"]=float(args.P_rad)
	if args.S_rad: rad["S"]=float(args.S_rad)
	if args.Bpu_rad: rad["Bpu"]=float(args.Bpu_rad)
	if args.Bpy_rad: rad["Bpy"]=float(args.Bpy_rad)
	#if args.pistacklen: rad["stack"]=float(args.pistacklen)
	
	#Solvent and ionic params
	if args.dielec:	charge.dielec=float(args.dielec)
	if args.iconc: charge.iconc=float(args.iconc)
	if args.irad: charge.irad=float(args.irad)
	if args.debye: charge.debye=True
	if args.debye_temp: charge.debye_temp=float(args.debye_temp)
	if args.debye_length: charge.inv_dl=1.0/float(args.debye_length)
	if args.elec_kcal: charge.kcal=True

	#input structure file
	pdbdata=[]
	if prot_contmap.type == 1: assert args.aa_pdb, "Error. No all-atom pdb provided. --aa_pdb"
	if args.aa_pdb: 
		assert not args.cg_pdb, "Error, Cannot use all atom and CG pdb together"
		nfiles=len(args.aa_pdb)
		for i in range(nfiles):
			pdbdata.append(PDB_IO(fileindex=i,nfiles=nfiles))
			pdbdata[-1].loadfile(infile=args.aa_pdb[i],renumber=True,CBgly=CB_gly)
	elif args.cg_pdb: 
		nfiles=len(args.cg_pdb)
		for i in range(nfiles):
			pdbdata.append(PDB_IO(fileindex=i,nfiles=nfiles))
			pdbdata[-1].loadfile(infile=args.cg_pdb[i],renumber=True,CBgly=CB_gly)
	else:
		pdbdata=[PDB_IO()]
		if args.idp_seq:
			assert args.baul2019 or args.baidya2022 or args.baratam2024 or args.sop_idr,\
				"Error, building CG PDB using idp_seq only supported with --baul2019 or --baratam2024"
			assert args.idp_seq.endswith((".fa",".fasta"))
			pdbdata[0].buildProtIDR(fasta=args.idp_seq,rad=rad,CBgly=CB_gly)
		else: 
			if args.baul2019 or args.baidya2022: assert args.idp_seq, "Provide --aa_pdb, --cg_pdb or --idp_seq"
			assert args.aa_pdb or args.cg_pdb, ("Error. Provide all-atom or coarse-grain pdb. --aa_pdb/--cg_pdb")

	if args.control or args.gen_cg:	#Use Protein with DNA/RNA bound at natve site
		opt.control_run=True
		assert not args.custom_nuc, "Error: --custom_nuc cannot be used with --control"
		opt.custom_nuc=False
	else:
		opt.custom_nuc=True
		if args.custom_nuc:
			custom_nucl_file=PDB_IO()
			custom_nucl_file.loadfile(infile=args.custom_nuc,renumber=True,CBgly=CB_gly)
			assert len(pdbdata)==1, \
				"Error: --custom_nuc file is only supported with single --aa_pdb/--cg_pdb. \
				Give your RNA/DNA file directly to --aa_pdb/--cc_pdb."
			print (">> Note: Using RNA/DNA from",custom_nucl_file.nucl.pdbfile)
			pdbdata[0].nucl=custom_nucl_file.nucl
			del(custom_nucl_file)
			pdbdata[0].coordinateTransform()
		else:
			if pdbdata[0].prot.pdbfile!="" and  pdbdata[0].nucl.pdbfile != "":
				print (">> Note: custom_nuc option being used without input, will use unbound version of native RNA/DNA ")
				pdbdata[0].coordinateTransform()

	if args.banerjee2023:
		assert len(pdbdata)==1, "Error, model supports only 1 PDB input."
		segmentdata=PDB_IO()
		segmentdata.extractPDBSegment(fasta=args.idp_seq,data=pdbdata[0])
		pdbdata.append(segmentdata)
		if args.nmol: args.nmol=[1]+args.nmol
		else: args.nmol=[1,1]

	if args.nmol:
		assert len(args.nmol)==len(pdbdata), "Error, number of values given to --nmol should be equal to values given to --aa_pdb/--cg_pdb"
		args.nmol=np.int_(args.nmol)
	else: args.nmol = np.ones(len(pdbdata),dtype=int)
	Nmol['prot'],Nmol["nucl"]=list(args.nmol),list(args.nmol)
	if CGlevel["prot"]==0: Nmol["prot"]=list(np.zeros(len(Nmol["prot"])))
	if CGlevel["nucl"]==0: Nmol["nucl"]=list(np.zeros(len(Nmol["nucl"])))
	for i in range(len(pdbdata)):
		if len(pdbdata[i].prot.lines)==0: Nmol["prot"][i]=0
		if len(pdbdata[i].nucl.lines)==0: Nmol["nucl"][i]=0

	if args.sym_intra:
		opt.intra_symmetrize=True
	if args.dswap:
		opt.intra_symmetrize=True
		for i in range(len(pdbdata)):
			if Nmol["prot"][i]==1: Nmol["prot"][i]=2
			if Nmol["nucl"][i]==1: Nmol["prot"][i]=2
		#if CGlevel["nucl"] != 0:
			#print ("--dwap not supported for RNA/DNA. Will only be applied to protein topology")
		print ("--dswap assumes the input protein is a single unit. For adding 2 chains, use 'genbox' of 'gmx insert-molecules'")
	assert len(Nmol["prot"])==len(Nmol["nucl"])
	if len(Nmol["prot"])==1: 
		if inter_contmap.type not in (-1,0): inter_contmap.type=-1

	#output grofiles
	if args.box: 
		#assert float(args.box)>0
		box_width=float(args.box)
	else: box_width=500.0
	if args.voxel:
		assert float(args.voxel)>0, "Error, voxel width cannot be <=0."
		voxel_width=float(args.voxel)
	else: voxel_width=1.618 #A
	if args.outgro: grofile=str(args.outgro)
	else: grofile="gromacs.gro"

	#set GROMACS .top file
	if args.outtop: topfile=str(args.outtop)
	else:topfile='gromacs.top'
	
	#set OpenSMOG XML file
	if args.opensmog:
		opt.opensmog = True
		if args.outxml: opt.xmlfile=str(args.outxml)
		else: opt.xmlfile="opensmog.xml"
		if not args.outtop: topfile="opensmog.top"
		if not args.outgro: grofile="opensmog.gro"
	if args.dihed2xml:
		assert opt.opensmog, "Error --dihed2xml only suuported with --opensmog flag, with modified version of OpenSMOG:https://github.com/sglabncbs/OpenSMOGmod"
		opt.dihed2xml = True
	if args.nbshift:
		assert opt.opensmog, "Error --nbshift only suuported with --opensmog flag"
		opt.nbshift=True

	nfiles=len(pdbdata)
	for i in range(nfiles):
		if args.cmap:
			cmap_files=pdbdata[i].cmapSplit(cmap=args.cmap[i])
			prot_contmap.file[i]=cmap_files["prot"]
			nucl_contmap.file[i]=cmap_files["nucl"]
			if not args.cmap_i:
				inter_contmap.file=cmap_files["inter"]			
		if prot_contmap.type!=0: prot_contmap.file.append("")
		if nucl_contmap.type!=0: nucl_contmap.file.append("")
	if inter_contmap.type!=-1:
		check_inter_contmap=len(open(inter_contmap.file).readlines())
		if check_inter_contmap==0:
			inter_contmap.file=""
			inter_contmap.type=-1

	#write CG file
	for i in range(nfiles):
		pdbdata[i].write_CG_protfile(CGlevel=CGlevel,CAcom=CA_com,CBcom=CB_com,CBfar=CB_far,CBgly=CB_gly,nucl_pos=nucl_pos,outgro=grofile)
	
	if not args.gen_cg:				
		if args.clementi2000:
			top=Clementi2000(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		elif args.afsar2008:
			top=ZarrineAfsar2008(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		elif args.pal2019 or args.dlprakash:
			if nucl_contmap.type==-1: top=Pal2019(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			if nucl_contmap.type>=0:
				print ("WARNING: Default Pal2019 only includes base-stacking intra RNA/DNA interactions. User is forcing to calculate a different RNA/DNA contact map")
				top=Topology(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		elif args.reddy2016:
			top=Reddy2016(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		elif args.baul2019 or args.baidya2022:
			top=Baul2019(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		elif args.baratam2024 or args.sop_idr:
			assert args.aa_pdb or args.cg_pdb, "Error, SOP-MULTI needs input structure"
			assert args.idp_seq, "Error, SOP-MULTI needs sequence and residue range for IDR"
			idrdata=PDB_IO()
			idrdata.buildProtIDR(fasta=args.idp_seq,rad=rad,CBgly=CB_gly)
			idrdata=idrdata.prot
			if args.baratam2024: 
				top=Baratam2024(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt,idrdata=idrdata)
			else:
				top=SOPSC_IDR(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt,idrdata=idrdata)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
			unfolded=PDB_IO()
			unfolded.buildProtIDR(fasta="unfolded.fa",rad=rad,CBgly=CB_gly,topbonds=top.bonds[0])
			unfolded.write_CG_protfile(CGlevel=CGlevel,CAcom=CA_com,CBcom=CB_com,CBfar=CB_far,CBgly=CB_gly,nucl_pos=nucl_pos,outgro=grofile)
		elif args.banerjee2023:
			top=Banerjee2023(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
		else:
			top=Topology(allatomdata=pdbdata,fconst=fconst,CGlevel=CGlevel,Nmol=Nmol,cmap=(prot_contmap,nucl_contmap,inter_contmap),opt=opt)
			topdata=top.write_topfile(outtop=topfile,excl=excl_rule,rad=rad,charge=charge,bond_function=bond_function,CBchiral=CB_chiral)
	
	groups=list()
	if sum(Nmol['prot'])!=0: groups.append("Protein")
	if sum(Nmol['nucl'])!=0:
		groups+=([["RNA","DNA"][int(y)] for x in range(nfiles) for y in pdbdata[x].nucl.deoxy])

	
	#write combined file
	molecule_order=[]
	molecule_order+=[(pdbdata[i].nucl.outgro,Nmol['nucl'][i]) for i in range(len(Nmol['nucl'])) if Nmol['nucl'][i]>0]
	molecule_order+=[(pdbdata[i].prot.outgro,Nmol['prot'][i]) for i in range(len(Nmol['prot'])) if Nmol['prot'][i]>0]
	if args.gen_cg:
		with open("molecule_order.list","w+") as fout:
			fout.write("#inp_ndx mol_typ num_mol\n")
			for i in range(len(molecule_order)):
				fout.write(" %7d %7s %7d\n"%(i,molecule_order[i][0].split("_")[0].rjust(7),molecule_order[i][1]))

	if box_width==0: fill_status=False
	else:
		fill=Fill_Box(outgro=grofile,radii=rad,box_width=box_width,voxel_width=voxel_width,order=molecule_order)
		fill_status=fill.status
		box_width=500.0
	if fill_status: print ("> Combined topology and structure files generated!!!")
	else: 
		if not args.gen_cg:
			print ("> Combined topology file(s) generated but failed to generate combined structure file. Try using genbox_commands.sh script (requires GROMACS) or run again with --gen_cg & different box width --box")
		if args.gen_cg:
			print ("> Failed to generate combined structure file. Try using genbox_commands.sh script (requires GROMACS) or run again with --gen_cg & different box width --box")
	CleanUP(grosuffix=grofile,topsuffix=topfile,xmlsuffix=opt.xmlfile,coulomb=charge,enrgrps=groups,box_width=box_width,fillstatus=fill_status,gen_cg=args.gen_cg)
if __name__ == '__main__':
    main()
