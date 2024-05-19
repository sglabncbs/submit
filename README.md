# SuBMIT: Structure Based Model(s) Input Topology #
## Package to generate Coarse-Grained Structure (.gro/.pdb) and Topology (.top/.xml) for using Enhanced Structure Based Models MD Simulations on GROMACS and OpenSMOG (OpenMM based) ##
### digvijaylp ###
 
 
<pre>
# Model presets allows user to auto-select parameters based on predefined models 
 
#CA-SBM (Clementi 2000) 
$ python submit.py --clementi2000 --aa_pdb [All-atom .pdb file] 
 
#CA-CB SOP-SC model (Reddy 2017) 
$ python submit.py --reddy2017 --aa_pdb [All-atom .pdb file]
$ python submit.py --reddy2017 --cg_pdb [Coarse-grained .pdb file]
 
#CA-CB SOP-SC-IDP model (Baidya 2022) 
$ python submit.py --aa_pdb/--cg__pdb [template AA/CG .pdb file] 
$ python submit.py --baidya2022 --idp_seq [IDP sequence .fa file]
 
#CA-CB SOP-SC-MULTI model (Baratam 2024) 
$ python submit.py --baratam2024 --idp_seq [IDP sequence .fa file (see models/baratam2024/example.fa)]

#CA-CB Protein+RNA/DNA model with DH-electrostatics (Pal 2019) 
$ python submit.py --pal2019 --aa_pdb [protein All-atom .pdb file] --custom_nuc [RNA/DNA all-atom .pdb file] 
 
#For every model, predefined parameters can be customized. For example, for chanding angle force constant in Pal 2019 model 
$ python submit.py --pal2019 --aa_pdb [protein All-atom .pdb file] --Ka_prot 40 
 
#For testing your own model or tweaking predefined ones, refer to options in --help 
$ python submit.py --help 
 
 
optional arguments:
  -h, --help            show this help message and exit
  --clementi2000, -clementi2000, --calpha_go2000, -calpha_go2000
                        Clementi et. al. 2000 CA-only model
  --azia2009, -azia2009
                        Azia 2009 CB-CA + Debye-Huckel model
  --pal2019, -pal2019, --levy2019, -levy2019
                        Pal & Levy 2019 Protein CB-CA & RNA/DNA P-S-B model
  --reddy2017, -reddy2017, --sopsc2017, -sopsc2017
                        Reddy & Thirumalai 2017 SOP-SC CA-CB
  --baidya2022, -baidya2022, --sopsc_idp, -sopsc_idp
                        Baidya & Reddy 2022 SOP-SC-IDP CA-CB
  --baratam2024, -baratam2024, --sop-multi, -sop-multi
                        Baratam & Srivastava 2024 SOP-MULTI CA-CB
  --dlprakash, -dlprakash
                        Codon pairs (duplex based weight) for Pal2019
  --CA_rad CA_RAD, -CA_rad CA_RAD
                        User defined radius for C-alpha (same for all beads) in Angstrom. Default: 4.0A
  --CA_com, -CA_com     Place C-alpha at COM of backbone. Default: False
  --CB_rad CB_RAD, -CB_rad CB_RAD
                        User defined radius for C-beta (same for all beads) in Angstrom for prot_cg 2. Default: Statistically Derived
                        for each AA-residue
  --CB_radii, -CB_radii
                        User defined C-beta radii from radii.dat (AA-3-letter-code radius-in-Angsrtom)
  --Kb_prot KB_PROT, -Kb_prot KB_PROT, --Kb KB_PROT, -Kb KB_PROT
                        User defined force constant K_bond for Proteins
  --Ka_prot KA_PROT, -Ka_prot KA_PROT, --Ka KA_PROT, -Ka KA_PROT
                        User defined force constant K_angle for Proteins
  --Kd_bb_prot KD_BB_PROT, -Kd_bb_prot KD_BB_PROT, --Kd KD_BB_PROT, -Kd KD_BB_PROT
                        User defined force constant K_dihedral for Proteins
  --Kd_sc_prot KD_SC_PROT, -Kd_sc_prot KD_SC_PROT, --Kd_chiral KD_SC_PROT, -Kd_chiral KD_SC_PROT
                        User defined force constant K_dihedral for Proteins
  --mulfac_prot MULFAC_PROT, -mulfac_prot MULFAC_PROT
                        User defined Multiplicity scale factor of K_dihedral/mulfac_prot for Proteins
  --CB_chiral, -CB_chiral
                        Improper dihedral for CB sidechain chirality. Default: False
  --uniqtype, -uniqtype
                        Each atom has unique atom type (only use for large systems)
  --bfunc BFUNC, -bfunc BFUNC
                        Bond function 1: harnomic, 7: FENE. Default: 1 (Harmonic)
  --idp_seq IDP_SEQ, -idp_seq IDP_SEQ
                        User input sequence for building IDRs/helices etc.
  --cutoff CUTOFF, -cutoff CUTOFF
                        User defined Cut-off (in Angstrom) for contact-map generation. Default: 4.5A
  --cutofftype CUTOFFTYPE, -cutofftype CUTOFFTYPE
                        -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1
  --W_cont, -W_cont     Weight (and normalize) CG contacts based on all atom contacts
  --cmap CMAP, -cmap CMAP
                        User defined cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)
  --scaling SCALING, -scaling SCALING
                        User defined scaling for mapping to all-atom contact-map.
  --contfunc CONTFUNC, -contfunc CONTFUNC
                        1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default: 2
  --cutoff_p CUTOFF_P, -cutoff_p CUTOFF_P
                        User defined Cut-off (in Angstrom) for Protein contact-map generation. Default: 4.5A
  --cutofftype_p CUTOFFTYPE_P, -cutofftype_p CUTOFFTYPE_P
                        For Proteins: -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1
  --W_cont_p, -W_cont_p
                        Weight (and normalize) Protein CG contacts based on all atom contacts
  --cmap_p CMAP_P, -cmap_p CMAP_P
                        User defined Protein cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)
  --scaling_p SCALING_P, -scaling_p SCALING_P
                        User defined scaling for mapping to all-atom contact-map.
  --contfunc_p CONTFUNC_P, -contfunc_p CONTFUNC_P
                        Proteins. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default:
                        2
  --cutoff_n CUTOFF_N, -cutoff_n CUTOFF_N
                        User defined Cut-off (in Angstrom) for RNA/DNA contact-map generation. Default: 4.5A
  --cutofftype_n CUTOFFTYPE_N, -cutofftype_n CUTOFFTYPE_N
                        For RNA/DNA. -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1
  --W_cont_n, -W_cont_n
                        Weight (and normalize) RNA/DNA CG contacts based on all atom contacts
  --cmap_n CMAP_N, -cmap_n CMAP_N
                        User defined RNA/DNA cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)
  --scaling_n SCALING_N, -scaling_n SCALING_N
                        User RNA/DNA defined scaling for mapping to all-atom contact-map.
  --contfunc_n CONTFUNC_N, -contfunc_n CONTFUNC_N
                        RNA/DNA. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default: 2
  --cutoff_i CUTOFF_I, -cutoff_i CUTOFF_I
                        User defined Cut-off (in Angstrom) for Protein RNA/DNA interface contact-map generation. Default: 4.5A
  --cutofftype_i CUTOFFTYPE_I, -cutofftype_i CUTOFFTYPE_I
                        For Protein RNA/DNA interface. -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain .
                        Default: 1
  --W_cont_i, -W_cont_i
                        Weight (and normalize) Protein RNA/DNA interface CG contacts based on all atom contacts
  --cmap_i CMAP_I, -cmap_i CMAP_I
                        User defined Protein RNA/DNA interface cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)
  --scaling_i SCALING_I, -scaling_i SCALING_I
                        User Protein RNA/DNA interface defined scaling for mapping to all-atom contact-map.
  --contfunc_i CONTFUNC_I, -contfunc_i CONTFUNC_I
                        Protein RNA/DNA interface. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi
                        Gauss . Default: 2
  --prot_cg PROT_CG, -prot_cg PROT_CG
                        Level of Amino-acid coarse-graining 1 for CA-only, 2 for CA+CB. Dafault: 2 (CA+CB)
  --CB_com, -CB_com     Put C-beta at side-chain COM (no hydrogens). Default: False
  --CB_far, -CB_far     Place C-beta on farthest non-hydrogen atom. Default: False
  --dsb, -dsb           Use desolvation barrier potential for contacts. Default: False
  --native_ca NATIVE_CA, -native_ca NATIVE_CA
                        Native file with only C-alphas. Just grep pdb.
  --aa_pdb AA_PDB, -aa_pdb AA_PDB
                        User input all-atom pdbfile/gro/mmCIF e.g. 1qys.pdb
  --cg_pdb CG_PDB, -cg_pdb CG_PDB
                        User input coarse grained pdbfile
  --grotop GROTOP, -grotop GROTOP
                        Gromacs topology file output name (tool adds prefix nucl_ and prot_ for independednt file). Default:
                        gromacs.top
  --pdbgro PDBGRO, -pdbgro PDBGRO
                        Name for output .gro file.(tool adds prefix nucl_ and prot_ for independednt file). Default: gromacs.gro
  --smogxml SMOGXML, -smogxml SMOGXML
                        Name for output .xml (openSMOG) file.(tool adds prefix nucl_ and prot_ for independednt file). Default:
                        opensmog.xml (and opensmog.top)
  --opensmog, -opensmog
                        Generate files ,xml and .top files for openSMOG. Default: False
  --CB_gly, --CB_GLY, -CB_gly, -CB_GLY
                        Add C-beta for glycine (pdb-file must have H-atoms). Default: Flase
  --btparams, -btparams
                        Use Betancourt-Thirumalai interaction matrix.
  --mjparams, -mjparams
                        Use Miyazawa-Jernighan interaction matrix.
  --P_rad P_RAD, -P_rad P_RAD
                        User defined radius for Backbone Phosphate bead. Default=3.7A
  --S_rad S_RAD, -S_rad S_RAD
                        User defined radius for Backbone Sugar bead. Default=3.7A
  --Bpu_rad BPU_RAD, -Bpu_rad BPU_RAD
                        User defined radius for N-Base Purine bead. Default=1.5A
  --Bpy_rad BPY_RAD, -Bpy_rad BPY_RAD
                        User defined radius for N-Base Pyrimidine bead. Default=1.5A
  --nucl_cg NUCL_CG, -nucl_cg NUCL_CG
                        Level of Amino-acid coarse-graining 1 for P-only, 3 for P-S-B, 5 for P-S-3B. Dafault: 3 (P-S-B)
  --Kb_nucl KB_NUCL, -Kb_nucl KB_NUCL, --nKb KB_NUCL, -nKb KB_NUCL
                        User defined force constant K_bond for RNA/DNA
  --Ka_nucl KA_NUCL, -Ka_nucl KA_NUCL, --nKa KA_NUCL, -nKa KA_NUCL
                        User defined force constant K_angle for RNA/DNA. Default=20
  --Kd_sc_nucl KD_SC_NUCL, -Kd_sc_nucl KD_SC_NUCL, --nKd KD_SC_NUCL, -nKd KD_SC_NUCL
                        User defined force constant K_dihedral for Bi-Si-Si+1-Bi+1. Default=0.5
  --Kd_bb_nucl KD_BB_NUCL, -Kd_bb_nucl KD_BB_NUCL, --P_nKd KD_BB_NUCL, -P_nKd KD_BB_NUCL
                        User defined force constant K_dihedral for Backbone Pi-Pi+1-Pi+2-Pi+3. Default=0.7
  --P_stretch P_STRETCH, -P_stretch P_STRETCH
                        Stretch the backbone dihedral to 180 degrees. Default=Use native backbone dihedral
  --mulfac_nucl MULFAC_NUCL, -mulfac_nucl MULFAC_NUCL
                        User defined Multiplicity scale factor of K_dihedral for Nucleic Acids
  --Bpu_pos BPU_POS, -Bpu_pos BPU_POS
                        Put input atom of Purine [N1,C2,H2-N2,N3,C4,C5,C6,O6-N6,N7,C8,N9,COM] as position of B.
                        Default=COM(Center_of_Mass)
  --Bpy_pos BPY_POS, -Bpy_pos BPY_POS
                        Put input atom of Pyrimidine [N1,C2,O2,N3,C4,O4-N4,C5,C6,COM] as position of B. Default=COM(Center_of_Mass)
  --S_pos S_POS, -S_pos S_POS
                        Put input atom of Sugar [C1',C2',C3',C4',C5',H2'-O2',O3',O4',O5',COM] as position of S.
                        Default=COM(Center_of_Mass)
  --P_pos P_POS, -P_pos P_POS
                        Put input atom of Phosphate [P,OP1,OP2,O5',COM] group as position of P. Default=COM(Center_of_Mass)
  --pistacklen PISTACKLEN
                        pi-pi stacking length. Default=3.6A
  --debye, -debye       Use Debye-Huckel electrostatic term.
  --debye_temp DEBYE_TEMP, -debye_temp DEBYE_TEMP
                        Temperature for Debye length calculation. Default=298K
  --debye_length DEBYE_LENGTH, -debye_length DEBYE_LENGTH
                        Debye length. in (A)
  --CA_charge, -CA_charge
                        Put charges on CA for K,L,H,D,E. Default: False
  --CB_charge, -CB_charge
                        Put charges on CB for K,L,H,D,E. Default: False
  --P_charge, -P_charge
                        Negative charge on Phosphate bead. Default: False
  --iconc ICONC, -iconc ICONC
                        Solvant ion conc.(N) for Debye length calcluation. Default=0.1M
  --irad IRAD, -irad IRAD
                        Solvant ion rad for Debye length calcluation. Default=1.4A
  --dielec DIELEC, -dielec DIELEC
                        Dielectric constant of solvant. Default=70
  --hpstrength HPSTRENGTH, -hpstrength HPSTRENGTH
                        Strength with which hydrophobic contacts interact.
  --ext_conmap EXT_CONMAP, -ext_conmap EXT_CONMAP
                        External contact map in format chain res chain res
  --interaction, -interaction
                        User defined interactions in file interactions.dat.
  --dswap, -dswap       For domain swapping runs. Symmetrised SBM is generated.
  --hphobic, -hphobic   Generate hydrophobic contacts.
  --hpdist HPDIST, -hpdist HPDIST
                        Equilibrium distance for hydrophobic contacts.
  --interface, -interface
                        Takes input for Nucleiotide_Protein interface from file nucpro_interface.input.
  --custom_nuc CUSTOM_NUC, -custom_nuc CUSTOM_NUC
                        Use custom non native DNA/RNA structure Eg.: polyT.pdb. Default: Use from native structure
  --control             Use the native system as control. Use DNA/RNA bound to native protein site. --custom_nuc will be disabled.
                        Default: False (Move DNA/RNA away from native binding site)
  --excl_rule EXCL_RULE
                        Use 1: Geometric mean. 2: Arithmatic mean
  --Kr_prot KR_PROT     Krepulsion. Default=1.0
  --Kr_nucl KR_NUCL     Krepulsion. Default=1.0
</pre>
