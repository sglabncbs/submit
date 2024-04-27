# Package to generate coarse-Grain structure and topology for using Enhanced Structure Based Models MD simulations on GROMACS and OpenSMOG (OpenMM based)
# digvijaylp #
</br>
</br>
# Model presets allows user to auto-select params based on prefedined models</br>
</br>
#CA-SBN (Clementi 2000)</br>
$ python esbm.py --clementi2000 --aa_pdb <All-atom .pdb file></br>
</br>
#CA-CB SOP-SC model (Reddy 2017)</br>
$ python esbm.py --reddy2017 --aa_pdb <All-atom .pdb file></br>
$ python esbm.py --reddy2017 --cg_pdb <Coarse-grain .pdb file></br>
</br>
#CA-CB SOP-SC-IDP model (Baidya 2000)</br>
$ python esbm.py --baidya2017 --prot_seq <IDP sequence .fa file></br>
</br>
#CA-CB Protein+RNA/DNA model (Pal 2019)</br>
$ python esbm.py --pal2019 --aa_pdb <protein All-atom .pdb file> --custom_nuc <RNA/DNA all-atom .pdb file></br>
</br>
#For every model, predefined parameters can be customized. For example, for chanding angle force constant in Pal 2019 model</br>
$ python esbm.py --pal2019 --aa_pdb <protein All-atom .pdb file> --Ka_prot 40</br>
</br>
#For testing your own model or tweeking predefined ones, refer to options in --help</br>
$ python esbm.py --help</br>
</br>
</br>
optional arguments:</br>
  -h, --help           show this help message and exit</br></br>
  --clementi2000, -clementi2000, --calpha_go2000, -calpha_go2000</br>
                        Clementi et. al. 2000 CA-only model</br></br>
  --azia2009, -azia2009</br>
                        Azia 2009 CB-CA + Debye-Huckel model</br></br>
  --pal2019, -pal2019, --levy2019, -levy2019</br>
                        Pal & Levy 2019 Protein CB-CA & RNA/DNA P-S-B model</br></br>
  --reddy2017, -reddy2017, --sopsc2017, -sopsc2017</br>
                        Reddy. 2017 SOP-SC CA-CB</br></br>
  --baidya2022, -baidya2022, --sopsc_idp, -sopsc_idp</br>
                        SOP-SC-IDP CA-CB</br></br>
  --CA_rad CA_RAD, -CA_rad CA_RAD</br>
                        User defined radius for C-alpha (same for all beads) in Angstrom. Default: 4.0A</br></br>
  --CA_com, -CA_com     Place C-alpha at COM of backbone. Default: False</br></br>
  --CB_rad CB_RAD, -CB_rad CB_RAD</br>
                        User defined radius for C-beta (same for all beads) in Angstrom for prot_cg 2. Default: Statistically Derived for each AA-residue</br></br>
  --CB_radii, -CB_radii</br>
                        User defined C-beta radii from radii.dat (AA-3-letter-code radius-in-Angsrtom)</br></br>
  --Kb_prot KB_PROT, -Kb_prot KB_PROT, --Kb KB_PROT, -Kb KB_PROT</br>
                        User defined force constant K_bond for Proteins</br></br>
  --Ka_prot KA_PROT, -Ka_prot KA_PROT, --Ka KA_PROT, -Ka KA_PROT</br>
                        User defined force constant K_angle for Proteins</br></br>
  --Kd_bb_prot KD_BB_PROT, -Kd_bb_prot KD_BB_PROT, --Kd KD_BB_PROT, -Kd KD_BB_PROT</br>
                        User defined force constant K_dihedral for Proteins</br></br>
  --Kd_sc_prot KD_SC_PROT, -Kd_sc_prot KD_SC_PROT, --Kd_chiral KD_SC_PROT, -Kd_chiral KD_SC_PROT</br>
                        User defined force constant K_dihedral for Proteins</br></br>
  --mulfac_prot MULFAC_PROT, -mulfac_prot MULFAC_PROT</br>
                        User defined Multiplicity scale factor of K_dihedral/mulfac_prot for Proteins</br></br>
  --CB_chiral, -CB_chiral</br>
                        Improper dihedral for CB sidechain chirality. Default: False</br></br>
  --uniqtype, -uniqtype</br>
                        Each atom has unique atom type (only use for large systems)</br></br>
  --bfunc BFUNC, -bfunc BFUNC</br>
                        Bond function 1: harnomic, 7: FENE. Default: 1 (Harmonic)</br></br>
  --prot_seq PROT_SEQ, -prot_seq PROT_SEQ</br>
                        User input sequence for building IDRs/helices etc.</br></br>
  --cutoff CUTOFF, -cutoff CUTOFF</br>
                        User defined Cut-off (in Angstrom) for contact-map generation. Default: 4.5A</br></br>
  --cutofftype CUTOFFTYPE, -cutofftype CUTOFFTYPE</br>
                        -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1</br></br>
  --W_cont, -W_cont     Weight (and normalize) CG contacts based on all atom contacts</br></br>
  --cmap CMAP, -cmap CMAP</br>
                        User defined cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)</br></br>
  --scaling SCALING, -scaling SCALING</br>
                        User defined scaling for mapping to all-atom contact-map.</br></br>
  --contfunc CONTFUNC, -contfunc CONTFUNC</br>
                        1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default: 2</br></br>
  --cutoff_p CUTOFF_P, -cutoff_p CUTOFF_P</br>
                        User defined Cut-off (in Angstrom) for Protein contact-map generation. Default: 4.5A</br></br>
  --cutofftype_p CUTOFFTYPE_P, -cutofftype_p CUTOFFTYPE_P</br>
                        For Proteins: -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1</br></br>
  --W_cont_p, -W_cont_p</br>
                        Weight (and normalize) Protein CG contacts based on all atom contacts</br></br>
  --cmap_p CMAP_P, -cmap_p CMAP_P</br>
                        User defined Protein cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)</br></br>
  --scaling_p SCALING_P, -scaling_p SCALING_P</br>
                        User defined scaling for mapping to all-atom contact-map.</br></br>
  --contfunc_p CONTFUNC_P, -contfunc_p CONTFUNC_P</br>
                        Proteins. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default=2</br></br>
  --cutoff_n CUTOFF_N, -cutoff_n CUTOFF_N</br>
                        User defined Cut-off (in Angstrom) for RNA/DNA contact-map generation. Default: 4.5A</br></br>
  --cutofftype_n CUTOFFTYPE_N, -cutofftype_n CUTOFFTYPE_N</br>
                        For RNA/DNA. -1 No map, 0 use -cmap file, 1 all-atom mapped to CG, 2: coarse-grain . Default: 1</br></br>
  --W_cont_n, -W_cont_n</br>
                        Weight (and normalize) RNA/DNA CG contacts based on all atom contacts</br></br>
  --cmap_n CMAP_N, -cmap_n CMAP_N</br>
                        User defined RNA/DNA cmap in format chain1 atom1 chain2 atom2 weight(opt) distance(opt)</br></br>
  --scaling_n SCALING_N, -scaling_n SCALING_N</br>
                        User RNA/DNA defined scaling for mapping to all-atom contact-map.</br></br>
  --contfunc_n CONTFUNC_N, -contfunc_n CONTFUNC_N</br>
                        RNA/DNA. 1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 5 Gauss no excl, 6 Gauss + excl, 7 Multi Gauss . Default: 2</br></br>
  --prot_cg PROT_CG, -prot_cg PROT_CG</br>
                        Level of Amino-acid coarse-graining 1 for CA-only, 2 for CA+CB. Dafault: 2 (CA+CB)</br></br>
  --CB_com, -CB_com     Put C-beta at side-chain COM (no hydrogens). Default: False</br></br>
  --CB_far, -CB_far     Place C-beta on farthest non-hydrogen atom. Default: False</br></br>
  --dsb, -dsb           Use desolvation barrier potential for contacts. Default: False</br></br>
  --native_ca NATIVE_CA, -native_ca NATIVE_CA</br>
                        Native file with only C-alphas. Just grep pdb.</br></br>
  --aa_pdb AA_PDB, -aa_pdb AA_PDB</br>
                        User input all-atom pdbfile/gro/mmCIF e.g. 1qys.pdb</br></br>
  --cg_pdb CG_PDB, -cg_pdb CG_PDB</br>
                        User input coarse grained pdbfile</br></br>
  --grotop GROTOP, -grotop GROTOP</br>
                        Gromacs topology file output name (tool adds prefix nucl_ and prot_ for independednt file). Default: gromacs.top</br></br>
  --pdbgro PDBGRO, -pdbgro PDBGRO</br>
                        Name for output .gro file.(tool adds prefix nucl_ and prot_ for independednt file). Default: gromacs.gro</br></br>
  --smogxml SMOGXML, -smogxml SMOGXML</br>
                        Name for output .xml (openSMOG) file.(tool adds prefix nucl_ and prot_ for independednt file). Default: opensmog.xml (and opensmog.top)</br></br>
  --opensmog, -opensmog</br>
                        Generate files ,xml and .top files for openSMOG. Default: False</br></br>
  --CB_gly, --CB_GLY, -CB_gly, -CB_GLY</br>
                        Add C-beta for glycine (pdb-file must have H-atoms). Default: Flase</br></br>
  --btparams, -btparams</br>
                        Use Betancourt-Thirumalai interaction matrix.</br></br>
  --mjparams, -mjparams</br>
                        Use Miyazawa-Jernighan interaction matrix.</br></br>
  --P_rad P_RAD, -P_rad P_RAD</br>
                        User defined radius for Backbone Phosphate bead. Default=3.7A</br></br>
  --S_rad S_RAD, -S_rad S_RAD</br>
                        User defined radius for Backbone Sugar bead. Default=3.7A</br></br>
  --Bpu_rad BPU_RAD, -Bpu_rad BPU_RAD</br>
                        User defined radius for N-Base Purine bead. Default=1.5A</br></br>
  --Bpy_rad BPY_RAD, -Bpy_rad BPY_RAD</br>
                        User defined radius for N-Base Pyrimidine bead. Default=1.5A</br></br>
  --nucl_cg NUCL_CG, -nucl_cg NUCL_CG</br>
                        Level of Amino-acid coarse-graining 1 for P-only, 3 for P-S-B, 5 for P-S-3B. Dafault: 3 (P-S-B)</br></br>
  --Kb_nucl KB_NUCL, -Kb_nucl KB_NUCL, --nKb KB_NUCL, -nKb KB_NUCL</br>
                        User defined force constant K_bond for RNA/DNA</br></br>
  --Ka_nucl KA_NUCL, -Ka_nucl KA_NUCL, --nKa KA_NUCL, -nKa KA_NUCL</br>
                        User defined force constant K_angle for RNA/DNA. Default=20</br></br>
  --Kd_sc_nucl KD_SC_NUCL, -Kd_sc_nucl KD_SC_NUCL, --nKd KD_SC_NUCL, -nKd KD_SC_NUCL</br>
                        User defined force constant K_dihedral for Bi-Si-Si+1-Bi+1. Default=0.5</br></br>
  --Kd_bb_nucl KD_BB_NUCL, -Kd_bb_nucl KD_BB_NUCL, --P_nKd KD_BB_NUCL, -P_nKd KD_BB_NUCL</br>
                        User defined force constant K_dihedral for Backbone Pi-Pi+1-Pi+2-Pi+3. Default=0.7</br></br>
  --P_stretch P_STRETCH, -P_stretch P_STRETCH</br>
                        Stretch the backbone dihedral to 180 degrees. Default = Use native backbone dihedral</br></br>
  --mulfac_nucl MULFAC_NUCL, -mulfac_nucl MULFAC_NUCL</br>
                        User defined Multiplicity scale factor of K_dihedral for Nucleic Acids</br></br>
  --Bpu_pos BPU_POS, -Bpu_pos BPU_POS</br>
                        Put input atom of Purine [N1,C2,H2-N2,N3,C4,C5,C6,O6-N6,N7,C8,N9,COM] as position of B.
                        Default=COM(Center_of_Mass)</br></br>
  --Bpy_pos BPY_POS, -Bpy_pos BPY_POS</br>
                        Put input atom of Pyrimidine [N1,C2,O2,N3,C4,O4-N4,C5,C6,COM] as position of B. Default=COM(Center_of_Mass)</br></br>
  --S_pos S_POS, -S_pos S_POS</br>
                        Put input atom of Sugar [C1',C2',C3',C4',C5',H2'-O2',O3',O4',O5',COM] as position of S.
                        Default=COM(Center_of_Mass)</br></br>
  --P_pos P_POS, -P_pos P_POS</br>
                        Put input atom of Phosphate [P,OP1,OP2,O5',COM] group as position of P. Default=COM(Center_of_Mass)</br></br>
  --pistacklen PISTACKLEN</br>
                        pi-pi stacking length. Default=3.6A</br></br>
  --debye, -debye       Use Debye-Huckel electrostatic term.</br></br>
  --debye_temp DEBYE_TEMP, -debye_temp DEBYE_TEMP</br>
                        Temperature for Debye length calculation. Default = 298K</br></br>
  --debye_length DEBYE_LENGTH, -debye_length DEBYE_LENGTH</br>
                        Debye length. in (A)</br></br>
  --CA_charge, -CA_charge</br>
                        Put charges on CA for K,L,H,D,E. Default: False</br></br>
  --CB_charge, -CB_charge</br>
                        Put charges on CB for K,L,H,D,E. Default: False</br></br>
  --P_charge, -P_charge</br>
                        Negative charge on Phosphate bead. Default: False</br></br>
  --iconc ICONC, -iconc ICONC</br>
                        Solvant ion conc.(N) for Debye length calcluation. Default=0.1M</br></br>
  --irad IRAD, -irad IRAD</br>
                        Solvant ion rad for Debye length calcluation. Default=1.4A</br></br>
  --dielec DIELEC, -dielec DIELEC</br>
                        Dielectric constant of solvant. Default=70</br></br>
  --hpstrength HPSTRENGTH, -hpstrength HPSTRENGTH</br>
                        Strength with which hydrophobic contacts interact.</br></br>
  --ext_conmap EXT_CONMAP, -ext_conmap EXT_CONMAP</br>
                        External contact map in format chain res chain res</br></br>
  --interaction, -interaction</br>
                        User defined interactions in file interactions.dat.</br></br>
  --dswap, -dswap       For domain swapping runs. Symmetrised SBM is generated.</br></br>
  --hphobic, -hphobic   Generate hydrophobic contacts.</br></br>
  --hpdist HPDIST, -hpdist HPDIST</br>
                        Equilibrium distance for hydrophobic contacts.</br></br>
  --interface, -interface</br>
                        Takes input for Nucleiotide_Protein interface from file nucpro_interface.input.</br></br>
  --custom_nuc CUSTOM_NUC, -custom_nuc CUSTOM_NUC</br>
                        Use custom non native DNA/RNA structure Eg.: polyT.pdb. Default: Use from native structure</br></br>
  --control             Use the native system as control. Use DNA/RNA bound to native protein site. --custom_nuc will be disabled.
                        Default: False (Move DNA/RNA away from native binding site)</br></br>
  --excl_rule EXCL_RULE</br>
                        Use 1: Geometric mean. 2: Arithmatic mean</br></br>
  --Kr_prot KR_PROT     Krepulsion. Default=1.0</br></br>
  --Kr_nucl KR_NUCL     Krepulsion. Default=1.0</br></br>
