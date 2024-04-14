# Package to generate coarse-Grain structure and topology for using Enhanced Structure Based Models MD simulations on GROMACS and OpenSMOG (OpenMM based)##
# digvijaylp @ shachilab16.ncbs.res #
$ python esbm.py --help

usage: esbm.py [-h] [--CA_rad CA_RAD] [--CA_com] [--CB_rad CB_RAD]
               [--CB_radii] [--Kb_prot KB_PROT] [--Ka_prot KA_PROT]
               [--Kd_bb_prot KD_BB_PROT] [--Kd_sc_prot KD_SC_PROT]
               [--mulfac_prot MULFAC_PROT] [--CB_chiral] [--bfunc BFUNC]
               [--cutoff CUTOFF] [--cutofftype CUTOFFTYPE] [--W_cont]
               [--cmap CMAP] [--scaling SCALING] [--contfunc CONTFUNC]
               [--prot_cg PROT_CG] [--CB_com] [--CB_far] [--dsb]
               [--native_ca NATIVE_CA] [--aa_pdb AA_PDB] [--cg_pdb CG_PDB]
               [--grotop GROTOP] [--pdbgro PDBGRO] [--smogxml SMOGXML]
               [--opensmog] [--pl_map] [--CB_gly] [--btmap] [--mjmap]
               [--P_rad P_RAD] [--S_rad S_RAD] [--Bpu_rad BPU_RAD]
               [--Bpy_rad BPY_RAD] [--nucl_cg NUCL_CG] [--Kb_nucl KB_NUCL]
               [--Ka_nucl KA_NUCL] [--Kd_sc_nucl KD_SC_NUCL]
               [--Kd_bb_nucl KD_BB_NUCL] [--P_stretch P_STRETCH]
               [--mulfac_nucl MULFAC_NUCL] [--Bpu_pos BPU_POS]
               [--Bpy_pos BPY_POS] [--S_pos S_POS] [--P_pos P_POS]
               [--pistacklen PISTACKLEN] [--debye] [--T T] [--CB_charge]
               [--P_charge] [--iconc ICONC] [--irad IRAD]
               [--dielec DIELEC] [--hpstrength HPSTRENGTH]
               [--ext_conmap EXT_CONMAP] [--interaction] [--dswap]
               [--hphobic] [--hpdist HPDIST] [--interface]
               [--custom_nuc CUSTOM_NUC] [--control]
               [--excl_rule EXCL_RULE] [--Kr KR]

Generate GROMACS and OPTIM potential files for Protein + Nucleic Acids
enhanced SBM models.

optional arguments:
  -h, --help            show this help message and exit
  --CA_rad CA_RAD, -CA_rad CA_RAD
                        User defined radius for C-alpha (same for all
                        beads) in Angstrom. Default: 4.0A
  --CA_com, -CA_com     Place C-alpha at COM of backbone. Default: False
  --CB_rad CB_RAD, -CB_rad CB_RAD
                        User defined radius for C-beta (same for all
                        beads) in Angstrom for prot_cg 2. Default:
                        Statistically Derived for each AA-residue
  --CB_radii, -CB_radii
                        User defined C-beta radii from radii.dat
                        (AA-3-letter-code radius-in-Angsrtom)
  --Kb_prot KB_PROT, -Kb_prot KB_PROT, --Kb KB_PROT, -Kb KB_PROT
                        User defined force constant K_bond for Proteins
  --Ka_prot KA_PROT, -Ka_prot KA_PROT, --Ka KA_PROT, -Ka KA_PROT
                        User defined force constant K_angle for Proteins
  --Kd_bb_prot KD_BB_PROT, -Kd_bb_prot KD_BB_PROT, --Kd KD_BB_PROT, -Kd KD_BB_PROT
                        User defined force constant K_dihedral for
                        Proteins
  --Kd_sc_prot KD_SC_PROT, -Kd_sc_prot KD_SC_PROT, --Kd_chiral KD_SC_PROT, -Kd_chiral KD_SC_PROT
                        User defined force constant K_dihedral for
                        Proteins
  --mulfac_prot MULFAC_PROT, -mulfac_prot MULFAC_PROT
                        User defined Multiplicity scale factor of
                        K_dihedral/mulfac_prot for Proteins
  --CB_chiral, -CB_chiral
                        Improper dihedral for CB sidechain chirality.
                        Default: False
  --bfunc BFUNC, -bfunc BFUNC
                        Bond function 1: harnomic, 7: FENE. Default: 1
                        (Harmonic)
  --cutoff CUTOFF, -cutoff CUTOFF
                        User defined Cut-off (in Angstrom) for contact-map
                        generation. Default: 4.5A
  --cutofftype CUTOFFTYPE, -cutofftype CUTOFFTYPE
                        -1 No map, 0 use -cmap file, 1 all-atom mapped to
                        CG, 2: coarse-grain . Default: 1
  --W_cont, -W_cont     Weight (and normalize) CG contacts based on all
                        atom contacts
  --cmap CMAP, -cmap CMAP
                        User defined cmap in format chain1 atom1 chain2
                        atom2 weight(opt) distance(opt)
  --scaling SCALING, -scaling SCALING
                        User defined scaling for mapping to all-atom
                        contact-map.
  --contfunc CONTFUNC, -contfunc CONTFUNC
                        1: LJ C6-C12, 2 LJ C10-C12, 3 LJ C12-C18, 4
                        Gauss-C12 . Default: 2
  --prot_cg PROT_CG, -prot_cg PROT_CG
                        Level of Amino-acid coarse-graining 1 for CA-only,
                        2 for CA+CB. Dafault: 2 (CA+CB)
  --CB_com, -CB_com     Put C-beta at side-chain COM (no hydrogens).
                        Default: False
  --CB_far, -CB_far     Place C-beta on farthest non-hydrogen atom.
                        Default: False
  --dsb, -dsb           Use desolvation barrier potential for contacts.
                        Default: False
  --native_ca NATIVE_CA, -native_ca NATIVE_CA
                        Native file with only C-alphas. Just grep pdb.
  --aa_pdb AA_PDB, -aa_pdb AA_PDB
                        User input all-atom pdbfile/gro/mmCIF e.g.
                        1qys.pdb
  --cg_pdb CG_PDB, -cg_pdb CG_PDB
                        User input coarse grained pdbfile
  --grotop GROTOP, -grotop GROTOP
                        Gromacs topology file output name (tool adds
                        prefix nucl_ and prot_ for independednt file).
                        Default: gromacs.top
  --pdbgro PDBGRO, -pdbgro PDBGRO
                        Name for output .gro file.(tool adds prefix nucl_
                        and prot_ for independednt file). Default:
                        gromacs.gro
  --smogxml SMOGXML, -smogxml SMOGXML
                        Name for output .xml (openSMOG) file.(tool adds
                        prefix nucl_ and prot_ for independednt file).
                        Default: opensmog.xml (and opensmog.top)
  --opensmog, -opensmog
                        Generate files ,xml and .top files for openSMOG.
                        Default: False
  --pl_map, -pl_map     Plot contact map for two bead model. Default:
                        False
  --CB_gly, --CB_GLY, -CB_gly, -CB_GLY
                        Add C-beta for glycine (pdb-file must have
                        H-atoms). Default: Flase
  --btmap, -btmap       Use Betancourt-Thirumalai interaction matrix.
  --mjmap, -mjmap       Use Miyazawa-Jernighan interaction matrix.
  --P_rad P_RAD, -P_rad P_RAD
                        User defined radius for Backbone Phosphate bead.
                        Default=3.7A
  --S_rad S_RAD, -S_rad S_RAD
                        User defined radius for Backbone Sugar bead.
                        Default=3.7A
  --Bpu_rad BPU_RAD, -Bpu_rad BPU_RAD
                        User defined radius for N-Base Purine bead.
                        Default=1.5A
  --Bpy_rad BPY_RAD, -Bpy_rad BPY_RAD
                        User defined radius for N-Base Pyrimidine bead.
                        Default=1.5A
  --nucl_cg NUCL_CG, -nucl_cg NUCL_CG
                        Level of Amino-acid coarse-graining 1 for P-only,
                        3 for P-S-B, 5 for P-S-3B. Dafault: 3 (P-S-B)
  --Kb_nucl KB_NUCL, -Kb_nucl KB_NUCL, --nKb KB_NUCL, -nKb KB_NUCL
                        User defined force constant K_bond for RNA/DNA
  --Ka_nucl KA_NUCL, -Ka_nucl KA_NUCL, --nKa KA_NUCL, -nKa KA_NUCL
                        User defined force constant K_angle for RNA/DNA.
                        Default=20
  --Kd_sc_nucl KD_SC_NUCL, -Kd_sc_nucl KD_SC_NUCL, --nKd KD_SC_NUCL, -nKd KD_SC_NUCL
                        User defined force constant K_dihedral for Bi-Si-
                        Si+1-Bi+1. Default=0.5
  --Kd_bb_nucl KD_BB_NUCL, -Kd_bb_nucl KD_BB_NUCL, --P_nKd KD_BB_NUCL, -P_nKd KD_BB_NUCL
                        User defined force constant K_dihedral for
                        Backbone Pi-Pi+1-Pi+2-Pi+3. Default=0.7
  --P_stretch P_STRETCH, -P_stretch P_STRETCH
                        Stretch the backbone dihedral to 180 degrees.
                        Default = Use native backbone dihedral
  --mulfac_nucl MULFAC_NUCL, -mulfac_nucl MULFAC_NUCL
                        User defined Multiplicity scale factor of
                        K_dihedral for Nucleic Acids
  --Bpu_pos BPU_POS, -Bpu_pos BPU_POS
                        Put input atom of Purine
                        [N1,C2,H2-N2,N3,C4,C5,C6,O6-N6,N7,C8,N9,COM] as
                        position of B. Default=COM(Center_of_Mass)
  --Bpy_pos BPY_POS, -Bpy_pos BPY_POS
                        Put input atom of Pyrimidine
                        [N1,C2,O2,N3,C4,O4-N4,C5,C6,COM] as position of B.
                        Default=COM(Center_of_Mass)
  --S_pos S_POS, -S_pos S_POS
                        Put input atom of Sugar
                        [C1',C2',C3',C4',C5',H2'-O2',O3',O4',O5',COM] as
                        position of S. Default=COM(Center_of_Mass)
  --P_pos P_POS, -P_pos P_POS
                        Put input atom of Phosphate [P,OP1,OP2,O5',COM]
                        group as position of P.
                        Default=COM(Center_of_Mass)
  --pistacklen PISTACKLEN
                        pi-pi stacking length. Default=3.6A
  --debye               Use Debye-Huckel electrostatic term.
  --T T                 Temperature for Debye length calculation. Default
                        = 298K
  --CB_charge, -CB_charge
                        Put charges on CB for K,L,H,D,E. Default: False
  --P_charge, -P_charge
                        Negative charge on Phosphate bead. Default: False
  --iconc ICONC         Solvant ion conc.(N) for Debye length calcluation.
                        Default=0.1M
  --irad IRAD           Solvant ion rad for Debye length calcluation.
                        Default=1.4A
  --dielec DIELEC       Dielectric constant of solvant. Default=70
  --hpstrength HPSTRENGTH, -hpstrength HPSTRENGTH
                        Strength with which hydrophobic contacts interact.
  --ext_conmap EXT_CONMAP, -ext_conmap EXT_CONMAP
                        External contact map in format chain res chain res
  --interaction, -interaction
                        User defined interactions in file interaction.dat.
  --dswap, -dswap       For domain swapping runs. Symmetrised SBM is
                        generated.
  --hphobic, -hphobic   Generate hydrophobic contacts.
  --hpdist HPDIST, -hpdist HPDIST
                        Equilibrium distance for hydrophobic contacts.
  --interface, -interface
                        Takes input for Nucleiotide_Protein interface from
                        file nucpro_interface.input.
  --custom_nuc CUSTOM_NUC, -custom_nuc CUSTOM_NUC
                        Use custom non native DNA/RNA structure Eg.:
                        polyT.pdb. Default: Use from native structure
  --control             Use the native system as control. Use DNA/RNA
                        bound to native protein site. --custom_nuc will be
                        disabled. Default: False (Move DNA/RNA away from
                        native binding site)
  --excl_rule EXCL_RULE
                        Use 1: Geometric mean. 2: Arithmatic mean
  --Kr KR               Krepulsion. Default=5.7A
