#FOR GROMACS 4.5.4
cp ../prot_gromacs.* .
editconf -f prot_gromacs.gro -o prot_gromacs.gro -c -box 50 50 50
grompp -f ../../../../run_coulvdwtable.mdp  -c prot_gromacs.gro -p prot_gromacs.top -o Output -po Output
mdrun -deffnm Output -nt 1 -v -pd -table ../table_lj1012.xvg -tablep ../table_lj1012.xvg
