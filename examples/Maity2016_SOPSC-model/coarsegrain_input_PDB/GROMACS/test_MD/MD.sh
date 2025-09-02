#FOR GROMACS 4.5.4
cp ../SuBMIT_Output/gromacs.* .
editconf -f gromacs.gro -o gromacs.gro -c -box 50 50 50
grompp -f ../../../run_coulvdwtable.mdp  -c gromacs.gro -p gromacs.top -o Output -po Output
mdrun -deffnm Output -nt 1 -v -pd -table ../SuBMIT_Output/Tables/table_coul_lj0612.xvg -tablep ../SuBMIT_Output/Tables/table_coul_lj0612.xvg -tableb ../SuBMIT_Output/Tables/table.xvg
rm \#*
