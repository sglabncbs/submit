#FOR GROMACS 4.5.4 SBM v1 (Gaussian)
cp ../SuBMIT_Output/gromacs.* .
editconf -f gromacs.gro -o gromacs.gro -c -box 50 50 50
grompp_gauss -f ../../../run_notable.mdp  -c gromacs.gro -p gromacs.top -o Output -po Output
mdrun_gauss -deffnm Output -nt 1 -v -pd 
rm \#*
