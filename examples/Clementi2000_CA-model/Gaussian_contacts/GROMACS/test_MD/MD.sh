#FOR GROMACS 4.5.4 SBM_1.0 
#Information and download link: https://smog-server.org/extension/#gauss
cp ../prot_gromacs.* .
editconf_gauss -f prot_gromacs.gro -o prot_gromacs.gro -c -box 50 50 50
grompp_gauss -f ../../../run_notable.mdp  -c prot_gromacs.gro -p prot_gromacs.top -o Output -po Output
mdrun_gauss -deffnm Output -nt 1 -v -pd 
rm \#*
