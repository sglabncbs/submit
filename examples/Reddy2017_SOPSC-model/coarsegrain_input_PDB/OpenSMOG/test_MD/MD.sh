cp ../prot_opensmog.* .
editconf -f prot_opensmog.gro -o prot_opensmog.gro -c -box 50 50 50
#conda activate SMOG #activate python env with OpenSMOG & OpenMM installed
python ../../../openSMOG_run.py prot_opensmog.*
rm \#*
