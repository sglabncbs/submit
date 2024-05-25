cp ../prot_opensmog.* .
editconf -f prot_opensmog.gro -o prot_opensmog.gro -c -box 250 250 250
#conda activate SMOG #activate python env with OpenSMOG & OpenMM installed
python ../../../openSMOG_run.py prot_opensmog.*
rm \#*
