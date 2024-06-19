cp ../SuBMIT_Output/opensmog.* .
editconf -f opensmog.gro -o opensmog.gro -c -box 50 50 50
#conda activate SMOG #activate python env with OpenSMOG & OpenMM installed
python ../../../openSMOG_run.py opensmog.*
rm \#*
