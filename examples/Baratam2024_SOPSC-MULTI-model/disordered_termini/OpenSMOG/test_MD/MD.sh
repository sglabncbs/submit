cp ../SuBMIT_Output/Tables/opensmog.* .
editconf -f opensmog.gro -o opensmog.gro -c -box 250 250 250
#conda activate SMOG #activate python env with OpenSMOG & OpenMM installed
python ../../../openSMOG_run.py opensmog.*
rm \#*
