import numpy as np
from tqdm import tqdm
codons = dict()
for x in [i+j+k for i in "5augct" for j in "AUGCT" for k in "augct3"]:
    if x[1] not in codons: codons[x[1]]=[]
    if x not in codons[x[1]]: codons[x[1]].append(x)

with open("../pal2019/adj_nbnb.stackparams.dat") as fin:
    pair = []
    fout=open("adj_nbnb.stackparams.dat","w+")
    for line in fin:
        if line.startswith("#"): fout.write(line)
        else:
            fout.write("#%s"%line)
            line=line.split()
            a1,a2=line[:2]
            data=line[2:]
            for x in codons[a1[2]]:
                for y in codons[a2[2]]:
                    if x[0]=="5" and y[0]=="5": continue
                    if x[2]=="3" and y[2]=="3": continue
                    p=[x,y];p.sort();p=tuple(p)
                    if p not in pair:
                        pair.append(p)
                        fout.write("%s%s %s%s"%(a1[:2],p[0],a2[:2],p[1]))
                        fout.write(len(data)*" %s"%tuple(data))
                        fout.write("\n")

    fout.close()

with open("../pal2019/inter_nbcb.stackparams.dat") as fin:
    fout=open("inter_nbcb.stackparams.dat","w+")
    for line in fin:
        if line.startswith("#"): fout.write(line)
        else:
            fout.write("#%s"%line)
            line=line.split()
            a1,a2=line[:2]
            data=line[2:]
            assert a2.startswith("CB")
            y=a2[2:]
            for x in codons[a1[2]]:
                fout.write("%s%s %s%s"%(a1[:2],x,a2[:2],y))
                fout.write(len(data)*" %s"%tuple(data))
                fout.write("\n")
    fout.close()

fout=open("codonduplex.bpairparams.dat","w+")
fout.write("#a1 a2 eps sig(A)\n")
comp={"A":["U","T"],"G":["C"],"C":["G"],"U":["A"],"T":["A"],"5":[],"3":[]}
pairs = []
for x in codons:
    for y in codons:
        if x in comp[y]:
            assert y in comp[x]
            for c1 in codons[x]:
                for c2 in codons[y]:
                    eps=-1+np.sum(np.int_([c1[i].upper() in comp[c2[-1+3-i].upper()] for i in range(3)]))
                    if eps>0: 
                        p=[c1,c2]; p.sort();p=tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            fout.write("#%s%s %s%s %.2f %5.2f\n"%("S0",p[0],"S0",p[1],eps,14.0))
                            fout.write("%s%s %s%s %.2f %5.2f\n"%("S0",p[0],"B0",p[1],eps,10.0))
                            fout.write("%s%s %s%s %.2f %5.2f\n"%("B0",p[0],"S0",p[1],eps,10.0))
                            fout.write("%s%s %s%s %.2f %5.2f\n"%("B0",p[0],"B0",p[1],eps,06.0))
fout.close()                        
                    