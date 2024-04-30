import numpy as np
from tqdm import trange,tqdm
from PDB_IO import *
from hybrid_36 import hy36encode,hy36decode

class Tables:
    def __init__(self) -> None:
        pass

    def __pad__(self,X):
        step = 0.001 #nm
        return np.int_(range(0,int(X[0]*1000)))*0.001

    def __write_bond_table__(self,index,X,V,V_1):
        with open("table_b"+str(index)+".xvg","w+") as fout:
            for x in self.__pad__(X):           
                fout.write("%e %e %e\n"%(x,V[0],0))
            for i in range(X.shape[0]):
                fout.write("%e %e %e\n"%(X[i],V[i],-V_1[i]))
        return
    
    def __electrostatics__(self,elec,r):
        jtocal = 1/elec.caltoj		#0.239  #Cal/J #SMOG tut mentiones 1/5 = 0.2
        D = elec.dielec
        if elec.debye:
            if elec.inv_dl == 0:
                """debye length
                                            e0*D*KB*T
                            dl**2 = -------------------------
                                    2*el*el*NA*l_to_nm3*iconc
                
                ref: https://www.weizmann.ac.il/CSB/Levy/sites/Structural_Biology.Levy/files/publications/Givaty_JMB_2009.pdf
                """
                pi = np.pi
                inv_4pieps =  138.935485    #KJ mol-1 e-2
                iconc = elec.iconc 		    #M L-1
                irad = 0.1*elec.irad        #nm 
                T = elec.debye_temp			#K				#Temperature
                e0 = 8.854e-21 			#C2 J-1 nm-1	#permitivity: The value have been converted into C2 J-1 nm-1
                NA = 6.022E+23          #n/mol  		#Avogadro's number
                KB = 1.3807E-23 	    #J/K			#Boltzmann constant
                #NA = elec.permol          
                #KB = elec.Kboltz*1000.0/NA
                el = 1.6e-19			#C/e				#charge on electron
                l_to_nm3 = 1e-24		#L/nm3			#converting l to nm3

                # [C2 J-1 nm-1] * [J/K] * [K] = [C2 nm-1]
                dl_N = e0*D*KB*T	#numerator
                #  [C2] * [n/mol] * [L/nm3] =  [C2 n L mol-1 nm-3]
                dl_D = 2*el*el*NA*l_to_nm3		#denom
                # [C2 nm-1]/[C2 n L mol-1 nm-3] = [nm2 (mol/n) L-1]
                dl = (dl_N/dl_D)**0.5 #debye length [nm M^0.5 L^-0.5]
                inv_dl = (iconc**0.5)/dl 		#nm-1
            else:
                inv_dl = 10*elec.inv_dl #A-1 to -nm-1
            Bk = np.exp(inv_dl*irad)/(1+inv_dl*irad)
            #K_debye = (jtocal*inv_4pieps/D)*Bk
        else:
            inv_dl = 0
            Bk = 1
        K_elec = jtocal*Bk/D #inv_4pieps*q1q2 is multiplied by gromacs
        V = K_elec*np.exp(-inv_dl*r)/r
        #(1/K_elecd) * V/dr = (1/r)*[d/dr (exp(-inv_dl*r)) ]
        #                       + exp(-inv_dl*r)*[d/dr (1/r)]
        #                   = (1/r)*exp(-inv_dl*r)*(-inv_dl)
        #                       + exp(-inv_dl*r)*(-1/r**2)
        #                   = -(inv_dl/r)*exp(-inv_dl*r) - exp(-inv_dl*r)/r**2
        #                   = -[(inv_dl/r)+(1/r**2)]*exp(-inv_dl*r)
        #                   = -[inv_dl.r + 1]*exp(-inv_dl*r)/r**2
        # for not debye, if Bk = 1 and inv_dl = 0
        # V_1 = K_elec*(-1)*(1/r**2) = -K_elec/r**2
        V_1 = K_elec*(-inv_dl*r-1)*np.exp(-inv_dl*r)/r**2
        return V,V_1
        
    def __write_pair_table__(self,ljtype,elec):
        #writing pairs table file

        r = np.int_(range(0,50000))*0.002 #100 nm
        cutoff = np.int_(r>=0.01)
        r[0]=10E-9  #buffer to avoid division by zero

        if ljtype == 1: 
            assert elec.debye, "Table file only needed if using Debye-Huckel Electrostatics."
            print (">> Writing LJ 6-12 table file",'table_lj0612.xvg')
            suffix = "0612.xvg"
            A 	= -1.0/r**6*cutoff  # r6
            A_1	= -6.0/r**7*cutoff  # r6_1
        elif ljtype == 2:
            print (">> Writing LJ 10-12 table file",'table_lj1012.xvg')
            suffix = "1012.xvg"
            A 	=  -1.0/r**10*cutoff # r10
            A_1	= -10.0/r**11*cutoff # r10_1

        B     =   1/r**12*cutoff
        B_1	=  12/r**13*cutoff

        if elec.CA or elec.CB or elec.P:
            V,V_1 = self.__electrostatics__(elec=elec,r=r)
            V = V*cutoff
            V_1 = -1*V_1*cutoff
            r = np.round(r,3)
            with open("table_coul_lj"+suffix,"w+") as fout1:
                for i in range(r.shape[0]):
                    fout1.write('%e %e %e %e %e %e %e\n' %(r[i],V[i],V_1[i],A[i],A_1[i],B[i],B_1[i]))
        r = np.round(r,3)
        with open("table_lj"+suffix,"w+") as fout:
            for i in range(r.shape[0]):
                fout.write('%e 0 0 %e %e %e %e\n' %(r[i],A[i],A_1[i],B[i],B_1[i]))

        return
    
class Calculate:
    def __init__(self,pdb) -> None:
        self.pdb = pdb
        self.CA_atn = dict()
        self.CB_atn = dict()
        self.P_atn = dict()
        self.S_atn = dict()
        self.B_atn = dict()
        self.bonds,self.angles,self.bb_dihedrals,self.sc_dihedrals = [],[],[],[]
        self.contacts = []

    def __distances__(self,pairs):
        #takes pairs, retuns array of distances in nm
        i,j = np.transpose(pairs)
        xyz = self.pdb.xyz
        return 0.1*np.sum((xyz[j]-xyz[i])**2,1)**0.5

    def __angles__(self,triplets):
        #takes list triplets, retuns array of angles (0-180) in deg
        i,j,k = np.transpose(triplets)
        xyz = self.pdb.xyz
        n1 = xyz[i]-xyz[j]; n2 = xyz[k]-xyz[j]
        n1n2 = (np.sum((n1**2),1)**0.5)*(np.sum((n2**2),1)**0.5)
        return np.arccos(np.sum(n1*n2,1)/n1n2)*180/np.pi

    def __torsions__(self,quadruplets):
        #takes list of quadruplets, retuns array of torsion angles (0-360) in deg
        i,j,k,l = np.transpose(quadruplets)
        xyz = self.pdb.xyz
        BA,BC = xyz[i]-xyz[j],xyz[k]-xyz[j]
        CD=xyz[l]-xyz[k];CB=BC
        n1 = np.cross(BA,BC)
        n2 = np.cross(CD,BC)
        direction = np.cross(n1,n2)
        cosine_d = np.sum(direction*BC,1)/((np.sum(direction**2,1)**0.5)*(np.sum(BC**2,1)**0.5))
        sign=np.float_(np.round(cosine_d))
        n1n2 = (np.sum((n1**2),1)**0.5)*(np.sum((n2**2),1)**0.5)
        angle_normalizer = np.int_(sign<0)*2*180
        phi = sign*np.arccos(np.sum(n1*n2,1)/n1n2)*180/np.pi
        phi += angle_normalizer
        return phi

    def processData(self,data):
        #converts list type data in to dict of [chain][resnum] = atom number
        if len(data.lines)>0:
            for x in range(len(data.res)):
                chain,rnum,rname,aname = data.res[x]
                if aname == "CA": 
                    if chain not in self.CA_atn: self.CA_atn[chain] = dict()
                    self.CA_atn[chain][rnum] = data.atn[x]
                elif aname == "CB":
                    if chain not in self.CB_atn: self.CB_atn[chain] = dict()
                    self.CB_atn[chain][rnum] = data.atn[x]
                elif aname.startswith("P"):
                    if chain not in self.P_atn: self.P_atn[chain] = dict()
                    self.P_atn[chain][rnum]= data.atn[x]
                elif aname.startswith(("C","R")):
                    if chain not in self.S_atn: self.S_atn[chain] = dict()
                    self.S_atn[chain][rnum] = data.atn[x]
                    #if rname not in self.S_atn[chain]: self.S_atn[chain][rnum] = []
                    #self.S_atn[chain][rnum].append(data.atn[x])
                elif aname.startswith(("N","B")): 
                    if chain not in self.B_atn: self.B_atn[chain] = dict()
                    self.B_atn[chain][rnum] = data.atn[x]
                    #if rname not in self.B_atn[chain]: self.B_atn[chain][rnum] = []
                    #self.B_atn[chain][rnum].append(data.atn[x])
        return

    def Bonds(self):
        # Getting Bond length info from the pre-supplied data

        if len(self.CA_atn) != 0: #protein exists
            for c in self.CA_atn:
                resnum = list(self.CA_atn[c].keys())
                resnum.sort()
                pairs = [(self.CA_atn[c][x],self.CA_atn[c][x+1]) for x in resnum if x+1 in self.CA_atn[c]]
                # add CB entries if side-chain data exits
                if len(self.CB_atn)!=0: pairs += [(self.CA_atn[c][x],self.CB_atn[c][x])   for x in resnum if  x  in self.CB_atn[c]]
                #determine bond details in chain order
                pairs = np.int_(pairs)
                D = self.__distances__(pairs)
                self.bonds.append((pairs,D))

        if len(self.P_atn): #RNA/DNA exists
            for c in self.P_atn:
                if len(self.S_atn) == 0: 
                    assert len(self.B_atn) == 0
                    resnum = list(self.P_atn[c].keys())
                    resnum.sort()
                    pairs = [(self.P_atn[c][x],self.P_atn[c][x+1]) for x in resnum if x+1 in self.P_atn[c]]
                else:
                    assert len(self.B_atn) > 0
                    resnum = list(self.S_atn[c].keys())
                    pairs = []
                    for x in resnum:
                        if x in self.P_atn[c]: pairs.append(((self.P_atn[c][x],self.S_atn[c][x])))
                        pairs.append((self.S_atn[c][x],self.B_atn[c][x]))
                        if x+1 in self.P_atn[c]: pairs.append((self.S_atn[c][x],self.P_atn[c][x+1]))
                pairs = np.int_(pairs)
                D = self.__distances__(pairs)
                self.bonds.append((pairs,D))
        
        return

    def Angles(self):
        # Getting Bond angle info from the pre-supplied data

        if len(self.CA_atn) != 0:
            for c in self.CA_atn:
                resnum = list(self.CA_atn[c].keys())
                resnum.sort()
                triplets = [(self.CA_atn[c][x],self.CA_atn[c][x+1],self.CA_atn[c][x+2]) for x in resnum if x+2 in self.CA_atn[c]]
                if len(self.CB_atn)!=0:
                    triplets += [(self.CA_atn[c][x],self.CA_atn[c][x+1],self.CB_atn[c][x+1]) for x in resnum if x+1 in self.CB_atn[c]]
                    triplets += [(self.CB_atn[c][x],self.CA_atn[c][x],self.CA_atn[c][x+1])   for x in resnum if x in self.CB_atn[c] and x+1 in self.CA_atn[c]]
                triplets = np.int_(triplets)
                A = self.__angles__(triplets=triplets)
                self.angles.append((triplets,A))

        if len(self.P_atn) != 0: #RNA/DNA exists
            for c in self.P_atn:
                if len(self.S_atn) == 0:    #Pi--Pi+1--Pi+2 angle
                    assert len(self.B_atn) == 0
                    resnum = list(self.P_atn[c].keys())
                    resnum.sort()
                    triplets = [(self.P_atn[c][x],self.P_atn[c][x+1],self.P_atn[c][x+2]) for x in resnum if x+2 in self.P_atn[c]]
                else:
                    assert len(self.B_atn) > 0
                    resnum = list(self.S_atn[c].keys())
                    resnum.sort()
                    triplets = []
                    for x in resnum:
                        if x in self.P_atn[c]:
                            triplets.append((self.P_atn[c][x],self.S_atn[c][x],self.B_atn[c][x]))
                            if x-1 in self.S_atn[c]:
                                triplets.append((self.S_atn[c][x-1],self.P_atn[c][x],self.S_atn[c][x]))
                            if x+1 in self.P_atn[c]:
                                triplets.append((self.P_atn[c][x],self.S_atn[c][x],self.P_atn[c][x+1]))
                                triplets.append((self.B_atn[c][x],self.S_atn[c][x],self.P_atn[c][x+1]))
                triplets = np.int_(triplets)
                A = self.__angles__(triplets=triplets)
                self.angles.append((triplets,A))

        return

    def Dihedrals(self):
        # Getting torsion angle info from the pre-supplied data
        if len(self.CA_atn) != 0:
            for c in self.CA_atn:
                resnum = list(self.CA_atn[c].keys())
                resnum.sort()
                quadruplets = [tuple([self.CA_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.CA_atn[c]]
                quadruplets = np.int_(quadruplets)
                T = self.__torsions__(quadruplets=quadruplets)
                self.bb_dihedrals.append((quadruplets,T))
                if len(self.CB_atn)!=0:
                    quadruplets = [(self.CA_atn[c][x-1],self.CA_atn[c][x+1],self.CA_atn[c][x],self.CB_atn[c][x]) for x in resnum if x+1 in self.CA_atn[c] and x-1 in self.CA_atn[c] and x in self.CB_atn[c]]
                    quadruplets = np.int_(quadruplets)
                    T = self.__torsions__(quadruplets=quadruplets)
                    self.sc_dihedrals.append((quadruplets,T))

        if len(self.P_atn) != 0: #RNA/DNA exists
            for c in self.P_atn:
                resnum = list(self.P_atn[c].keys())
                resnum.sort()
                quadruplets = [tuple([self.P_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.P_atn[c]]
                quadruplets = np.int_(quadruplets)
                T = self.__torsions__(quadruplets=quadruplets)
                self.bb_dihedrals.append((quadruplets,T))
            if len(self.S_atn) != 0:
                assert len(self.B_atn) > 0
                for c in self.S_atn:
                    resnum = list(self.S_atn[c].keys())
                    resnum.sort()
                    quadruplets = [(self.B_atn[c][x],self.S_atn[c][x], \
                                    self.S_atn[c][x+1],self.B_atn[c][x+1]) \
                                    for x in resnum if x+1 in self.S_atn[c]]
                    quadruplets = np.int_(quadruplets)
                    T = self.__torsions__(quadruplets=quadruplets)
                    self.sc_dihedrals.append((quadruplets,T))
        
        return
    
    def Pairs(self,cmap,aa_data):
        # Getting Non-bonded contact pairs info from the pre-supplied data
        temp_p,temp_w,temp_d = [],[],[]
        pairs,weights,distances = [],[],[]
        
        if cmap.type == -1: return  # Generating top without pairs 

        elif cmap.type == 0:        # Use pairs from user input in format cid_i, atnum_i, cid_j, atnum_j, weight_ij (opt), dist_ij (opt)
            assert cmap.file != ""
            print ("> Using cmap file (c1 a1 c2 a2 w d)",cmap.file)
            with open(cmap.file) as fin:
                for line in fin:
                    line = line.split()
                    c1,a1,c2,a2 = np.int_(line[:4])-1
                    if len(line) < 6:
                        w,d = 1.0,0.0
                        if len(line)==5: w = np.float(line[4])
                        temp_p.append((a1,a2));temp_w.append(w)
                    elif len(line)==6: 
                        w,d = np.float_(line[4:])
                        pairs.append((a1,a2));weights.append(w);distances.append(d)
            if len(temp_p)!=0: temp_d = list(self.__distances__(pairs=np.int_(temp_p)))
            pairs += temp_p; weights += temp_w; distances += temp_d
            pairs = np.int_(pairs); weights = np.float_(weights); distances = np.float_(distances)

        elif cmap.type == 1:        # Calculating contacts from all-atom structure and maping to CG structure
            group = []
            if len(self.CA_atn) != 0:
                if len(self.CB_atn) == 0:   #0 for CA
                    for r in aa_data.res: group.append(tuple(list(r[:2])+[0]))
                else:                       #1 for CB
                    for r in aa_data.res: group.append(tuple(list(r[:2])+[int(r[-1] not in ("N","C","CA","O"))]))
            if len(self.P_atn) != 0:
                if len(self.B_atn) == 0:    #2 or P
                    for r in aa_data.res: group.append(tuple(list(r[:2])+[2]))
                else:                       # 3 for S #5 for B
                    for r in aa_data.res: 
                        group.append(tuple(list(r[:2])+[3+2*int("P" in r[-1])-1*int("'" in r[-1])]))

            faa = open(aa_data.pdbfile+".AAcont","w+")
            fcg = open(aa_data.pdbfile+".CGcont","w+")
            cid,rnum,bb_sc = np.transpose(np.array(group))
            del (group)
            aa2cg = {0:self.CA_atn,1:self.CB_atn,\
                     5:self.P_atn,2:self.S_atn,3:self.B_atn}
            
            cutoff = cmap.cutoff*cmap.scale
            resgap = 4 
            contacts_dict = dict()
            
            print ("> Determining contacts for %d*%d atom pairs using %.2f A cutoff and %.2f scaling-factor"%(aa_data.xyz.shape[0],aa_data.xyz.shape[0],cmap.cutoff,cmap.scale))
            for i in trange(aa_data.xyz.shape[0]):
                #resgap = 4:CA-CA, 3:CA-CB, 3:CB-CB, 
                gap=resgap-np.int_(bb_sc+bb_sc[i]>0) #aa2cg bbsc CA:0,CB:1
                #resgap 1: P/B/S-P/S/B
                gap=gap+(1-gap)*int(bb_sc[i]>=2) #aa2cg bbsc P:5,B:3,S:2

                calculate = np.int_( (np.int_(rnum-gap>=rnum[i]) * \
                            np.int_(cid==cid[i]) + np.int_(cid>cid[i])) > 0 )
                contact=np.where(np.int_(np.sum((aa_data.xyz-aa_data.xyz[i])**2,1)**0.5<=cutoff)*calculate==1)[0]
                for x in contact:
                    faa.write("%d %d %d %d\n"%(cid[i]+1,i+1,cid[x]+1,x+1))
                    cg_a1 = aa2cg[bb_sc[i]][cid[i]][rnum[i]]
                    cg_a2 = aa2cg[bb_sc[x]][cid[x]][rnum[x]]
                    set = (cid[i],cid[x]),(cg_a1,cg_a2)
                    if set not in contacts_dict: contacts_dict[set] = 0
                    contacts_dict[set] += 1
            contacts_dict = {y:(x,contacts_dict[(x,y)]) for x,y in contacts_dict}
            pairs = list(contacts_dict.keys()); pairs.sort()
            weights = np.float_([contacts_dict[x][1] for x in pairs])
            weights = weights*(weights.shape[0]/np.sum(weights))
            if not cmap.W: weights = np.ones(weights.shape)
            cid = np.int_([contacts_dict[x][0] for x in pairs])
            pairs = np.int_(pairs)
            distances = self.__distances__(pairs=pairs)
            for x in range(pairs.shape[0]):
                c,a = cid[x]+1,pairs[x]+1
                w,d = weights[x],distances[x]
                fcg.write("%d %d %d %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
            faa.close();fcg.close()

        elif cmap.type == 2:        # Calculating contacts from CG structure
            cid = []
            if len(self.CA_atn) != 0:
                cacasep=4;cacbsep=3;cbcbsep=3
                for c1 in self.CA_atn:
                    pairs += [(self.CA_atn[c1][x],self.CA_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c1] if y-x>=cacasep]
                    if len(self.CB_atn) != 0:
                        pairs += [(self.CA_atn[c1][x],self.CB_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CA_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CB_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c1] if y-x>=cbcbsep]
                    cid += [(c1,c1) for x in range(len(pairs)-len(cid))]
                    for c2 in self.CA_atn:
                        if c2>c1: 
                            pairs += [(self.CA_atn[c1][x],self.CA_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c2]]
                            if len(self.CB_atn)!=0: 
                                pairs += [(self.CA_atn[c1][x],self.CB_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CA_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CB_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c2]]
                            cid += [(c1,c2) for x in range(len(pairs)-len(cid))]
            if len(self.P_atn)!=0:
                ppsep,ressep=1,1
                if len(self.S_atn)==0:
                    for c1 in self.P_atn:
                        pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c1] if y-x>=ppsep]
                        cid += [(c1,c1) for x in range(len(pairs)-len(cid))]
                        for c2 in self.P_atn: 
                            if c2>c1:
                                pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c2]]
                                cid += [(c1,c2) for x in range(len(pairs)-len(cid))]
                else:
                    assert len(self.B_atn)!=0
                    for c1 in self.S_atn:
                        all_atn_c1 = list(self.P_atn[c1].items())+list(self.S_atn[c1].items())+list(self.B_atn[c1].items())
                        pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c1 if ry-rx>=ressep]
                        cid += [(c1,c1) for x in range(len(pairs)-len(cid))]
                        for c2 in self.S_atn:
                            if c2>c1:
                                all_atn_c2 = list(self.P_atn[c2].items())+list(self.S_atn[c2].items())+list(self.B_atn[c2].items())
                                pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c2]
                                cid += [(c1,c2) for x in range(len(pairs)-len(cid))]

            pairs = np.int_(pairs)
            cid = np.int_(cid)
            cutoff = 0.1*cmap.cutoff
            distances = self.__distances__(pairs)
            contacts = np.where(np.int_(distances<=cutoff))[0]
            pairs = pairs[contacts]
            cid = cid[contacts]
            distances = distances[contacts]
            check_dist = self.__distances__(pairs)
            for x in range(pairs.shape[0]): 
                assert check_dist[x] == distances[x] and distances[x] < cutoff
            weights = np.ones(pairs.shape[0])
            with  open(aa_data.pdbfile+".CGcont","w+") as fcg:
                for x in range(pairs.shape[0]):
                    c,a = cid[x]+1,pairs[x]+1
                    w,d = weights[x],distances[x]
                    fcg.write("%d %d %d %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
        
        self.contacts.append((pairs,distances,weights))
        return

class MergeTop:
    
    def __topParse__(self,topfile):
        print ("> Parsing",topfile)
        top = {x.split("]")[0].strip():x.split("]")[1].strip().split("\n") for x in open(topfile).read().split("[") if len(x.split("]"))==2}
        extras = [x.split() for x in open(topfile).read().split("[") if len(x.split("]"))<2][0]
        order = [x.split("]")[0].strip() for x in open(topfile).read().split("[") if len(x.split("]"))==2]
        return top,extras,order

    def nPlaces(self,n,count2str):
        return "0"*(n-len(str(count2str)))+str(count2str)

    def __writeAtomsSection__(self,fsec,inp,nmol,tag,prev_at_count=0):
        #writing merged atoms section
        offset = 0
        for y in inp:
            if y.strip().startswith(";") or y.strip() == "": continue
            offset = 1 - int(y.split()[0])
            break
        atoms_in_mol = offset
        for x in range(0,nmol):
            fsec.write(tag+self.nPlaces(n=3,count2str=x+1)+"\n")
            for y in inp:
                if y.strip() == "": continue
                elif y.strip().startswith(";"): fsec.write(y+"\n")
                else:
                    i=y.strip().split()
                    a0 = offset + int(i[0]) + prev_at_count + x*atoms_in_mol
                    a5 = offset + int(i[5]) + prev_at_count + x*atoms_in_mol
                    fsec.write("  %5d %5s %4s %5s %5s %5d %5s %5s\n" % (a0, i[1], i[2], i[3], i[4], a5, i[6], i[7]))
                    if x==0: atoms_in_mol = a0
        return atoms_in_mol,offset

    def __writeInteractions__(self,fsec,nparticles,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset):
        for x in range(0,nmol):
            fsec.write(tag+self.nPlaces(n=3,count2str=x+1)+"\n")
            for y in inp:
                if y.strip() == "": continue
                elif y.strip().startswith(";"): fsec.write(y+"\n")
                else:
                    line=y.strip() .split()
                    a = [atnum_offset + int(line[i]) + prev_at_count + x*atoms_in_mol for i in range(nparticles)]
                    fsec.write(nparticles*" %5d"%tuple(a))
                    fsec.write(len(line[nparticles:])*" %5s"%tuple(line[nparticles:]))
                    fsec.write("\n")
        return

    def __writeSymPaIrs__(self,fsec,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset):
        fsec.write(tag+"symmetrized_interactions_"+self.nPlaces(n=3,count2str=nmol)+"\n")
        for y in inp:
            if y.strip() == "": continue
            elif y.strip().startswith(";"): fsec.write(y+"\n")
            else:
                line=y.strip() .split()
                group = []
                I = [atnum_offset + int(line[0]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
                J = [atnum_offset + int(line[1]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
                for i in range(len(I)):
                    for j in range(len(J)): 
                        if i!=j: 
                            fsec.write(2*" %5d"%(I[i],J[j]))
                            fsec.write(len(line[2:])*" %5s"%tuple(line[2:]))
                            fsec.write("\n")
        return

    def __writeNonbondParams__(self,fsec):
        print ("> Writing user given custom nonbond_params:",self.interface)
        
        eps,sig = {},{}
        if self.interface:
            with open("interactions.dat") as fin:
                for line in fin:
                    if line.startswith(("#",";","@")): continue
                    k0,k1 = line.split()[:2]
                    eps[(k0,k1)] = float(line.split()[2])
                    eps[(k1,k0)] = eps[(k0,k1)]
                    sig[(k0,k1)] = 0.1*float(line.split()[3])
                    sig[(k1,k0)] = sig[(k0,k1)]

        if self.excl_rule == 2:
            pairs = []
            Krep = (self.fconst.Kr_prot+self.fconst.Kr_nucl)*0.5
            for x in self.excl_volume:
                if not x.startswith(("CA","CB")):
                    for y in self.excl_volume:
                        if y.startswith(("CA","CB")):
                            C10,C12 = 0.0, Krep*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                            p = [x,y]; p=tuple(p)
                            if p not in pairs and p not in eps:
                                fsec.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))
                                pairs.append(p)
        if len(eps)>0:
            fsec.write("; Custom Protein-RNA/DNA interactions\n")
            if self.nucl_cmap.func in (5,6):
                fsec.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            for x,y in eps:
                if y.startswith(("CA","CB")) and not x.startswith(("CA","CB")):
                    if x not in self.excl_volume or y not in self.excl_volume: continue
                    p = (x,y)
                    if self.nucl_cmap.func==1: #6-12
                        C06 = 2*eps[p]*((sig[p])**6)
                        C12 = 1*eps[p]*((sig[p])**12)
                        fsec.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C06,C12))
                    elif self.nucl_cmap.func==2:
                        C10 = 6*eps[p]*((sig[p])**10)
                        C12 = 5*eps[p]*((sig[p])**12)
                        fsec.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))
                    elif self.nucl_cmap.func in (5,6):
                        func,sd = 6,0.05
                        if self.excl_rule==1: c12 = Krep*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                        elif self.excl_rule==2: C12 = Krep*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                        fsec.write(" %s %s\t%d\t%.3f %e %e %e\n"%(p[0].ljust(5),p[1].ljust(5),func,eps[p],sig[p],sd,C12))
        return
                            
    def __init__(self,Nprot,Nnucl,topfile,opt,excl_volume,excl_rule,fconst,cmap):
        self.nucl_cmap = cmap["nucl"]
        self.prot_cmap = cmap["prot"]
        self.interface = opt.interface
        self.excl_volume = excl_volume
        self.excl_rule = excl_rule
        self.fconst = fconst
        self.__merge__(Nprot=Nprot,Nnucl=Nnucl,topfile=topfile,opt=opt,excl_volume=excl_volume)

    def __merge__(self,Nprot,Nnucl,topfile,opt,excl_volume):
        #combining top file
        outfile = "merged_"
        assert Nprot+Nnucl>1, "Error, cannot use combine with 1 molecule"
        prot_top,nucl_top={},{}
        if Nnucl>0:
            nucl_top,extras,oerder = self.__topParse__(topfile="nucl_"+topfile)
            outfile += "nucl"+self.nPlaces(3,Nnucl)+"_"
        if Nprot>0: 
            prot_top,extras,order = self.__topParse__(topfile="prot_"+topfile)
            outfile += "prot"+self.nPlaces(3,Nprot)+"_"
        outfile = outfile+topfile       
        if Nnucl==0: nucl_top = {k:[""] for k in prot_top}
        if Nprot==0: prot_top = {k:[""] for k in nucl_top}
        print (">>> writing Combined GROMACS toptology", outfile)
        with open(outfile,"w+") as fout:
            Nparticles = {"bonds":2,"angles":3,"pairs":2,"dihedrals":4,"exclusions":2}
            fout.write("\n; Topology file generated by eSBM.\n")
            for header in order:
                prot_data,nucl_data = prot_top[header],nucl_top[header]
                print ("> Writing",outfile+".top",header,"section.")
                fout.write("\n[ "+header+" ]\n")
                if header in ["nonbond_params","atomtypes"]:
                    if Nnucl>0: status=[fout.write(i+"\n") for i in nucl_data if i.strip() != ""]
                    if Nprot>0: status=[fout.write(i+"\n") for i in prot_data if i.strip() != ""]
                    if Nnucl>0 and Nprot>0: 
                        if header == "nonbond_params": self.__writeNonbondParams__(fsec=fout)
                elif header == "atoms":
                    natoms_nucl,off0= self.__writeAtomsSection__(fsec=fout,inp=nucl_data,nmol=Nnucl,tag=";RNA/DNA_")
                    natoms_prot,off1 = self.__writeAtomsSection__(fsec=fout,inp=prot_data,nmol=Nprot,tag=";Protein_",\
                                                                            prev_at_count=Nnucl*natoms_nucl)
                elif header in ["bonds","angles","pairs","dihedrals","exclusions"]:                
                    self.__writeInteractions__(fsec=fout,nparticles=Nparticles[header],inp=nucl_data,nmol=Nnucl,\
                                        prev_at_count=0,atoms_in_mol=natoms_nucl,tag=";RNA/DNA_",atnum_offset=off0)
                    self.__writeInteractions__(fsec=fout,nparticles=Nparticles[header],inp=prot_data,nmol=Nprot, \
                        prev_at_count=Nnucl*natoms_nucl,atoms_in_mol=natoms_prot,tag=";Protein_",atnum_offset=off1)
                    if opt.intra_symmetrize and header in ["pairs","exclusions"]:
                        self.__writeSymPaIrs__(fsec=fout,inp=prot_data,nmol=Nprot, \
                            prev_at_count=Nnucl*natoms_nucl,atoms_in_mol=natoms_prot,tag=";Protein_",atnum_offset=off1)
                else:
                    if Nnucl>0: status=[fout.write(i+"\n") for i in nucl_data if i.strip() != ""]
                    elif Nprot>0: status=[fout.write(i+"\n") for i in prot_data if i.strip() != ""]

class Topology:
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

    def __write_header__(self,fout,combrule) -> None:
            print (">> Writing header section")
            fout.write("\n; Topology file generated by eSBM. \n")
            fout.write("\n[ defaults  ]\n")
            fout.write("; nbfunc comb-rule gen-pairs\n")
            fout.write("  1      1         no   \n\n")
            return

    def __write_footer__(self,fout) -> None:
            print (">> Writing tail section")
            fout.write("\n%s\n"%("[ system ]"))
            fout.write("%s\n"%(";name"))
            fout.write("  %s\n"%("Macromolecule"))
            fout.write("\n%s\n"%("[ molecules ]"))
            fout.write("%s\n"%(";name    #molec"))
            fout.write("%s\n"%("Macromolecule     1"))
            return

    def __write_atomtypes__(self,fout,type,rad,seq,data):
        print (">> Writing atomtypes section")
        #1:CA model or 2:CA+CB model
        fout.write('%s\n'%("[ atomtypes ]"))
        fout.write(6*"%s".ljust(5)%("; name","mass","charge","ptype","C6(or C10)","C12"))

        if len(data.CA_atn) != 0:
            assert type<=2
            self.excl_volume["CA"] = 2*rad["CA"]
            C12 = self.fconst.Kr_prot*(2*rad["CA"])**12.0
            fout.write("\n %s %8.3f %8.3f %s %e %e; %s\n"%("CA".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CA"))
            if type == 2:
                for s in seq:
                    bead = "CB"+s
                    if bead in self.excl_volume or s == " ": continue
                    C12 = self.fconst.Kr_prot*(2*rad[bead])**12.0
                    self.excl_volume[bead] = 2*rad[bead]
                    fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CB"))
            lego = False
            if lego: print ("WIP for lego")
        if len(data.P_atn) != 0:
            assert type in (1,3,5)
            self.excl_volume["P"] = 2*rad["P"]
            C12 = self.fconst.Kr_nucl*(2*rad["P"])**12.0
            fout.write("\n %s %8.3f %8.3f %s %e %e; %s\n"%("P".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"P"))
            if type > 1:
                codon_based = False
                seq_list = seq.split()
                assert len(seq_list) == len(data.S_atn)
                for c in data.S_atn:
                    assert c in data.B_atn
                    seq = "5"+seq_list[c]+"3"
                    for i in range(1,len(seq)-1):
                        if codon_based: codon = seq[i-1].lower()+seq[i]+seq[i+1].lower()
                        else: codon = seq[i]
                        for tag in ("S","B"):
                            bead = tag+"0"+codon
                            if bead not in self.excl_volume:
                                self.excl_volume[bead] = 2*rad[tag+seq[i]]
                                C12 = self.fconst.Kr_prot*(self.excl_volume[bead])**12.0
                                fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,"S"))
        return

    def __write_nonbond_params__(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))
        if type == 1 or excl_rule == 1: return
        if len(data.CA_atn) > 0:
            if excl_rule == 2 and type == 2:
                pairs = []
                for x in self.excl_volume:
                    if x.startswith(("CA","CB")):
                        for y in self.excl_volume:
                            if y.startswith(("CA","CB")):
                                C10,C12 = 0.0, self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                                p = [x,y]; p.sort(); p=tuple(p)
                                if p not in pairs:
                                    fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))
                                    pairs.append(p)
        if len(data.P_atn) > 0:
            if excl_rule == 2 and type in (3,5):
                pairs = []
                for x in self.excl_volume:
                    if x.startswith(("P","S","B")):
                        for y in self.excl_volume:
                            if y.startswith(("P","S","B")):
                                C10,C12 = 0.0, self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                                p = [x,y]; p.sort(); p=tuple(p)
                                if p not in pairs:
                                    fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))
                                    pairs.append(p)

        return 0

    def __write_moleculetype__(self,fout):
        print (">> Writing moleculetype section")
        fout.write("\n%s\n"%("[ moleculetype ]"))
        fout.write("%s\n"%("; name            nrexcl"))
        fout.write("%s\n"%("  Macromolecule   3"))

    def __write_atoms__(self,fout,type,data,inc_charge):
        print (">> Writing atoms section")
        cgfile = data.pdbfile
        #if type==1: cgfile = data.prot.bb_file
        #else: cgfile = data.prot.sc_file
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = dict()
        if inc_charge: 
            Q.update({x:1 for x in ["CBK","CBR","CBH"]})
            Q.update({x:-1 for x in ["CBD","CBE","P"]})
        with open(cgfile) as fin:
            for line in fin:
                if line.startswith("ATOM"):
                    atnum=hy36decode(5,line[6:11])
                    atname=line[12:16].strip()
                    resname=line[17:20].strip()
                    resnum=hy36decode(4,line[22:26])
                    atype=atname
                    if atype=="CB": atype+=data.prot.amino_acid_dict[resname]
                    elif atype.endswith("'"): atype="S"+atype[1]+resname[-1]
                    elif atype.startswith("N"): atype="B"+atype[1]+resname[-1]
                    if atype not in Q: Q[atype] = 0
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
        dswap = False
        if dswap: print ("WIP")
        return 

    def __write_protein_bonds__(self,fout,data,func):
        print (">> Writing bonds section")
        #GROMACS IMPLEMENTS Ebonds = (Kx/2)*(r-r0)^2
        #Input units KJ mol-1 A-2 GROMACS units KJ mol-1 nm-1 (100 times the input value) 
        Kb = float(self.fconst.Kb_prot)*100.0

        #GROMACS 4.5.4 : FENE=7 AND HARMONIC=1
        #if dsb: func = 9
        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "r0(nm)", "Kb"))

        data.Bonds()
        for pairs,dist in data.bonds:
            I,J = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,dist[x],Kb))
        return 

    def __write_protein_angles__(self,fout,data):
        print (">> Writing angless section")
        #V_ang = (Ktheta/2)*(r-r0)^2
        #Input units KJ mol-1 #GROMACS units KJ mol-1 
        Ka = float(self.fconst.Ka_prot)

        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))

        func = 1
        data.Angles()
        for triplets,angles in data.angles:
            I,J,K = 1+np.transpose(triplets)
            for x in range(triplets.shape[0]): 
                fout.write(" %5d %5d %5d %5d %e %e\n"%(I[x],J[x],K[x],func,angles[x],Ka))
        return

    def __write_protein_dihedrals__(self,fout,data,chiral):
        print (">> Writing dihedrals section")

        #GROMACS IMPLEMENTATION: Edihedrals Kphi*(1 + cos(n(phi-phi0)))
        #Our implementaion: Edihedrals = Kphi*(1 - cos(n(phi-phi0)))
        #The negative sign is included by adding phase = 180 to the phi0
        #Kphi*(1 + cos(n(phi-180-phi0))) = Kphi*(1 + cos(n180)*cos(n(phi-phi0)))
        #if n is odd i.e. n=1,3.... then cos(n180) = -1
        #hence Edihedrals = Kphi*(1 - cos(n(phi-phi0)))

        Kd_bb = float(self.fconst.Kd_prot["bb"])
        Kd_sc = float(self.fconst.Kd_prot["sc"])
        mfac = float(self.fconst.Kd_prot["mf"])

        phase = 180

        fout.write("\n%s\n"%("[ dihedrals ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s %5s %5s\n" % (";ai","aj","ak","al","func","phi0(deg)","Kd","mult"))

        data.Dihedrals()
        func = 1
        for quads,diheds in data.bb_dihedrals:
            I,J,K,L = 1+np.transpose(quads)
            diheds += phase
            for x in range(quads.shape[0]):
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_bb,1))
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_bb/mfac,3))
        if chiral and len(data.CB_atn) != 0:
            func = 2
            fout.write("; %5s %5s %5s %5s %5s %5s %5s \n" % (";ai","aj","ak","al","func","phi0(deg)","Kd"))
            for quads,diheds in data.sc_dihedrals:
                I,J,K,L = 1+np.transpose(quads)
                for x in range(quads.shape[0]):fout.write(" %5d %5d %5d %5d %5d %e %e\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_sc))
        return

    def __write_protein_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing pairs section")
        cmap = self.cmap["prot"]
        data.Pairs(cmap=cmap,aa_data=self.allatomdata.prot)
        if cmap.scsc_custom:
            assert cmap.type in (1,2)
            with open("interactions.dat") as fin:
                scscmat = {tuple(line.split()[:2]):float(line.split()[2]) \
                        for line in fin if not line.startswith(("#",";","@"))}
                scscmat.update({(k[1],k[0]):v for k,v in scscmat.items()})                
            CB_atn = {v:"CB"+Prot_Data().amino_acid_dict[k[2]] \
                for k,v in self.allatomdata.prot.CB_atn.items()}
            all_atn = CB_atn.copy()
            all_atn.update({v:"CA"+Prot_Data().amino_acid_dict[k[2]] \
                        for k,v in self.allatomdata.prot.CA_atn.items()})
            for index in range(len(data.contacts)):
                pairs,dist,eps = data.contacts[index]
                I,J = np.transpose(pairs)
                interaction_type = \
                        np.int_([x in CB_atn for x in I])+ \
                        np.int_([x in CB_atn for x in J])
                scscmat.update({(all_atn[I[x]],all_atn[J[x]]):0.0 for x in range(I.shape[0]) if (all_atn[I[x]],all_atn[J[x]]) not in scscmat})
                eps_scsc = np.float_([scscmat[(all_atn[I[x]],all_atn[J[x]])] for x in range(I.shape[0])])
                eps_bbsc = np.float_(eps)
                eps_bbbb = np.float_(eps)
                eps = eps_bbbb*np.int_(interaction_type==0) \
                    + eps_bbsc*np.int_(interaction_type==1) \
                    + eps_scsc*np.int_(interaction_type==2) 
                #for x in range(eps.shape[0]): print(eps[x],interaction_type[x],new_eps[x])
                #eps = np.round(eps + (eps_IJ - eps)*interaction_type,3)
                data.contacts[index] = pairs,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        if cmap.func==1:
            print ("> Using LJ C6-C12 for contacts")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
                c06 = 2*eps*(dist**6.0)
                c12 = eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))
        elif cmap.func==2:
            print ("> Using LJ C10-C12 for contacts. Note: Require Table file(s)")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
                c10 = 6*eps*(dist**10.0)
                c12 = 5*eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c10[x],c12[x]))
        elif cmap.func==3:
            print ("> Using LJ C12-C18 for contacts. Note: Require Table file(s) or ")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C12(Att)","C18(Rep)"))
            func = 3
            assert func!=3, "Error, func 3 not encoded yes. WIP"
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
        elif cmap.func in (5,6):
            fout.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            func = 6
            sd = 0.05
            for pairs,dist,eps in data.contacts:
                I,J = np.transpose(pairs)
                I = np.float_([self.excl_volume[self.atomtypes[x]] for x in I])
                J = np.float_([self.excl_volume[self.atomtypes[x]] for x in J])
                if excl_rule == 1: c12 = ((I**12.0)*(J**12.0))**0.5
                elif excl_rule == 2: c12 = ((I+J)/2.0)**12.0
                I,J = 1+np.transpose(pairs)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(I[x],J[x],func,eps[x],dist[x],sd,c12[x]))
        return 

    def __write_nucleicacid_bonds__(self,fout,data,func):
        print (">> Writing bonds section")
        #GROMACS IMPLEMENTS Ebonds = (Kx/2)*(r-r0)^2
        #Input units KJ mol-1 A-2 GROMACS units KJ mol-1 nm-1 (100 times the input value) 
        Kb = float(self.fconst.Kb_nucl)*100.0

        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "r0(nm)", "Kb"))

        data.Bonds()
        for pairs,dist in data.bonds:
            I,J = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,dist[x],Kb))
        return 

    def __write_nucleicacid_angles__(self,fout,data):
        print (">> Writing angless section")
        #V_ang = (Ktheta/2)*(r-r0)^2
        #Input units KJ mol-1 #GROMACS units KJ mol-1 
        Ka = float(self.fconst.Ka_nucl)

        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))

        func = 1
        data.Angles()
        for triplets,angles in data.angles:
            I,J,K = 1+np.transpose(triplets)
            for x in range(triplets.shape[0]): 
                fout.write(" %5d %5d %5d %5d %e %e\n"%(I[x],J[x],K[x],func,angles[x],Ka))
        return

    def __write_nucleicacid_dihedrals__(self,fout,data,chiral):
        print (">> Writing dihedrals section")

        #GROMACS IMPLEMENTATION: Edihedrals Kphi*(1 + cos(n(phi-phi0)))
        #Our implementaion: Edihedrals = Kphi*(1 - cos(n(phi-phi0)))
        #The negative sign is included by adding phase = 180 to the phi0
        #Kphi*(1 + cos(n(phi-180-phi0))) = Kphi*(1 + cos(n180)*cos(n(phi-phi0)))
        #if n is odd i.e. n=1,3.... then cos(n180) = -1
        #hence Edihedrals = Kphi*(1 - cos(n(phi-phi0)))

        Kd_bb = float(self.fconst.Kd_nucl["bb"])
        Kd_sc = float(self.fconst.Kd_nucl["sc"])
        mfac = float(self.fconst.Kd_nucl["mf"])

        phase = 180

        fout.write("\n%s\n"%("[ dihedrals ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s %5s %5s\n" % (";ai","aj","ak","al","func","phi0(deg)","Kd","mult"))

        data.Dihedrals()

        func = 1
        for quads,diheds in data.bb_dihedrals:
            I,J,K,L = 1+np.transpose(quads)
            if self.opt.P_stretch: diheds=180*np.ones(diheds.shape)
            diheds += phase
            for x in range(quads.shape[0]):
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_bb,1))
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_bb/mfac,3))
        if len(data.S_atn):
            for quads,diheds in data.sc_dihedrals:
                I,J,K,L = 1+np.transpose(quads)
                diheds += phase
                for x in range(quads.shape[0]):
                    fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_sc,1))
                    fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_sc/mfac,3))
        return

    def __write_nucleicacid_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing pairs section")
        cmap = self.cmap["nucl"]
        data.Pairs(cmap=cmap,aa_data=self.allatomdata.nucl)

        fout.write("\n%s\n"%("[ pairs ]"))
        if cmap.func==1:
            print ("> Using LJ C6-C12 for contacts")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
                c06 = 2*eps*(dist**6.0)
                c12 = eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))
        elif cmap.func==2:
            print ("> Using LJ C10-C12 for contacts. Note: Require Table file(s)")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
                c10 = 6*eps*(dist**10.0)
                c12 = 5*eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c10[x],c12[x]))
        elif cmap.func==3:
            print ("> Using LJ C12-C18 for contacts. Note: Require Table file(s) or ")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C12(Att)","C18(Rep)"))
            func = 3
            assert func!=3, "Error, func 3 not encoded yes. WIP"
            for pairs,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
        elif cmap.func in (5,6):
            fout.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            func = 6
            sd = 0.05
            for pairs,dist,eps in data.contacts:
                I,J = np.transpose(pairs)
                I = np.float_([self.excl_volume[self.atomtypes[x]] for x in I])
                J = np.float_([self.excl_volume[self.atomtypes[x]] for x in J])
                if excl_rule == 1: c12 = ((I**12.0)*(J**12.0))**0.5
                elif excl_rule == 2: c12 = ((I+J)/2.0)**12.0
                I,J = 1+np.transpose(pairs)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(I[x],J[x],func,eps[x],dist[x],sd,c12[x]))
        return 

    def __write_exclusions__(self,fout,data):
        print (">> Writing exclusions section")
        fout.write("\n%s\n"%("[ exclusions ]"))
        fout.write("; %5s %5s\n"%("i","j"))
        for pairs,dist,eps in data.contacts:
            I,J = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d\n"%(I[x],J[x]))
        return

    def write_topfile(self,outtop,excl,charge,bond_function,CBchiral,rad):

        cgpdb = PDB_IO()
        if len(self.allatomdata.prot.lines) > 0 and self.CGlevel["prot"] in (1,2):
            if self.CGlevel["prot"]==1: cgpdb.loadfile(infile=self.allatomdata.prot.bb_file,refine=False)
            elif self.CGlevel["prot"]==2: cgpdb.loadfile(infile=self.allatomdata.prot.sc_file,refine=False)
            prot_topfile = "prot_"+outtop
            with open(prot_topfile,"w+") as ftop:
                print (">>> writing Protein GROMACS toptology", prot_topfile)
                proc_data = Calculate(pdb=cgpdb.prot)
                proc_data.processData(data=cgpdb.prot)
                self.__write_header__(fout=ftop,combrule=excl)
                self.__write_atomtypes__(fout=ftop,data=proc_data,type=self.CGlevel["prot"],seq=cgpdb.prot.seq,rad=rad)
                self.__write_nonbond_params__(fout=ftop,data=proc_data,type=self.CGlevel["prot"],excl_rule=excl)
                self.__write_moleculetype__(fout=ftop)
                self.__write_atoms__(fout=ftop,type=self.CGlevel["prot"], data=cgpdb,inc_charge=(charge.CA or charge.CB))
                self.__write_protein_pairs__(fout=ftop, data=proc_data,excl_rule=excl,charge=charge)
                self.__write_protein_bonds__(fout=ftop, data=proc_data,func=bond_function)
                self.__write_protein_angles__(fout=ftop, data=proc_data)
                self.__write_protein_dihedrals__(fout=ftop, data=proc_data,chiral=CBchiral)
                self.__write_exclusions__(fout=ftop,data=proc_data)
                self.__write_footer__(fout=ftop)

        if len(self.allatomdata.nucl.lines) > 0 and self.CGlevel["nucl"] in (1,3,5):
            if self.CGlevel["nucl"]==1: cgpdb.loadfile(infile=self.allatomdata.nucl.bb_file,refine=False)
            elif self.CGlevel["nucl"] in (3,5): cgpdb.loadfile(infile=self.allatomdata.nucl.sc_file,refine=False)
            nucl_topfile = "nucl_"+outtop
            with open(nucl_topfile,"w+") as ftop:
                print (">>> writing RNA/DNA GROMACS toptology", nucl_topfile)
                proc_data = Calculate(pdb=cgpdb.nucl)
                proc_data.processData(data=cgpdb.nucl)
                self.__write_header__(fout=ftop,combrule=excl)
                self.__write_atomtypes__(fout=ftop,type=self.CGlevel["nucl"],data=proc_data,seq=cgpdb.nucl.seq,rad=rad)
                self.__write_nonbond_params__(fout=ftop,data=proc_data,type=self.CGlevel["nucl"],excl_rule=excl)
                self.__write_moleculetype__(fout=ftop)
                self.__write_atoms__(fout=ftop,type=self.CGlevel["prot"], data=cgpdb,inc_charge=charge.P)
                self.__write_nucleicacid_pairs__(fout=ftop, data=proc_data,excl_rule=excl,charge=charge)
                self.__write_nucleicacid_bonds__(fout=ftop, data=proc_data,func=bond_function)
                self.__write_nucleicacid_angles__(fout=ftop, data=proc_data)
                self.__write_nucleicacid_dihedrals__(fout=ftop, data=proc_data,chiral=CBchiral)
                self.__write_exclusions__(fout=ftop,data=proc_data)
                self.__write_footer__(fout=ftop)

        Nmol = self.Nmol
        #if len(self.allatomdata.nucl.lines) > 0 and self.CGlevel["nucl"] in (1,3,5):
        if Nmol["prot"]+Nmol["nucl"] > 1:
            if len(self.allatomdata.prot.lines) > 0 and self.CGlevel["prot"] in (1,2):
                merge=MergeTop(Nprot=Nmol["prot"],Nnucl=Nmol["nucl"],topfile=outtop,opt=self.opt,excl_volume=self.excl_volume,excl_rule=excl,fconst=self.fconst,cmap=self.cmap)

        table = Tables()
        if self.cmap["prot"].func == 2 or self.cmap["nucl"].func == 2:
            table.__write_pair_table__(elec=charge,ljtype=2)
        elif self.cmap["prot"].func == 1 or self.cmap["nucl"].func == 1:
            if charge.debye: table.__write_pair_table__(elec=charge,ljtype=1)
        return

class Clementi2000(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

class Pal2019(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

    def __write_nucleicacid_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing pairs section")
        cmap = self.cmap["nucl"]
        fout.write("\n%s\n"%("[ pairs ]"))
        assert cmap.func==2
        print ("> Using LJ C10-C12 for Stackubg. Note: Require Table file(s)")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
        func = 1

        with open("interactions.dat") as fin:
            epsmat,stack = {},{}
            for line in fin:
                if line.startswith(("#",";","@")): continue
                k0,k1 = [x[-1] for x in line.split()[:2]]
                if k0 in "AGCUT" and k1 in "AGCUT":
                    epsmat[k0+k1] = float(line.split()[2])
                    epsmat[k1+k0] = epsmat[k0+k1]
                    stack[k0+k1] = 0.1*float(line.split()[3])
                    stack[k1+k0] = stack[k0+k1]

        B_atn = {v:k[2] for k,v_list in self.allatomdata.nucl.B_atn.items() for v in v_list}
        for c in data.B_atn:
            resnum = list(data.B_atn[c].keys())
            resnum.sort()
            pairs = np.int_([(data.B_atn[c][x],data.B_atn[c][x+1]) for x in resnum if x+1 in data.B_atn[c]])
            I,J = np.transpose(pairs)
            eps = np.float_([epsmat[B_atn[I[x]][-1]+B_atn[J[x]][-1]] for x in range(I.shape[0])])
            dist = np.float_([stack[B_atn[I[x]][-1]+B_atn[J[x]][-1]] for x in range(I.shape[0])])
            data.contacts.append((pairs,dist,eps))
            I,J = 1+np.transpose(pairs)
            c10 = 6*eps*(dist**10.0)
            c12 = 5*eps*(dist**12.0)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c10[x],c12[x]))
        return 

class Reddy2017(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.__check_H_atom__()
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

    def __check_H_atom__(self):
        # checking presence of H-atom. Important for adding GLY CB
        data = self.allatomdata.prot.res
        data = [(x[1],x[-1]) for x in data if "GLY" in x]
        atlist = {}
        for x in data:
            if x[0] not in atlist: atlist[x[0]] = list()
            atlist[x[0]].append(x[1])
        for rnum in atlist:
            assert "HA3" in atlist[rnum] \
                or "HA2" in atlist[rnum] \
                or "CB" in atlist[rnum], "Error, SOP-SC needs H-atoms in the PDB file."
        return

    def __write_nonbond_params__(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))
        assert type==2 and excl_rule == 2
        pairs = []
        for x in self.excl_volume:
            if x.startswith(("CA","CB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        C06 = -1*self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**6
                        C12 = 0.0
                        p = [x,y]; p.sort(); p=tuple(p)
                        if p not in pairs:
                            fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C06,C12))
                            pairs.append(p)
        return 0

    def __write_protein_bonds__(self,fout,data,func):
        print (">> Writing SOP-SC bonds section")
        print ("Note: Function not supported by GROMACS. Use table files or enable --opensmog")

        #GROMACS IMPLEMENTS Ebonds = (Kx/2)*(r-r0)^2
        #Input units KJ mol-1 A-2 GROMACS units KJ mol-1 nm-1 (100 times the input value) 
        K = float(self.fconst.Kb_prot)*100.0


        #GROMACS 4.5.4 : FENE=7 AND HARMONIC=1
        #if dsb: func = 9
        assert func == 8
        R = 0.2

        # V = - (K/2)*R^2*ln(1-((r-r0)/R)^2)
        # V_1 = dV/dr = -K*0.5*R^2*(1/(1-((r-r0)/R)^2))*(-2*(r-r0)/R^2)
        #             = -K*0.5*(-2)(r-r0)/(1-((r-r0)/R)^2)
        #             = K*(R^2)(r-r0)/(R^2-(r-r0)^2)

        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "table_no.", "Kb"))
        data.Bonds()
        table_idx = dict()
        for pairs,dist in data.bonds:
            I,J = 1+np.transpose(pairs) 
            for i in range(pairs.shape[0]): 
                r0 = np.round(dist[i],3)
                if r0 not in table_idx: table_idx[r0]=len(table_idx)
                r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                V = -0.5*(R**2)*np.log(1-((r-r0)/R)**2)
                #V_1 = -0.5*(R**2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                V_1 = (R**2)*(r-r0)/(R**2-(r-r0)**2)
                Tables().__write_bond_table__(X=r,index=table_idx[r0],V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e\n"%(I[i],J[i],func,table_idx[r0],K))
        return 

    def __write_protein_angles__(self,fout,data):
        print (">> Not Writing angless section")
        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))
        return

    def __write_protein_dihedrals__(self,fout,data,chiral):
        print (">> Not Writing dihedrals section")
        fout.write("\n%s\n"%("[ dihedrals ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s %5s %5s\n" % (";ai","aj","ak","al","func","phi0(deg)","Kd","mult"))
        return

    def __write_protein_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing SOP-SC pairs section")
        cmap = self.cmap["prot"]
        data.Pairs(cmap=cmap,aa_data=self.allatomdata.prot)
        assert cmap.scsc_custom and cmap.type in (-1,0,2) and cmap.func==1
        
        with open("interactions.dat") as fin:
            scscmat = {tuple(line.split()[:2]):float(line.split()[2]) \
                    for line in fin if not line.startswith(("#",";","@"))}
            scscmat.update({(k[1],k[0]):v for k,v in scscmat.items()})                
            
        CA_atn = {v:"CA"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CA_atn.items()}
        CB_atn = {v:"CB"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CB_atn.items()}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())
        eps_bbbb = 0.5*self.fconst.caltoj
        eps_bbsc = 0.5*self.fconst.caltoj
        Kboltz = self.fconst.Kboltz #*self.fconst.caltoj/self.fconst.caltoj
        for index in range(len(data.contacts)):
            pairs,dist,eps = data.contacts[index]
            I,J = np.transpose(pairs)
            interaction_type = np.int_(\
                np.int_([x in CB_atn for x in I])+ \
                np.int_([x in CB_atn for x in J]))

            scscmat.update({(all_atn[I[x]],all_atn[J[x]]):0.0 for x in range(I.shape[0]) if (all_atn[I[x]],all_atn[J[x]]) not in scscmat})
            eps_scsc = np.float_([scscmat[(all_atn[I[x]],all_atn[J[x]])] for x in range(I.shape[0])])
            eps_scsc = 0.5*(0.7-eps_scsc)*300*Kboltz
            eps = np.float_(eps)
            eps = eps_bbbb*np.int_(interaction_type==0) \
                + eps_bbsc*np.int_(interaction_type==1) \
                + eps_scsc*np.int_(interaction_type==2) 
            data.contacts[index] = pairs,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        print ("> Using LJ C6-C12 for contacts")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
        func = 1
        for pairs,dist,eps in data.contacts:
            I,J = 1+np.transpose(pairs)
            c06 = 2*eps*(dist**6.0)
            c12 = eps*(dist**12.0)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))

        print ("> Using -C6 repulsion for local beads")
        fout.write(";angle based rep temp\n;%5s %5s %5s %5s %5s\n"%("i","j","func","-C06(Rep)","C12 (N/A)"))        
        data.Angles()
        diam = self.excl_volume.copy()
        diam.update({"CA"+k[-1]:diam["CA"] for k in diam.keys() if k.startswith("CB")})
        eps_bbbb = 1.0*self.fconst.caltoj
        eps_bbsc = 1.0*self.fconst.caltoj
        
        for index in range(len(data.angles)):
            triplets,angles = data.angles[index]
            I,J,K = np.transpose(triplets)
            interaction_type = np.int_(\
                np.int_([x in CB_atn for x in I])+ \
                np.int_([x in CB_atn for x in K]))
            sig = [(all_atn[I[x]],all_atn[K[x]]) for x in range(K.shape[0])]
            sig = 0.5*np.float_([diam[x]+diam[y] for x,y in sig])
            assert 2 not in interaction_type
            c06 = -1*eps_bbbb*((1.0*sig)**6)*np.int_(interaction_type==0) \
                + -1*eps_bbsc*((0.8*sig)**6)*np.int_(interaction_type==1) 
            pairs = np.int_([(I[x],K[x]) for x in range(I.shape[0])])
            data.contacts.append((pairs,np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
            I,J,K = 1+np.transpose(triplets)
            for x in range(triplets.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))
        return 

class Baidya2022(Reddy2017):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1]}
        self.opt = opt
        self.eps_bbbb = 0.12*self.fconst.caltoj 
        self.eps_bbsc = 0.24*self.fconst.caltoj 
        self.eps_scsc = 0.18*self.fconst.caltoj 
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

    def __write_nonbond_params__(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6  C12'))
        assert type==2 and excl_rule == 2
        pairs = []
        eps = dict()
        eps[("CA","CA")] = self.eps_bbbb
        eps[("CB","CB")] = self.eps_bbsc
        eps[("CA","CB")] = self.eps_scsc

        with open("interactions.dat") as fin:
            epsmat = {tuple(line.split()[:2]):float(line.split()[2]) \
                for line in fin if not line.startswith(("#",";","@"))}
            epsmat = {k:np.abs(0.7-epsmat[k]) for k in epsmat}
            epsmat.update({(k[1],k[0]):epsmat[k] for k in epsmat})
            epsmat.update({("CA",k[0]):1.0 for k in epsmat if "CA" not in k})
            epsmat.update({(k[0],"CA"):1.0 for k in epsmat if "CA" not in k})
            epsmat.update({("CA","CA"):1.0})

        for x in self.excl_volume:
            if x.startswith(("CA","CB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        sig = (self.excl_volume[x]+self.excl_volume[y])/2.0
                        p = [x[:2],y[:2]]; p.sort(); p = tuple(p)
                        C06 = 2*eps[p]*epsmat[(x,y)]*(sig)**6
                        C12 = 1*eps[p]*epsmat[(x,y)]*(sig)**12
                        if p not in pairs:
                            fout.write(" %s %s\t1\t%e %e\n"%(x.ljust(5),y.ljust(5),C06,C12))
                            pairs.append(p)
        return 0  

    def __write_protein_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing SOP-SC pairs section")
        cmap = self.cmap["prot"]
        assert cmap.scsc_custom and cmap.type in (-1,0,2) and cmap.func==1

        CA_atn = {v:"CA"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CA_atn.items()}
        CB_atn = {v:"CB"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CB_atn.items()}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())

        fout.write("\n%s\n"%("[ pairs ]"))
        fout.write(";angle based rep temp\n;%5s %5s %5s %5s %5s\n"%("i","j","func","-C06(Rep)","C12(N/A)"))        
        func = 1
        diam = self.excl_volume.copy()
        diam.update({"CA"+k[-1]:diam["CA"] for k in diam.keys() if k.startswith("CB")})
        eps_bbbb = 1.0*self.fconst.caltoj
        eps_bbsc = 1.0*self.fconst.caltoj
        eps_scsc = 1.0*self.fconst.caltoj
    
        pairs  = []
        pairs += [(x,y) for x in CA_atn for y in range(x+3,x+6) if y in all_atn]
        pairs += [(x,y) for x in CB_atn for y in range(x+1,x+5) if y in all_atn]

        I,K = np.transpose(np.int_(pairs))
        interaction_type = np.int_(\
            np.int_([x in CB_atn for x in I])+ \
            np.int_([x in CB_atn for x in K]))
        sig = [(all_atn[I[x]],all_atn[K[x]]) for x in range(K.shape[0])]
        sig = 0.5*np.float_([diam[x]+diam[y] for x,y in sig])
        c06 = -1*eps_bbbb*((1.0*sig)**6)*np.int_(interaction_type==0) \
            + -1*eps_bbsc*((1.0*sig)**6)*np.int_(interaction_type==1) \
            + -1*eps_scsc*((1.0*sig)**6)*np.int_(interaction_type==2) 
        pairs = np.int_([(I[x],K[x]) for x in range(I.shape[0])])
        data.contacts.append((pairs,np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
        I,K = I+1,K+1
        for x in range(len(pairs)): 
            fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))
        return 
