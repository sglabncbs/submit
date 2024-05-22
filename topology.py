import numpy as np
from tqdm import trange,tqdm
from PDB_IO import *
from hybrid_36 import hy36encode,hy36decode

class Tables:
    def __init__(self) -> None:
        pass

    def __prePad__(self,X):
        step = 0.001 #nm
        return np.int_(range(0,int(X[0]*1000)))*0.001

    def __write_bond_table__(self,index,X,V,V_1):
        with open("table_b"+str(index)+".xvg","w+") as fout:
            for x in self.__prePad__(X):           
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

        r = np.int_(range(0,100000))*0.002 #100 nm
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
    def __init__(self,aa_pdb) -> None:
        self.allatpdb = aa_pdb
        self.cgpdb = Prot_Data
        self.cgpdb_n,self.cgpdb_p = self.cgpdb,self.cgpdb
        self.CA_atn = dict()
        self.CB_atn = dict()
        self.P_atn = dict()
        self.S_atn = dict()
        self.B_atn = dict()
        self.bonds,self.angles,self.bb_dihedrals,self.sc_dihedrals = [],[],[],[]
        self.contacts = []

    def __distances__(self,pairs,xyz0=None,xyz1=None):
        #takes pairs, retuns array of distances in nm
        i,j = np.transpose(pairs)
        if xyz0 is None: xyz0 = self.cgpdb.xyz
        if xyz1 is None: xyz1 = self.cgpdb.xyz
        return 0.1*np.sum((xyz1[j]-xyz0[i])**2,1)**0.5

    def __angles__(self,triplets):
        #takes list triplets, retuns array of angles (0-180) in deg
        i,j,k = np.transpose(triplets)
        xyz = self.cgpdb.xyz
        n1 = xyz[i]-xyz[j]; n2 = xyz[k]-xyz[j]
        n1n2 = (np.sum((n1**2),1)**0.5)*(np.sum((n2**2),1)**0.5)
        return np.arccos(np.sum(n1*n2,1)/n1n2)*180/np.pi

    def __torsions__(self,quadruplets):
        #takes list of quadruplets, retuns array of torsion angles (0-360) in deg
        i,j,k,l = np.transpose(quadruplets)
        xyz = self.cgpdb.xyz
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
        self.cgpdb=data
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

    def Interactions(self,interface=False,nonbond=False,pairs=False):
        assert int(interface)+int(pairs)+int(nonbond) in (0,1),\
            "Error, only one of interfae, pairs, nonbond can be True"
        eps,sig = {},{}
        if interface: infile="interactions.interface.dat"
        elif nonbond: infile="interactions.nonbond.dat"
        elif pairs:   infile="interactions.pairs.dat"
        else: return eps,sig
        with open(infile) as fin:
            for line in fin:
                if line.startswith(("#",";","@")): continue
                line=line.split()
                k0,k1 = line[:2]
                eps[(k0,k1)] = float(line[2])
                eps[(k1,k0)] = eps[(k0,k1)]
                if len(line[2:])>=2:
                    sig[(k0,k1)]= 0.1*np.float_(line[3:])
                    sig[(k1,k0)] = sig[(k0,k1)]
        return eps,sig

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
    
    def Pairs(self,cmap,group="all"):
        # Getting Non-bonded contact pairs info from the pre-supplied data
        temp_p,temp_c,temp_w,temp_d = [],[],[],[]
        pairs,chains,weights,distances = [],[],[],[]

        #identify what group to determine contacts for
        if group.lower() in ("p","prot","protein"): tagforfile,group="prot","prot"
        elif group.lower() in ("n","nucl","nucleic","rna","dna"): tagforfile,group="nucl","nucl"
        elif group.lower() == ("a","all"): tagforfile,group="all","all"
        elif group.lower() in ("i","inter"): tagforfile,group="interProtNucl","inter"

        if cmap.type == -1: return  # Generating top without pairs 

        elif cmap.type == 0:        # Use pairs from user input in format cid_i, atnum_i, cid_j, atnum_j, weight_ij (opt), dist_ij (opt)
            assert cmap.file != ""
            print ("> Using cmap file (c1 a1 c2 a2 w d)",cmap.file)
            with open(cmap.file) as fin:
                for line in fin:
                    line = line.split()
                    c1,a1,c2,a2 = line[:4]
                    a1,a2 = np.int_([a1,a2])-1
                    if c1!=c2 and c2.startswith("nucl"):
                        c1,a1,c2,a2 = c2,a2,c1,a1
                    if len(line) < 6:
                        w,d = 1.0,0.0
                        if len(line)==5: w = np.float(line[4])
                        temp_p.append((a1,a2));temp_w.append(w);temp_c.append((c1,c2))
                    elif len(line)==6: 
                        w,d = np.float_(line[4:])
                        pairs.append((a1,a2));chains.append((c1,c2))
                        weights.append(w);distances.append(d)
            if len(temp_p)!=0: 
                if group!="inter": temp_d = list(self.__distances__(pairs=np.int_(temp_p)))
                elif group=="inter": 
                    temp_d = list(self.__distances__(pairs=np.int_(temp_p),xyz0=self.cgpdb_n.xyz,xyz1=self.cgpdb_p.xyz))
            pairs += temp_p; chains += temp_c; weights += temp_w; distances += temp_d
            pairs = np.int_(pairs); weights = np.float_(weights); distances = np.float_(distances)

        elif cmap.type == 1:        # Calculating contacts from all-atom structure and maping to CG structure
            if group == "prot":
                aa_data_res,aa_data_xyz = self.allatpdb.prot.res.copy(),self.allatpdb.prot.xyz.copy()
            elif group == "nucl": 
                aa_data_res,aa_data_xyz = self.allatpdb.nucl.res.copy(),self.allatpdb.nucl.xyz.copy()
            elif group in ("all","inter"):
                aa_data_res = list(self.allatpdb.nucl.res.copy())+list(self.allatpdb.prot.res.copy())
                aa_data_xyz = np.float_(list(self.allatpdb.nucl.xyz.copy())+list(self.allatpdb.prot.xyz.copy()))
            
            atomgroup,mol_id = [],[]
            for r in aa_data_res:
                if len(r[2])==3:
                    mol_id.append(1)
                    if len(self.CA_atn) != 0:
                        if len(self.CB_atn) == 0:   #0 for CA
                            atomgroup.append(tuple(list(r[:2])+[0]))
                        else:                       #1 for CB
                            atomgroup.append(tuple(list(r[:2])+[int(r[-1] not in ("N","C","CA","O"))]))
                if len(r[2])<=2:
                    mol_id.append(0)
                    if len(self.P_atn) != 0:
                        if len(self.B_atn) == 0:    #2 or P
                            atomgroup.append(tuple(list(r[:2])+[2]))
                        else:                       # 3 for S #5 for B
                            atomgroup.append(tuple(list(r[:2])+[3+2*int("P" in r[-1])-1*int("'" in r[-1])]))

            faa = open(tagforfile+".AAcont","w+")
            fcg = open(tagforfile+".CGcont","w+")
            cid,rnum,bb_sc = np.transpose(np.array(atomgroup))
            del (atomgroup)

            aa2cg = {0:self.CA_atn,1:self.CB_atn,\
                     5:self.P_atn,2:self.S_atn,3:self.B_atn}
            
            cutoff = cmap.cutoff*cmap.scale
            resgap = 4 
            contacts_dict = dict()
            
            mol_id = np.int_(mol_id)
            str_cid  = np.array([["nucl_","prot_"][mol_id[x]]+str(cid[x]+1) for x in range(mol_id.shape[0])])

            if group != "inter":  #loop over all vs all atom pairs
                bb_sc0,mol_id0,cid0,rnum0,aa_data_xyz0,str_cid0=bb_sc,mol_id,cid,rnum,aa_data_xyz,str_cid
            else:                 #loop over inter atom pairs
                p0,p1=np.where(mol_id==0)[0],np.where(mol_id==1)[0]
                bb_sc0,mol_id0,cid0,rnum0,aa_data_xyz0,str_cid0=bb_sc[p0],mol_id[p0],cid[p0],rnum[p0],aa_data_xyz[p0],str_cid[p0]
                bb_sc,mol_id,cid,rnum,aa_data_xyz,str_cid=bb_sc[p1],mol_id[p1],cid[p1],rnum[p1],aa_data_xyz[p1],str_cid[p1]
            
            print ("> Determining contacts for %d*%d atom pairs using %.2f A cutoff and %.2f scaling-factor"%(aa_data_xyz0.shape[0],aa_data_xyz.shape[0],cmap.cutoff,cmap.scale))
            for i in trange(aa_data_xyz0.shape[0]):
                #resgap = 4:CA-CA, 3:CA-CB, 3:CB-CB, 
                gap=resgap-np.int_(bb_sc+bb_sc0[i]>0) #aa2cg bbsc CA:0,CB:1
                #resgap 1: P/B/S-P/S/B
                gap=gap+(1-gap)*int(bb_sc0[i]>=2) #aa2cg bbsc P:5,B:3,S:2

                if group in ("prot","nucl"): #calculate for intra molecule
                    calculate = np.int_(mol_id==mol_id0[i])*\
                                np.int_( (np.int_(rnum-gap>=rnum0[i])*np.int_(cid==cid0[i])\
                                         + np.int_(cid>cid0[i])) > 0 )
                elif group == "inter":      #calculate for inter molecule
                    if mol_id0[i] == 1: continue
                    calculate = np.int_(mol_id>mol_id0[i])
                elif group == "all":        #calculate all
                    calculate = np.int_( np.int_(mol_id!=mol_id0[i] + \
                                         np.int_(rnum-gap>=rnum0[i])*np.int_(cid==cid0[i])\
                                         + np.int_(cid>cid0[i])) > 0)
                    
                contact=np.where(np.int_(np.sum((aa_data_xyz-aa_data_xyz0[i])**2,1)**0.5<=cutoff)*calculate==1)[0]
                for x in contact:
                    faa.write("%s %d %s %d\n"%(str_cid0[i],i+1,str_cid[x],x+1))
                    cg_a1 = aa2cg[bb_sc0[i]][cid0[i]][rnum0[i]]
                    cg_a2 = aa2cg[bb_sc[x]][cid[x]][rnum[x]]
                    set = (str_cid0[i],str_cid[x]),(cg_a1,cg_a2)
                    if set not in contacts_dict: contacts_dict[set] = 0
                    contacts_dict[set] += 1
            contacts_dict = {y:(x,contacts_dict[(x,y)]) for x,y in contacts_dict}
            pairs = list(contacts_dict.keys()); pairs.sort()
            weights = np.float_([contacts_dict[x][1] for x in pairs])
            weights = weights*(weights.shape[0]/np.sum(weights))
            if not cmap.W: weights = np.ones(weights.shape)
            #cid = np.int_([contacts_dict[x][0] for x in pairs])
            chains = [contacts_dict[x][0] for x in pairs]
            pairs = np.int_(pairs)
            if group!="inter": distances = self.__distances__(pairs=pairs)
            elif group=="inter": distances = self.__distances__(pairs=pairs,xyz0=self.cgpdb_n.xyz,xyz1=self.cgpdb_p.xyz)
            for x in range(pairs.shape[0]):
                c,a = chains[x],pairs[x]+1
                w,d = weights[x],distances[x]
                fcg.write("%s %d %s %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
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
                    cid += [("prot_"+str(c1+1),"prot_"+str(c1+1)) for x in range(len(pairs)-len(cid))]
                    for c2 in self.CA_atn:
                        if c2>c1: 
                            pairs += [(self.CA_atn[c1][x],self.CA_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c2]]
                            if len(self.CB_atn)!=0: 
                                pairs += [(self.CA_atn[c1][x],self.CB_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CA_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CB_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c2]]
                            cid += [("prot_"+str(c1+1),"prot_"+str(c2+1)) for x in range(len(pairs)-len(cid))]
            if len(self.P_atn)!=0:
                ppsep,ressep=1,1
                if len(self.S_atn)==0:
                    for c1 in self.P_atn:
                        pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c1] if y-x>=ppsep]
                        cid += [("nucl_"+str(c1+1),"nucl_"+str(c1+1)) for x in range(len(pairs)-len(cid))]
                        for c2 in self.P_atn: 
                            if c2>c1:
                                pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c2]]
                                cid += [("nucl_"+str(c1+1),"nucl_"+str(c2+1)) for x in range(len(pairs)-len(cid))]
                else:
                    assert len(self.B_atn)!=0
                    for c1 in self.S_atn:
                        all_atn_c1 = list(self.P_atn[c1].items())+list(self.S_atn[c1].items())+list(self.B_atn[c1].items())
                        pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c1 if ry-rx>=ressep]
                        cid += [("nucl_"+str(c1+1),"nucl_"+str(c1+1)) for x in range(len(pairs)-len(cid))]
                        for c2 in self.S_atn:
                            if c2>c1:
                                all_atn_c2 = list(self.P_atn[c2].items())+list(self.S_atn[c2].items())+list(self.B_atn[c2].items())
                                pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c2]
                                cid += [("nucl_"+str(c1+1),"nucl_"+str(c2+1)) for x in range(len(pairs)-len(cid))]

            pairs = np.int_(pairs)
            cid = np.array(cid)
            cutoff = 0.1*cmap.cutoff
            distances = self.__distances__(pairs)
            contacts = np.where(np.int_(distances<=cutoff))[0]
            pairs = pairs[contacts]
            chains = cid[contacts]
            distances = distances[contacts]
            check_dist = self.__distances__(pairs)
            for x in range(pairs.shape[0]): 
                assert check_dist[x] == distances[x] and distances[x] < cutoff
            weights = np.ones(pairs.shape[0])
            with  open(tagforfile+".CGcont","w+") as fcg:
                for x in range(pairs.shape[0]):
                    c,a = chains[x],pairs[x]+1
                    w,d = weights[x],distances[x]
                    fcg.write("%s %d %s %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
        
        self.contacts.append((pairs,chains,distances,weights))
        return

class MergeTop:

    def __init__(self,proc_data,Nprot,Nnucl,topfile,opt,excl_volume,excl_rule,fconst,cmap):
        self.data = proc_data
        self.nucl_cmap = cmap["nucl"]
        self.prot_cmap = cmap["prot"]
        self.inter_cmap= cmap["inter"]
        self.opt = opt
        self.excl_volume = excl_volume
        self.excl_rule = excl_rule
        self.fconst = fconst
        self.__merge__(Nprot=Nprot,Nnucl=Nnucl,topfile=topfile,opt=opt,excl_volume=excl_volume)

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

    def __writeInterPairs__(self,fsec,nmol,atoms_in_mol,tag,atnum_offset,sym):
        print ("> Determining Inter pairs")
        cmap = self.inter_cmap
        data=self.data
        inp=self.data.contacts
        if not sym: assert nmol[0]==1 and nmol[1]==1, "Multplte units not supported if not symmeterized"
        prev_at_count = [0,nmol[0]*atoms_in_mol[0]]


        for x0 in range(nmol[0]):
            fsec.write(tag[0]+self.nPlaces(n=3,count2str=x0+1)+"|") 
            for x1 in range(nmol[1]):
                fsec.write(tag[1]+self.nPlaces(n=3,count2str=x1+1)+"\n")    
                for pairs,chains,dist,eps in data.contacts:
                    I,J = np.transpose(pairs)
                    func = 1
                    if cmap.func==1: values = 2*eps*(dist**6.0),eps*(dist**12.0)
                    elif cmap.func==2: values = 6*eps*(dist**10.0),5*eps*(dist**12.0)
                    elif cmap.func==3: assert cmap.func!=3,"Error 18-12 not supported yet"
                    elif cmap.func in (5,6):
                        func,sd = 6,0.05
                        I = np.float_([self.excl_volume[self.atomtypes[x]] for x in I])
                        J = np.float_([self.excl_volume[self.atomtypes[x]] for x in J])
                        if self.excl_rule == 1: c12 = ((I**12.0)*(J**12.0))**0.5
                        elif self.excl_rule == 2: c12 = ((I+J)/2.0)**12.0
                        values = eps,dist,sd,c12
                    I = 1 + I + atnum_offset[0] + prev_at_count[0] + x0*atoms_in_mol[0]
                    J = 1 + J + atnum_offset[1] + prev_at_count[1] + x1*atoms_in_mol[1]
                    values = np.transpose(values)                    
                    for x in range(pairs.shape[0]): 
                        fsec.write(" %5d %5d %5d"%(I[x],J[x],func))
                        fsec.write(len(values[x])*" %e"%tuple(values[x]))
                        fsec.write("\n")
        return

    def __writeNonbondParams__(self,fsec):
        print ("> Writing user given custom nonbond_params:",self.opt.interface)
        
        eps,sig = self.data.Interactions(interface=self.opt.interface)
           
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
            cmap_func=self.inter_cmap.func
            fsec.write("; Custom Protein-RNA/DNA interactions\n")
            if self.nucl_cmap.func in (5,6):
                fsec.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            for p in eps:
                if p[0].startswith(("CA","CB")): continue
                if not p[1].startswith(("CA","CB")): continue
                if p[0] not in self.excl_volume: continue
                if p[1] not in self.excl_volume: continue
                p=list(p); p.sort(); p=tuple(p)
                if p in pairs: continue
                pairs.append(p)
                if p not in sig: cmap_func=-1
                else: sig[p]=sig[p][0] #to be changed for Gaussian
                func=1
                if cmap_func ==-1:
                    if self.excl_rule==1: c12 = eps[p]*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif self.excl_rule==2: C12 = eps[p]*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = 0.0,C12
                elif cmap_func==1: values = 2*eps[p]*((sig[p])**6),1*eps[p]*((sig[p])**12)
                elif cmap_func==2: values = 6*eps[p]*((sig[p])**10),5*eps[p]*((sig[p])**12)
                elif cmap_func in (5,6):
                    func,sd = 6,0.05
                    if self.excl_rule==1: c12 = (((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif self.excl_rule==2: C12 = ((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = eps[p],sig[p],sd,C12
                fsec.write(" %5s %5s\t%d\t"%(p[0],p[1],func))
                fsec.write(len(values)*" %e"%tuple(values))
                fsec.write("\n")
        return
                            
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
        if Nprot>0 and Nnucl>0:
            if self.inter_cmap.type>=0: 
                if not opt.control_run:
                    assert self.inter_cmap.type == 0, \
                        "Error, calculating inter Protein-RNA/DNA contacts only supported for --control runs. Provide custom file --cmap_i" 
                self.data.Pairs(cmap=self.inter_cmap,group="inter")
            
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
                    if header in ["pairs","exclusions"]:
                        if opt.intra_symmetrize: self.__writeSymPaIrs__(fsec=fout,inp=prot_data,nmol=Nprot, 
                                prev_at_count=Nnucl*natoms_nucl,atoms_in_mol=natoms_prot,tag=";Protein_",atnum_offset=off1)
                        if len(self.data.contacts)>0:
                            self.__writeInterPairs__(fsec=fout,nmol=[Nnucl,Nprot],atoms_in_mol=[natoms_nucl,natoms_prot],\
                                                     tag=[";Protein_","RNA/DNA_"],atnum_offset=[off0,off1],sym=opt.inter_symmetrize)
                else:
                    if Nnucl>0: status=[fout.write(i+"\n") for i in nucl_data if i.strip() != ""]
                    elif Nprot>0: status=[fout.write(i+"\n") for i in prot_data if i.strip() != ""]

class OpenSMOGXML:
    def __init__(self,xmlfile) -> None:
        self.fxml=open(xmlfile,"w+")
        self.fxml.write('<OpenSMOGforces>\n')
        self.nb_count,self.pairs_count=0,0

    def write_nonbond_xml(self,pairs=[],expression='Krep*((C/r)^12)',params={}):
        self.fxml.write(' <nonbond>\n')
        self.fxml.write('  <nonbond_bytype>\n')
        self.fxml.write('   <expression expr="%s"/>\n'%expression)
        for p in params: self.fxml.write('   <parameter>%s</parameter>\n'%p)
        for x in range(len(pairs)):
            self.fxml.write('   <nonbond_param type1="%s" type2="%s"'%tuple(pairs[x]))
            for p in params: self.fxml.write(' %s="%e"'%(p,params[p][x]))
            self.fxml.write('/>\n')
        self.fxml.write('  </nonbond_bytype>\n')
        self.fxml.write(' </nonbond>\n') 
        self.nb_count+=1
        return

    def write_pairs_xml(self,pairs=[],params={},name="contacts_LJ-10-12",\
                            expression="eps*( 5*((sig/r)^12) - 6*((sig/r)^10) )"):
        if self.pairs_count==0: self.fxml.write(' <contacts>\n')
        self.fxml.write('  <contacts_type name="%s">\n'%name)
        self.fxml.write('   <expression expr="%s"/>\n'%expression)
        for p in params: self.fxml.write('   <parameter>%s</parameter>\n'%p)
        I,J = 1+np.transpose(pairs)
        for x in range(pairs.shape[0]): 
            self.fxml.write('   <interaction i="%d" j="%d"'%(I[x],J[x]))
            for p in params: self.fxml.write(' %s="%e"'%(p,params[p][x]))
            self.fxml.write('/>\n')
        self.fxml.write('  </contacts_type>\n')
        self.pairs_count+=1
        return

    def __del__(self):
        if self.pairs_count>0:self.fxml.write(' </contacts>\n')
        self.fxml.write('</OpenSMOGforces>\n')
        self.fxml.close()

class Topology:
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
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
                seq_list = seq.split()
                assert len(seq_list) == len(data.S_atn)
                for c in data.S_atn:
                    assert c in data.B_atn
                    seq = "5"+seq_list[c]+"3"
                    for i in range(1,len(seq)-1):
                        if self.opt.codon_pairs: codon = seq[i-1].lower()+seq[i]+seq[i+1].lower()
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

        eps,sig = data.Interactions(nonbond=self.opt.nonbond)
        pairs,repul_C12 = [],[]
        if len(data.CA_atn) > 0:
            cmap_func=self.cmap["prot"].func
            if excl_rule == 2 and type == 2:
                for x in self.excl_volume:
                    if x.startswith(("CA","CB")):
                        for y in self.excl_volume:
                            if y.startswith(("CA","CB")):
                                C10,C12 = 0.0, self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                                p = [x,y]; p.sort(); p=tuple(p)
                                if p not in pairs:
                                    if p in eps: continue
                                    pairs.append(p)
                                    repul_C12.append(C12) 
                                    if self.opt.opensmog: continue
                                    fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))                            
        if len(data.P_atn) > 0:
            cmap_func=self.cmap["nucl"].func
            if excl_rule == 2 and type in (3,5):
                for x in self.excl_volume:
                    if x.startswith(("P","S","B")):
                        for y in self.excl_volume:
                            if y.startswith(("P","S","B")):
                                C10,C12 = 0.0, self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                                p = [x,y]; p.sort(); p=tuple(p)
                                if p not in pairs:
                                    if p in eps: continue
                                    pairs.append(p)
                                    repul_C12.append(C12) 
                                    if self.opt.opensmog: continue
                                    fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C10,C12))
        if len(eps)>0:
            fout.write("; Custom Nnobond interactions\n")
            if cmap_func in (5,6):
                fout.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            for p in eps:
                if p[0] not in self.excl_volume: continue
                if p[1] not in self.excl_volume: continue
                p=list(p); p.sort(); p=tuple(p)
                if p in pairs: continue
                pairs.append(p)
                if p not in sig: cmap_func=-1
                else: sig[p]=sig[p][0] #to be changed for Gaussian
                func=1
                if cmap_func ==-1:
                    if excl_rule==1: c12 = eps[p]*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif excl_rule==2: C12 = eps[p]*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = 0.0,C12
                elif cmap_func==1: values = 2*eps[p]*((sig[p])**6),1*eps[p]*((sig[p])**12)
                elif cmap_func==2: values = 6*eps[p]*((sig[p])**10),5*eps[p]*((sig[p])**12)
                elif cmap_func in (5,6):
                    func,sd = 6,0.05
                    if excl_rule==1: c12 = (((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif excl_rule==2: C12 = ((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = eps[p],sig[p],sd,C12
                repul_C12.append(values[-1])
                if self.opt.opensmog: continue
                fout.write(" %5s %5s\t%d\t"%(p[0],p[1],func))
                fout.write(len(values)*" %e"%tuple(values))
                fout.write("\n")

        if self.opt.opensmog and len(repul_C12)!=0:
            if len(eps)==0: 
                expression,params="C12(type1,type2)/(r^12)",{"C12":repul_C12}
            else:
                params={"eps_att":[],"sig":[],"C12":repul_C12}
                for p in pairs: 
                    if p in eps: params["eps_att"].append(eps[p])
                    else: params["eps_att"].append(1)
                    if p in sig: params["sig"].append(sig[p])
                    else: params["sig"].appen(0)
                assert len(params["eps_att"])==len(params["C12"])
                if cmap_func==1: expression="C12(type1,type2)/(r^12) - 2*eps_att(type1,type2)*(sig(type1,type2)/r)^6"
                elif cmap_func==2: expression="C12(type1,type2)/(r^12) - 6*eps_att(type1,type2)*(sig(type1,type2)/r)^10"
                elif cmap_func in (5,6): 
                    expression="eps_att(type1,type2)*( (1+C12(type1,type2)/(r^12))*(1-exp(((r-sig(type1,type2))^2)/(2*(sd^2))) - 1); sd=%e"%sd
            if len(data.CA_atn)!=0: self.prot_xmlfile.write_nonbond_xml(pairs=pairs,\
                                        expression=expression,params=params)
            if len(data.P_atn)!=0: self.nucl_xmlfile.write_nonbond_xml(pairs=pairs,\
                                        expression=expression,params=params)
        return 0

    def __write_moleculetype__(self,fout):
        print (">> Writing moleculetype section")
        fout.write("\n%s\n"%("[ moleculetype ]"))
        fout.write("%s\n"%("; name            nrexcl"))
        fout.write("%s\n"%("  Macromolecule   3"))

    def __write_atoms__(self,fout,type,cgfile,seq,inc_charge):
        print (">> Writing atoms section")
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = dict()
        if inc_charge: 
            Q.update({x:1 for x in ["CBK","CBR","CBH"]})
            Q.update({x:-1 for x in ["CBD","CBE","P"]})

        prev_resnum,seqcount,rescount="",0,0
        if ".nucl." in cgfile: seq=["5%s3"%x for x in seq.split()]
        elif ".prot." in cgfile: seq=["_%s_"%x for x in seq.split()]
        self.atomtypes=[]
        with open(cgfile) as fin:
            for line in fin:
                if line.startswith("ATOM"):
                    atnum=hy36decode(5,line[6:11])
                    atname=line[12:16].strip()
                    resname=line[17:20].strip()
                    resnum=hy36decode(4,line[22:26])
                    atype=atname
                    if resnum !=prev_resnum: prev_resnum,rescount=resnum,1+rescount
                    if len(resname)<=2 and atype!="P":
                        if self.opt.codon_pairs: 
                            codon=atype[1]+seq[seqcount][rescount-1].lower()\
                                 +seq[seqcount][rescount]+seq[seqcount][rescount+1].lower()
                        else: codon=atype[1]+seq[seqcount][rescount]
                    if atype=="CB": atype+=seq[seqcount][rescount]
                    elif atype.endswith("'"): atype="S"+codon
                    elif atype.startswith("N"): atype="B"+codon
                    if atype not in Q: Q[atype] = 0
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
                elif line.startswith("TER"): seqcount,rescount=1+seqcount,0
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
        #data.Pairs(cmap=cmap,aa_data=self.allatomdata.prot)
        data.Pairs(cmap=cmap,group="prot")
        if cmap.custom_pairs:
            epsmat,sigmat=data.Interactions(pairs=cmap.custom_pairs)
            CA_atn = {data.CA_atn[c][r]:self.atomtypes[data.CA_atn[c][r]] for c in data.CA_atn for r in data.CA_atn[c]}
            CB_atn = {data.CB_atn[c][r]:self.atomtypes[data.CB_atn[c][r]] for c in data.CB_atn for r in data.CB_atn[c]}
            all_atn = CA_atn.copy()
            all_atn.update(CB_atn.copy())
            for index in range(len(data.contacts)):
                pairs,chains,dist,eps = data.contacts[index]
                I,J = np.transpose(pairs)
                interaction_type = \
                        np.int_([x in CB_atn for x in I])+ \
                        np.int_([x in CB_atn for x in J])
                epsmat.update({(all_atn[I[x]],all_atn[J[x]]):1.0 })
                for x in range(I.shape[0]):
                    if (all_atn[I[x]],all_atn[J[x]]) in epsmat:
                        eps[x] *= epsmat[(all_atn[I[x]],all_atn[J[x]])]
                    if (all_atn[I[x]],all_atn[J[x]]) in sigmat:
                        dist[x] = sigmat[(all_atn[I[x]],all_atn[J[x]])]
                data.contacts[index] = pairs,chains,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        if cmap.func==1:
            print ("> Using LJ C6-C12 for contacts")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
            func = 1
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                if self.opt.opensmog:
                    self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-06-12"%c,\
                            expression="eps*( 1*((r0/r)^12) - 2*((r0/r)^6) )")
                    continue
                I,J = 1+np.transpose(pairs)
                c06 = 2*eps*(dist**6.0)
                c12 = eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))
        elif cmap.func==2:
            print ("> Using LJ C10-C12 for contacts. Note: Require Table file(s)")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
            func = 1
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                if self.opt.opensmog:
                    self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-10-12"%c,\
                            expression="eps*( 5*((r0/r)^12) - 6*((r0/r)^10) )")
                    continue
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
            for pairs,chains,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
        elif cmap.func in (5,6):
            fout.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            func = 6
            sd = 0.05
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                I,J = np.transpose(pairs)
                I = np.float_([self.excl_volume[self.atomtypes[x]] for x in I])
                J = np.float_([self.excl_volume[self.atomtypes[x]] for x in J])
                if excl_rule == 1: c12 = ((I**12.0)*(J**12.0))**0.5
                elif excl_rule == 2: c12 = ((I+J)/2.0)**12.0
                if self.opt.opensmog:
                    self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps,"C12":c12},\
                            name="contacts%d_Gaussian-12"%c,\
                            expression="eps*( (1+C12/(r^12))*(1-exp(((r-r0)^2)/(2*(sd^2))) - 1); sd=%e"%sd)
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
        #data.Pairs(cmap=cmap,aa_data=self.allatomdata.nucl)
        data.Pairs(cmap=cmap,group="nucl")

        fout.write("\n%s\n"%("[ pairs ]"))
        if cmap.func==1:
            print ("> Using LJ C6-C12 for contacts")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
            func = 1
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                if self.opt.opensmog:
                    self.nucl_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-06-12"%c,\
                            expression="eps*( 1*((r0/r)^12) - 2*((r0/r)^6) )")
                    continue
                I,J = 1+np.transpose(pairs)
                c06 = 2*eps*(dist**6.0)
                c12 = eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))
        elif cmap.func==2:
            print ("> Using LJ C10-C12 for contacts. Note: Require Table file(s)")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
            func = 1
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                if self.opt.opensmog:
                    self.nucl_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-10-12"%c,\
                            expression="eps*( 5*((r0/r)^12) - 6*((r0/r)^10) )")
                    continue
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
            for pairs,chains,dist,eps in data.contacts:
                I,J = 1+np.transpose(pairs)
        elif cmap.func in (5,6):
            fout.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            func = 6
            sd = 0.05
            for c in range(len(data.contacts)):
                pairs,chains,dist,eps=data.contacts[c]
                I,J = np.transpose(pairs)
                I = np.float_([self.excl_volume[self.atomtypes[x]] for x in I])
                J = np.float_([self.excl_volume[self.atomtypes[x]] for x in J])
                if excl_rule == 1: c12 = ((I**12.0)*(J**12.0))**0.5
                elif excl_rule == 2: c12 = ((I+J)/2.0)**12.0
                if self.opt.opensmog:
                    self.nucl_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps,"C12":c12},\
                            name="contacts%d_Gaussian-12"%c,\
                            expression="eps*( (1+C12/(r^12))*(1-exp(((r-r0)^2)/(2*(sd^2))) - 1); sd=%e"%sd)
                I,J = 1+np.transpose(pairs)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(I[x],J[x],func,eps[x],dist[x],sd,c12[x]))
        return 

    def __write_exclusions__(self,fout,data):
        print (">> Writing exclusions section")
        fout.write("\n%s\n"%("[ exclusions ]"))
        fout.write("; %5s %5s\n"%("i","j"))
        for pairs,chains,dist,eps in data.contacts:
            I,J = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d\n"%(I[x],J[x]))
        return

    def write_topfile(self,outtop,excl,charge,bond_function,CBchiral,rad):
        cgpdb = PDB_IO()
        Data = Calculate(aa_pdb=self.allatomdata)
        if len(self.allatomdata.prot.lines) > 0 and self.CGlevel["prot"] in (1,2):
            if self.CGlevel["prot"]==1: cgpdb.loadfile(infile=self.allatomdata.prot.bb_file,refine=False)
            elif self.CGlevel["prot"]==2: cgpdb.loadfile(infile=self.allatomdata.prot.sc_file,refine=False)
            prot_topfile = "prot_"+outtop
            if self.opt.opensmog: self.prot_xmlfile=OpenSMOGXML(xmlfile="prot_"+self.opt.xmlfile)
            with open(prot_topfile,"w+") as ftop:
                print (">>> writing Protein GROMACS toptology", prot_topfile)
                proc_data_p = Calculate(aa_pdb=self.allatomdata)
                proc_data_p.processData(data=cgpdb.prot)
                self.__write_header__(fout=ftop,combrule=excl)
                self.__write_atomtypes__(fout=ftop,data=proc_data_p,type=self.CGlevel["prot"],seq=cgpdb.prot.seq,rad=rad)
                self.__write_nonbond_params__(fout=ftop,data=proc_data_p,type=self.CGlevel["prot"],excl_rule=excl)
                self.__write_moleculetype__(fout=ftop)
                self.__write_atoms__(fout=ftop,type=self.CGlevel["prot"],cgfile=cgpdb.pdbfile,seq=cgpdb.prot.seq,inc_charge=(charge.CA or charge.CB)*(not self.opt.opensmog))
                self.__write_protein_pairs__(fout=ftop, data=proc_data_p,excl_rule=excl,charge=charge)
                self.__write_protein_bonds__(fout=ftop, data=proc_data_p,func=bond_function)
                self.__write_protein_angles__(fout=ftop, data=proc_data_p)
                self.__write_protein_dihedrals__(fout=ftop, data=proc_data_p,chiral=CBchiral)
                self.__write_exclusions__(fout=ftop,data=proc_data_p)
                self.__write_footer__(fout=ftop)
                self.proc_data_p = proc_data_p
            if self.opt.opensmog: del self.prot_xmlfile
        if len(self.allatomdata.nucl.lines) > 0 and self.CGlevel["nucl"] in (1,3,5):
            if self.CGlevel["nucl"]==1: cgpdb.loadfile(infile=self.allatomdata.nucl.bb_file,refine=False)
            elif self.CGlevel["nucl"] in (3,5): cgpdb.loadfile(infile=self.allatomdata.nucl.sc_file,refine=False)
            nucl_topfile = "nucl_"+outtop
            if self.opt.opensmog: self.nucl_xmlfile=OpenSMOGXML(xmlfile="nucl_"+self.opt.xmlfile)
            with open(nucl_topfile,"w+") as ftop:
                print (">>> writing RNA/DNA GROMACS toptology", nucl_topfile)
                proc_data_n = Calculate(aa_pdb=self.allatomdata)
                proc_data_n.processData(data=cgpdb.nucl)
                self.__write_header__(fout=ftop,combrule=excl)
                self.__write_atomtypes__(fout=ftop,type=self.CGlevel["nucl"],data=proc_data_n,seq=cgpdb.nucl.seq,rad=rad)
                self.__write_nonbond_params__(fout=ftop,data=proc_data_n,type=self.CGlevel["nucl"],excl_rule=excl)
                self.__write_moleculetype__(fout=ftop)
                self.__write_atoms__(fout=ftop,type=self.CGlevel["nucl"],cgfile=cgpdb.pdbfile,seq=cgpdb.nucl.seq,inc_charge=charge.P*(not self.opt.opensmog))
                self.__write_nucleicacid_pairs__(fout=ftop, data=proc_data_n,excl_rule=excl,charge=charge)
                self.__write_nucleicacid_bonds__(fout=ftop, data=proc_data_n,func=bond_function)
                self.__write_nucleicacid_angles__(fout=ftop, data=proc_data_n)
                self.__write_nucleicacid_dihedrals__(fout=ftop, data=proc_data_n,chiral=CBchiral)
                self.__write_exclusions__(fout=ftop,data=proc_data_n)
                self.__write_footer__(fout=ftop)
                self.prot_data_n = proc_data_n
            if self.opt.opensmog: del nucl_xmlfile

        Nmol = self.Nmol
        #if len(self.allatomdata.nucl.lines) > 0 and self.CGlevel["nucl"] in (1,3,5):
        if Nmol["prot"]+Nmol["nucl"] > 1:
            if Nmol["prot"]>0 and len(proc_data_p.CA_atn) != 0:
                Data.CA_atn,Data.CB_atn = proc_data_p.CA_atn,proc_data_p.CB_atn
                Data.cgpdb_p = proc_data_p.cgpdb
                assert len(proc_data_p.P_atn) == 0
            if Nmol["nucl"]>0 and len(proc_data_n.P_atn) != 0: 
                Data.P_atn,Data.S_atn,Data.B_atn = proc_data_n.P_atn,proc_data_n.S_atn,proc_data_n.B_atn
                Data.cgpdb_n = proc_data_n.cgpdb
                assert len(proc_data_n.CA_atn) == 0
            if len(self.allatomdata.prot.lines) > 0 and self.CGlevel["prot"] in (1,2):
                merge=MergeTop(proc_data=Data,Nprot=Nmol["prot"],Nnucl=Nmol["nucl"],topfile=outtop,opt=self.opt,excl_volume=self.excl_volume,excl_rule=excl,fconst=self.fconst,cmap=self.cmap)

        if self.opt.opensmog: return #don't write table
        table = Tables()
        if self.cmap["prot"].func == 2 or self.cmap["nucl"].func == 2 or self.cmap["inter"].func == 2 :
            table.__write_pair_table__(elec=charge,ljtype=2)
        if self.cmap["prot"].func == 1 or self.cmap["nucl"].func == 1 or self.cmap["inter"].func == 1:
            if charge.debye: table.__write_pair_table__(elec=charge,ljtype=1)
        
        return 

class Clementi2000(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

class Pal2019(Topology):
    def __write_nucleicacid_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing pairs section")
        cmap = self.cmap["nucl"]
        fout.write("\n%s\n"%("[ pairs ]"))
        assert cmap.func==2
        print ("> Using LJ C10-C12 for Stackubg. Note: Require Table file(s)")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
        func = 1
        epsmat,stack = data.Interactions(pairs=True)

        B_atn = {v:self.atomtypes[v] for k,v_list in self.allatomdata.nucl.B_atn.items() for v in v_list}
        for p in stack:stack[p]=stack[p][0] #multiple distances not supported
        for c in data.B_atn:
            resnum = list(data.B_atn[c].keys())
            resnum.sort()
            pairs = np.int_([(data.B_atn[c][x],data.B_atn[c][x+1]) for x in resnum if x+1 in data.B_atn[c]])
            I,J = np.transpose(pairs)
            eps = np.float_([epsmat[(B_atn[I[x]],B_atn[J[x]])] for x in range(I.shape[0])])
            dist = np.float_([stack[(B_atn[I[x]],B_atn[J[x]])] for x in range(I.shape[0])])
            chains = [tuple(["nucl_"+str(c+1)]*2) for x in range(I.shape[0])]
            data.contacts.append((pairs,chains,dist,eps))
            if self.opt.opensmog:
                self.nucl_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="base-stacking%d_LJ-10-12"%c,\
                            expression="eps*( 5*((r0/r)^12) - 6*((r0/r)^10) )")
                continue
            I,J = 1+np.transpose(pairs)
            c10 = 6*eps*(dist**10.0)
            c12 = 5*eps*(dist**12.0)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c10[x],c12[x]))
        return 

class Reddy2017(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        #self.__check_H_atom__()
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
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
        pairs,excl_rad1,excl_rad2 = [],[],[]
        for x in self.excl_volume:
            if x.startswith(("CA","CB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        C06 = -1*self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**6
                        C12 = 0.0
                        p = [x,y]; p.sort(); p=tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            excl_rad1.append(self.excl_volume[p[0]])
                            excl_rad2.append(self.excl_volume[p[1]])
                            if self.opt.opensmog: continue
                            fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C06,C12))
        
        if self.opt.opensmog: 
            self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"excl_r1":excl_rad1,"excl_r2":excl_rad2},\
                    expression='Krep*(sig/r)^6;sig=0.5*(excl_r1(type1,type2)+excl_r2(type1,type2));Krep=%e'%self.fconst.Kr_prot)
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

        # V = -(K/2)*R^2*ln(1-((r-r0)/R)^2)
        # V_1 = dV/dr = -K*0.5*R^2*(1/(1-((r-r0)/R)^2))*(-2*(r-r0)/R^2)
        #             = -K*0.5*(-2)(r-r0)/(1-((r-r0)/R)^2)
        #             = K*(R^2)(r-r0)/(R^2-(r-r0)^2)

        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "table_no.", "Kb"))
        data.Bonds()
        table_idx = dict()
        for c in range(len(data.bonds)):
            pairs,dist=data.bonds[c]
            if self.opt.opensmog:
                print (">Writing chain %d Bonds as OpenSMOG contacts"%c)
                self.prot_xmlfile.write_pairs_xml( pairs=pairs,params={"r0":dist},\
                            name="FENE_bonds%d_R=0.2"%c,\
                            expression="-(K/2)*(R^2)*log(1-((r-r0)/R)^2); R=%.2f; K=%e"%(R,K))
                data.contacts.append((pairs,c*np.ones(pairs.shape),dist,K*np.ones(dist.shape)))
                continue
            I,J = 1+np.transpose(pairs) 
            for i in range(pairs.shape[0]): 
                r0 = np.round(dist[i],3)
                if r0 not in table_idx: table_idx[r0]=len(table_idx)
                if r0-R>0:r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                else: r=0.001*np.int_(range(int(1000*(0+0.001)),int(1000*(r0+R-0.001))))
                V = -0.5*(R**2)*np.log(1-((r-r0)/R)**2)
                #V_1 = -0.5*(R**2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                V_1 = (R**2)*(r-r0)/(R**2-(r-r0)**2)
                Tables().__write_bond_table__(X=r,index=table_idx[r0],V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e; d=%.3f\n"%(I[i],J[i],func,table_idx[r0],K,r0))
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
        data.Pairs(cmap=cmap,group="prot")
        assert cmap.custom_pairs and cmap.type in (-1,0,2) and cmap.func==1
        scscmat,sigmat=data.Interactions(pairs=cmap.custom_pairs)
        assert len(sigmat)==0

        CA_atn = {data.CA_atn[c][r]:self.atomtypes[data.CA_atn[c][r]] for c in data.CA_atn for r in data.CA_atn[c]}
        CB_atn = {data.CB_atn[c][r]:self.atomtypes[data.CB_atn[c][r]] for c in data.CB_atn for r in data.CB_atn[c]}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())
        eps_bbbb = 0.5*self.fconst.caltoj
        eps_bbsc = 0.5*self.fconst.caltoj
        Kboltz = self.fconst.Kboltz #*self.fconst.caltoj/self.fconst.caltoj
        for index in range(len(data.contacts)):
            pairs,chains,dist,eps = data.contacts[index]
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
            data.contacts[index] = pairs,chains,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        print ("> Using LJ C6-C12 for contacts")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
        func = 1
        for c in range(len(data.contacts)):
            pairs,chains,dist,eps=data.contacts[c]
            if self.opt.opensmog:
                self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-06-12"%c,\
                            expression="eps*( 1*((r0/r)^12) - 2*((r0/r)^6) )")
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
            data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
            if self.opt.opensmog:
                self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==0)],\
                    params={"sig":sig[np.where(interaction_type==0)]},name="Local_backbone-backbone_rep%d"%c,\
                    expression="eps*((sig/r)^6);eps=%e"%eps_bbbb)
                self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==1)],\
                    params={"sig":sig[np.where(interaction_type==1)]},name="Local_backbone-sidechain_rep%d"%c,\
                    expression="eps*((0.8*sig/r)^6);eps=%e"%eps_bbsc)
                continue
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
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
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
        pairs, = [],
        eps = dict()
        eps[("CA","CA")] = self.eps_bbbb
        eps[("CB","CB")] = self.eps_bbsc
        eps[("CA","CB")] = self.eps_scsc


        epsmat,sigmat=data.Interactions(nonbond=True)
        assert len(epsmat)!=0 and len(sigmat)==0
        epsmat = {k:np.abs(0.7-epsmat[k]) for k in epsmat}
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
                        p = [x,y]; p.sort(); p = tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            if self.opt.opensmog: continue
                            fout.write(" %s %s\t1\t%e %e\n"%(x.ljust(5),y.ljust(5),C06,C12))
        
        if self.opt.opensmog:
            excl_rad1,excl_rad2=np.transpose([(self.excl_volume[x],self.excl_volume[y]) for x,y in pairs])
            epsmat=[epsmat[p] for p in pairs]
            eps=[eps[(x[:2],y[:2])] for x,y in pairs]
            self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"eps":eps,"f":epsmat,"r1":excl_rad1,"r2":excl_rad2},\
                                            expression='eps(type1,type2)*f(type1,type2)*((s/r)^12 - 2*(s/r)^6); s=0.5*(r1(type1,type2)+r2(type1,type2))')
        return 0  

    def __write_protein_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing SOP-SC pairs section")
        cmap = self.cmap["prot"]
        assert cmap.custom_pairs and cmap.type in (-1,0,2) and cmap.func==1

        CA_atn = {data.CA_atn[c][r]:self.atomtypes[data.CA_atn[c][r]] for c in data.CA_atn for r in data.CA_atn[c]}
        CB_atn = {data.CB_atn[c][r]:self.atomtypes[data.CB_atn[c][r]] for c in data.CB_atn for r in data.CB_atn[c]}
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
        data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
        if self.opt.opensmog:
            c=0
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==0)],\
                params={"sig":sig[np.where(interaction_type==0)]},name="Local_backbone-backbone_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_bbbb)
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==1)],\
                params={"sig":sig[np.where(interaction_type==1)]},name="Local_backbone-sidechain_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_bbsc)
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==2)],\
                params={"sig":sig[np.where(interaction_type==2)]},name="Local_sidechain-sidechain_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_scsc)
        else:
            I,K = I+1,K+1
            for x in range(len(pairs)): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))
        return 

class Baratam2024(Reddy2017):
    def __init__(self,allatomdata,idrdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.idrdata=idrdata
        self.ordered=self.allatomdata.prot
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
        self.opt = opt
        self.eps_idr_bbbb = 0.12*self.fconst.caltoj 
        self.eps_idr_bbsc = 0.24*self.fconst.caltoj 
        self.eps_idr_scsc = 0.18*self.fconst.caltoj 
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

    def __write_unfolded_cgpdb__(self,rad):
        print ("> Writing unfolded CG-PDB file")
        residues=set(self.idrdata.res+self.ordered.res)
        residues = [x for x in residues if "CA" in x]        
        residues.sort()
        chain,prev_rnum="",0
        outfasta="unfolded.fa"
        with open(outfasta,"w+") as fout:
            for cnum,rnum,rname,atname in residues:
                assert atname=="CA"
                if cnum!=chain or rnum-prev_rnum not in (0,1):
                    fout.write("\n\n>chain:%s:%s\n"%(cnum,rnum))
                fout.write(self.idrdata.amino_acid_dict[rname])
                chain = cnum
                prev_rnum=rnum
        residues.sort()
        #get unfolded CG-pdb
        unfolded = PDB_IO()
        unfolded.buildProtIDR(fasta=outfasta,rad=rad)
        self.unfolded_cgpdb=unfolded.prot
        #get unfolded CG-pdb processed data
        unfolded = Calculate(aa_pdb=self.allatomdata)
        unfolded.processData(data=self.unfolded_cgpdb)
        self.unfolded_data=unfolded
        return 0

    def __write_atomtypes__(self,fout,type,rad,seq,data):
        print (">> Writing atomtypes section")
        self.__write_unfolded_cgpdb__(rad=rad)
        #1:CA model or 2:CA+CB model
        fout.write('%s\n'%("[ atomtypes ]"))
        fout.write(6*"%s".ljust(5)%("; name","mass","charge","ptype","C6(or C10)","C12\n"))

        assert len(data.CA_atn)!=0 and type==2
        self.excl_volume["CA"] = 2*rad["CA"]
        self.excl_volume["iCA"] = self.excl_volume["CA"]
        C12 = self.fconst.Kr_prot*(2*rad["CA"])**12.0
        fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%("CA".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CA"))
        fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%("iCA".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CA"))
        for s in seq+self.idrdata.seq:
            bead = "CB"+s
            if bead in self.excl_volume or s == " ": continue
            C12 = self.fconst.Kr_prot*(2*rad[bead])**12.0
            self.excl_volume[bead] = 2*rad[bead]
            fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CB"))
            if s in self.idrdata.seq:
                self.excl_volume["i"+bead]=self.excl_volume[bead]
                fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%("i"+bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CB"))
        
        return 0

    def __write_nonbond_params__(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))
        fout.write(";C6 based repulsion term\n")
        assert type==2 and excl_rule == 2
        pairs,values = [],[]
        for x in self.excl_volume:
            if x.startswith(("CA","CB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        C06 = -1*self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**6
                        C12 = 0.0
                        p = [x,y]; p.sort(); p=tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            values.append((C06,C12))
                            if self.opt.opensmog: continue
                            fout.write(" %s %s\t1\t%e %e\n"%(p[0].ljust(5),p[1].ljust(5),C06,C12))
        


        fout.write(";IDR C6-C12 interactions\n")
        eps = dict()
        eps[("CA","CA")] = self.eps_idr_bbbb
        eps[("CB","CB")] = self.eps_idr_scsc
        eps[("CA","CB")] = self.eps_idr_bbsc
        eps[("CB","CA")] = self.eps_idr_bbsc
        

        epsmat,sigmat=data.Interactions(nonbond=True)
        assert len(epsmat)!=0 and len(sigmat)==0
        epsmat = {k:np.abs(0.7-epsmat[k]) for k in epsmat}
        epsmat.update({(k[1],k[0]):epsmat[k] for k in epsmat})
        epsmat.update({("CA",k[0]):1.0 for k in epsmat if "CA" not in k})
        epsmat.update({(k[0],"CA"):1.0 for k in epsmat if "CA" not in k})
        epsmat.update({("CA","CA"):1.0})

        for x in self.excl_volume:
            if x.startswith(("iCA","iCB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB","iCA","iCB")):
                        sig = (self.excl_volume[x]+self.excl_volume[y])/2.0
                        if y.startswith("C"): p = [x[1:3],y[:2]]; p.sort(); p = tuple(p)
                        elif y.startswith("iC"): p = [x[1:3],y[1:3]]; p.sort(); p = tuple(p)
                        q=(x.strip("i"),y.strip("i"))
                        C06 = 2*eps[p]*epsmat[q]*(sig)**6
                        C12 = 1*eps[p]*epsmat[q]*(sig)**12
                        p = [x,y]; p.sort(); p = tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            values.append((C06,C12))
                            if self.opt.opensmog: continue
                            fout.write(" %s %s\t1\t%e %e "%(x.ljust(5),y.ljust(5),C06,C12))
                            if y.startswith("i"): fout.write("; IDR-IDR\n")
                            else: fout.write("; OR-IDR\n")
        if self.opt.opensmog:
            C06,C12 = np.transpose(values)
            self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"C12":C12,"C6":C06},\
                        expression='C12(type1,type2)/(r^12) - C6(type1,type2)/(r^6)')
        return 0  

    def __write_atoms__(self,fout,type,cgfile,seq,inc_charge):
        print (">> Writing atoms section")
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = dict()
        if inc_charge: 
            Q.update({x:1 for x in ["CBK","CBR","CBH"]})
            Q.update({x:-1 for x in ["CBD","CBE","P"]})
        
        assert cgfile==self.ordered.sc_file
        cgfile=self.unfolded_cgpdb.pdbfile
        seq=self.unfolded_cgpdb.seq
        prev_resnum,seqcount,rescount="",0,0
        #if ".nucl." in cgfile: seq=["5%s3"%x for x in seq.split()]
        assert ".prot." in cgfile
        seq=["_%s_"%x for x in seq.split()]
        self.atomtypes=[]
        with open(cgfile) as fin:
            for line in fin:
                if line.startswith("ATOM"):
                    atnum=hy36decode(5,line[6:11])
                    atname=line[12:16].strip()
                    resname=line[17:20].strip()
                    resnum=hy36decode(4,line[22:26])
                    atinfo = (seqcount,resnum,resname,atname)
                    atype=atname
                    if resnum !=prev_resnum: prev_resnum,rescount=resnum,1+rescount
                    if atype=="CB": atype+=seq[seqcount][rescount]
                    if atype not in Q: Q[atype] = 0
                    if atinfo in self.idrdata.res:
                        Q["i"+atype]=Q[atype]
                        atype="i"+atype
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
                elif line.startswith("TER"): seqcount,rescount=1+seqcount,0
        return

    def __get_idr_bonds__(self,data):
        unfolded=self.unfolded_data
        new2old_atn = {unfolded.CA_atn[c][r]:data.CA_atn[c][r] \
                    for c in unfolded.CA_atn for r in unfolded.CA_atn[c] \
                            if c in data.CA_atn and r in data.CA_atn[c]}
        new2old_atn.update({unfolded.CB_atn[c][r]:data.CB_atn[c][r] \
                        for c in unfolded.CB_atn for r in unfolded.CB_atn[c] \
                        if c in data.CB_atn and r in data.CB_atn[c]})

        unfolded.Bonds()
        dist_dict = {tuple(pairs[y]):dist[y] \
                        for pairs,dist in data.bonds \
                        for y in range(dist.shape[0]) }

        for x in range(len(unfolded.bonds)):
            pairs,dist = unfolded.bonds[x]
            for y in range(dist.shape[0]):
                p=pairs[y]
                if p[0] in new2old_atn and p[1] in new2old_atn:
                    p = (new2old_atn[p[0]],new2old_atn[p[1]])
                    #print ("%.3f %.3f"%(dist[y],dist_dict[p]))
                    dist[y]=dist_dict[p]
            unfolded.bonds[x]=pairs,dist

        return unfolded

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
        adjusted_data=self.__get_idr_bonds__(data=data)
        data.bonds=adjusted_data.bonds

        table_idx = dict()
        for c in range(len(adjusted_data.bonds)):
            pairs,dist=adjusted_data.bonds[c]
            if self.opt.opensmog:
                print (">Writing chain %d Bonds as OpenSMOG contacts"%c)
                self.prot_xmlfile.write_pairs_xml( pairs=pairs,params={"r0":dist},\
                            name="FENE_bonds%d_R=0.2"%c,\
                            expression="-(K/2)*(R^2)*log(1-((r-r0)/R)^2); R=%.2f; K=%e"%(R,K))
                data.contacts.append((pairs,c*np.ones(pairs.shape),dist,K*np.ones(dist.shape)))
                continue
            I,J = 1+np.transpose(pairs) 
            for i in range(pairs.shape[0]): 
                r0 = np.round(dist[i],3)
                if r0 not in table_idx: table_idx[r0]=len(table_idx)
                if r0-R>0:
                    r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                else:
                    r=0.001*np.int_(range(int(1000*(0+0.001)),int(1000*(r0+R-0.001))))
                V = -0.5*(R**2)*np.log(1-((r-r0)/R)**2)
                #V_1 = -0.5*(R**2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                V_1 = (R**2)*(r-r0)/(R**2-(r-r0)**2)
                Tables().__write_bond_table__(X=r,index=table_idx[r0],V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e; d=%.3f\n"%(I[i],J[i],func,table_idx[r0],K,r0))
        return 

    def __write_protein_pairs__(self,fout,data,excl_rule,charge):
        print (">> Writing SOP-SC pairs section")
        cmap = self.cmap["prot"]
        data.Pairs(cmap=cmap,group="prot")
        assert cmap.custom_pairs and cmap.type in (-1,0,2) and cmap.func==1
        scscmat,sigmat=data.Interactions(pairs=cmap.custom_pairs)
        assert len(sigmat)==0

        CA_atn = {data.CA_atn[c][r]:self.atomtypes[data.CA_atn[c][r]] for c in data.CA_atn for r in data.CA_atn[c]}
        CB_atn = {data.CB_atn[c][r]:self.atomtypes[data.CB_atn[c][r]] for c in data.CB_atn for r in data.CB_atn[c]}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())
        eps_bbbb = 0.5*self.fconst.caltoj
        eps_bbsc = 0.5*self.fconst.caltoj
        Kboltz = self.fconst.Kboltz #*self.fconst.caltoj/self.fconst.caltoj
        
        old2new_atn = {data.CA_atn[c][r]:self.unfolded_data.CA_atn[c][r] \
            for c in self.unfolded_data.CA_atn for r in self.unfolded_data.CA_atn[c] \
            if c in data.CA_atn and r in data.CA_atn[c]}
        old2new_atn.update({data.CB_atn[c][r]:self.unfolded_data.CB_atn[c][r] \
            for c in self.unfolded_data.CB_atn for r in self.unfolded_data.CB_atn[c] \
            if c in data.CB_atn and r in data.CB_atn[c]})
       
        for index in range(len(data.contacts)):
            pairs,chains,dist,eps = data.contacts[index]
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
            pairs = np.int_([(old2new_atn[x],old2new_atn[y]) for x,y in pairs])
            data.contacts[index] = pairs,chains,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        print ("> Using LJ C6-C12 for contacts")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
        func = 1
        for c in range(len(data.contacts)):
            pairs,chains,dist,eps=data.contacts[c]
            if self.opt.opensmog:
                self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps},\
                            name="contacts%d_LJ-06-12"%c,\
                            expression="eps*( 1*((r0/r)^12) - 2*((r0/r)^6) )")
                continue
            I,J = 1+np.transpose(pairs)
            c06 = 2*eps*(dist**6.0)
            c12 = eps*(dist**12.0)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],J[x],func,c06[x],c12[x]))

        print ("> Using -C6 repulsion for local beads")
        fout.write(";angle based rep temp\n;%5s %5s %5s %5s %5s\n"%("i","j","func","-C06(Rep)","C12 (N/A)"))        
        func=1
        data.Angles()
        diam = self.excl_volume.copy()
        diam.update({"CA"+k[-1]:diam["CA"] for k in diam.keys() if k.startswith("CB")})
        eps_bbbb = 1.0*self.fconst.caltoj
        eps_bbsc = 1.0*self.fconst.caltoj
        eps_scsc = 1.0*self.fconst.caltoj

        new2old_atn={v:k for k,v in old2new_atn.items()}
        
        u_data=self.unfolded_data
        pairs,idr_pairs=[],[]
        for c in self.unfolded_data.CA_atn:
            resnum = list(u_data.CA_atn[c].keys())
            resnum.sort()
            for x in resnum:
                if u_data.CA_atn[c][x] in new2old_atn:
                    if x+2 in u_data.CA_atn[c]:pairs.append((u_data.CA_atn[c][x],u_data.CA_atn[c][x+2]))
                    if x+1 in u_data.CB_atn[c]:pairs.append((u_data.CA_atn[c][x],u_data.CB_atn[c][x+1]))
                    if x in data.CB_atn[c] and x+1 in u_data.CA_atn[c]:
                        pairs.append((u_data.CB_atn[c][x],u_data.CA_atn[c][x+1]))
                else:
                    if x+2 in u_data.CA_atn[c]:idr_pairs.append((u_data.CA_atn[c][x],u_data.CA_atn[c][x+2]))
                    for y in (x+1,x+2):
                        if y in u_data.CB_atn[c]:idr_pairs.append((u_data.CA_atn[c][x],u_data.CB_atn[c][y]))
                        if x in u_data.CB_atn[c]:
                            if y in u_data.CA_atn[c]:
                                idr_pairs.append((u_data.CB_atn[c][x],u_data.CA_atn[c][y]))
                            if x in u_data.CB_atn[c] and y in u_data.CB_atn[c]:
                                idr_pairs.append((u_data.CB_atn[c][x],u_data.CB_atn[c][y]))
        pairs,idr_pairs=np.int_(pairs),np.int_(idr_pairs)

        CA_atn = {u_data.CA_atn[c][r]:self.atomtypes[u_data.CA_atn[c][r]] for c in u_data.CA_atn for r in u_data.CA_atn[c]}
        CB_atn = {u_data.CB_atn[c][r]:self.atomtypes[u_data.CB_atn[c][r]] for c in u_data.CB_atn for r in u_data.CB_atn[c]}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())
        I,K = np.transpose(pairs)   
        interaction_type = np.int_(np.int_([x in CB_atn for x in I])+ np.int_([x in CB_atn for x in K]))
        sig = [(all_atn[I[x]],all_atn[K[x]]) for x in range(K.shape[0])]
        sig = 0.5*np.float_([diam[x]+diam[y] for x,y in sig])
        assert 2 not in interaction_type
        c06 = -1*eps_bbbb*((1.0*sig)**6)*np.int_(interaction_type==0) \
            + -1*eps_bbsc*((0.8*sig)**6)*np.int_(interaction_type==1) 
        data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
        if self.opt.opensmog:
            c=0
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==0)],\
                    params={"sig":sig[np.where(interaction_type==0)]},name="Local_backbone-backbone_rep%d"%c,\
                    expression="eps*((sig/r)^6);eps=%e"%eps_bbbb)
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==1)],\
                    params={"sig":sig[np.where(interaction_type==1)]},name="Local_backbone-sidechain_rep%d"%c,\
                    expression="eps*((0.8*sig/r)^6);eps=%e"%eps_bbsc)
        else:
            I,K = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))

        print ("> Using -C6 repulsion for IDR local beads")
        fout.write(";IDR angle based rep temp\n;%5s %5s %5s %5s %5s\n"%("i","j","func","-C06(Rep)","C12 (N/A)"))        

        I,K = np.transpose(idr_pairs)
        interaction_type = np.int_(\
            np.int_([x in CB_atn for x in I])+ \
            np.int_([x in CB_atn for x in K]))
        sig = [(all_atn[I[x]],all_atn[K[x]]) for x in range(K.shape[0])]
        sig = 0.5*np.float_([diam[x]+diam[y] for x,y in sig])
        c06 = -1*eps_bbbb*((1.0*sig)**6)*np.int_(interaction_type==0) \
            + -1*eps_bbsc*((1.0*sig)**6)*np.int_(interaction_type==1) \
            + -1*eps_scsc*((1.0*sig)**6)*np.int_(interaction_type==2) 
        pairs = np.int_([(I[x],K[x]) for x in range(I.shape[0])])
        data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
        if self.opt.opensmog:
            c=0
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==0)],\
                params={"sig":sig[np.where(interaction_type==0)]},name="IDR_Local_backbone-backbone_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_bbbb)
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==1)],\
                params={"sig":sig[np.where(interaction_type==1)]},name="IDR_Local_backbone-sidechain_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_bbsc)
            self.prot_xmlfile.write_pairs_xml( pairs=pairs[np.where(interaction_type==2)],\
                params={"sig":sig[np.where(interaction_type==2)]},name="IDR_Local_sidechain-sidechain_rep%d"%c,\
                expression="eps*((sig/r)^6);eps=%e"%eps_scsc)
        else:
            I,K = 1+np.transpose(idr_pairs)
            for x in range(len(pairs)): 
              fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))
            return 
