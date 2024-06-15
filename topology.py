import numpy as np
from PDB_IO import *
from hybrid_36 import hy36encode,hy36decode
try: from tqdm import trange,tqdm
except: tqdm,trange=list,range

class Tables:
    def __init__(self) -> None:
        pass

    def __prePad(self,X):
        step = 0.001 #nm
        return np.int_(range(0,int(X[0]*1000)))*0.001

    def write_bond_table(self,index,X,V,V_1):
        with open("table_b"+str(index)+".xvg","w+") as fout:
            for x in self.__prePad(X):           
                fout.write("%e %e %e\n"%(x,V[0],0))
            for i in range(X.shape[0]):
                fout.write("%e %e %e\n"%(X[i],V[i],-V_1[i]))
        return
    
    def __electrostatics__(self,coulomb,r):
        jtocal = 1/coulomb.caltoj		#0.239  #Cal/J #SMOG tut mentiones 1/5 = 0.2
        D = coulomb.dielec
        if coulomb.debye:
            if coulomb.inv_dl == 0:
                """debye length
                                            e0*D*KB*T
                            dl**2 = -------------------------
                                    2*el*el*NA*l_to_nm3*iconc
                
                ref: https://www.weizmann.ac.il/CSB/Levy/sites/Structural_Biology.Levy/files/publications/Givaty_JMB_2009.pdf
                """
                pi = np.pi
                inv_4pieps =  138.935485    #KJ mol-1 e-2
                iconc = coulomb.iconc 		    #M L-1
                irad = 0.1*coulomb.irad        #nm 
                T = coulomb.debye_temp			#K				#Temperature
                e0 = 8.854e-21 			#C2 J-1 nm-1	#permitivity: The value have been converted into C2 J-1 nm-1
                NA = 6.022E+23          #n/mol  		#Avogadro's number
                KB = 1.3807E-23 	    #J/K			#Boltzmann constant
                #NA = coulomb.permol          
                #KB = coulomb.Kboltz*1000.0/NA
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
                inv_dl = 10*coulomb.inv_dl #A-1 to -nm-1
            Bk = np.exp(inv_dl*irad)/(1+inv_dl*irad)
            #K_debye = (jtocal*inv_4pieps/D)*Bk
        else:
            inv_dl = 0
            Bk = 1
        K_elec = jtocal*Bk/D #inv_4pieps*q1q2 is multiplied by gromacs
        self.Bk,self.inv_dl=Bk,inv_dl
        if len(r)==0: return
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
        
    def write_pair_table(self,ljtype,coulomb):
        #writing pairs table file

        r = np.int_(range(0,250000))*0.002 #100 nm
        cutoff = np.int_(r>=0.01)
        r[0]=10E-9  #buffer to avoid division by zero

        if ljtype == 1: 
            assert coulomb.debye, "Table file only needed if using Debye-Huckel Electrostatics."
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

        if coulomb.CA or coulomb.CB or coulomb.P:
            V,V_1 = self.__electrostatics__(coulomb=coulomb,r=r)
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
    
class Preprocess:
    def __init__(self,aa_pdb,pdbindex="") -> None:
        self.file_ndx=str(pdbindex)
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
        if len(pairs)==0: return []
        i,j = np.transpose(pairs)
        if xyz0 is None: xyz0 = self.cgpdb.xyz
        if xyz1 is None: xyz1 = self.cgpdb.xyz
        return 0.1*np.sum((xyz1[j]-xyz0[i])**2,1)**0.5

    def __angles__(self,triplets):
        #takes list triplets, retuns array of angles (0-180) in deg
        if len(triplets)==0: return []
        i,j,k = np.transpose(triplets)
        xyz = self.cgpdb.xyz
        n1 = xyz[i]-xyz[j]; n2 = xyz[k]-xyz[j]
        n1n2 = (np.sum((n1**2),1)**0.5)*(np.sum((n2**2),1)**0.5)
        return np.arccos(np.sum(n1*n2,1)/n1n2)*180/np.pi

    def __torsions__(self,quadruplets):
        #takes list of quadruplets, retuns array of torsion angles (0-360) in deg
        if len(quadruplets)==0: return []
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
                if len(pairs)!=0:
                    pairs = np.int_(pairs)
                    D = self.__distances__(pairs)
                    self.bonds.append((pairs,D))

        if len(self.P_atn): #RNA/DNA existss
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
                if len(pairs)!=0:
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
                if len(triplets)!=0:
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
                if len(triplets)!=0:
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
                if len(quadruplets)!=0:
                    quadruplets = np.int_(quadruplets)
                    T = self.__torsions__(quadruplets=quadruplets)
                    self.bb_dihedrals.append((quadruplets,T))
                if len(self.CB_atn)!=0:
                    quadruplets = [(self.CB_atn[c][x],self.CA_atn[c][x],self.CA_atn[c][x+1],self.CB_atn[c][x+1]) for x in resnum if x+1 in self.CB_atn[c] and x in self.CB_atn[c]]
                    if len(quadruplets)!=0:
                        quadruplets = np.int_(quadruplets)
                        T = self.__torsions__(quadruplets=quadruplets)
                        self.sc_dihedrals.append((quadruplets,T))

        if len(self.P_atn) != 0: #RNA/DNA exists
            if len(self.S_atn) == 0:
                for c in self.P_atn:
                    resnum = list(self.P_atn[c].keys())
                    resnum.sort()
                    quadruplets = [tuple([self.P_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.P_atn[c]]
                    if len(quadruplets)!=0:
                        quadruplets = np.int_(quadruplets)
                        T = self.__torsions__(quadruplets=quadruplets)
                        self.bb_dihedrals.append((quadruplets,T))
            if len(self.S_atn) != 0:
                psps,spsp = True,True
                bsps,spsb = True,True
                assert len(self.B_atn) > 0
                for c in self.S_atn:
                    resnum = list(self.S_atn[c].keys())
                    resnum.sort()
                    if psps:
                        quadruplets = [(self.P_atn[c][x],self.S_atn[c][x], \
                                    self.P_atn[c][x+1],self.S_atn[c][x+1]) \
                                    for x in resnum if x+1 in self.S_atn[c] and \
                                    x in self.P_atn[c] and x+1 in self.P_atn[c]]
                        if len(quadruplets)!=0:
                            quadruplets = np.int_(quadruplets)
                            T = self.__torsions__(quadruplets=quadruplets)
                            self.bb_dihedrals.append((quadruplets,T))
                    if spsp:
                        quadruplets = [(self.S_atn[c][x],self.P_atn[c][x+1], \
                                    self.S_atn[c][x+1],self.P_atn[c][x+2]) \
                                    for x in resnum if x+1 in self.S_atn[c] and \
                                    x+1 in self.P_atn[c] and x+2 in self.P_atn[c]]
                        if len(quadruplets)!=0:
                            quadruplets = np.int_(quadruplets)
                            T = self.__torsions__(quadruplets=quadruplets)
                            self.bb_dihedrals.append((quadruplets,T))
                    if bsps:
                        quadruplets = [(self.B_atn[c][x],self.S_atn[c][x], \
                                       self.P_atn[c][x+1],self.S_atn[c][x+1]) \
                                    for x in resnum if x+1 in self.S_atn[c] and x+1 in self.P_atn[c]]
                        if len(quadruplets)!=0:
                            quadruplets = np.int_(quadruplets)
                            T = self.__torsions__(quadruplets=quadruplets)
                            self.sc_dihedrals.append((quadruplets,T))
                    if spsb:
                        quadruplets = [(self.S_atn[c][x],self.P_atn[c][x+1], \
                                    self.S_atn[c][x+1],self.B_atn[c][x+1]) \
                                    for x in resnum if x+1 in self.S_atn[c] and x+1 in self.P_atn[c]]
                        if len(quadruplets)!=0:
                            quadruplets = np.int_(quadruplets)
                            T = self.__torsions__(quadruplets=quadruplets)
                            self.sc_dihedrals.append((quadruplets,T))

        return
    
    def Impropers(self):
        # Getting torsion angle info from the pre-supplied data
        if len(self.CA_atn) != 0:
            for c in self.CA_atn:
                resnum = list(self.CA_atn[c].keys())
                resnum.sort()
                quadruplets = [tuple([self.CA_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.CA_atn[c]]
                if len(quadruplets)!=0:
                    quadruplets = np.int_(quadruplets)
                    T = self.__torsions__(quadruplets=quadruplets)
                    self.bb_dihedrals.append((quadruplets,T))
                if len(self.CB_atn)!=0:
                    quadruplets = [(self.CA_atn[c][x-1],self.CA_atn[c][x+1],self.CA_atn[c][x],self.CB_atn[c][x]) for x in resnum if x+1 in self.CA_atn[c] and x-1 in self.CA_atn[c] and x in self.CB_atn[c]]
                    if len(quadruplets)!=0:
                        quadruplets = np.int_(quadruplets)
                        T = self.__torsions__(quadruplets=quadruplets)
                        self.sc_dihedrals.append((quadruplets,T))

        if len(self.P_atn) != 0: #RNA/DNA exists
            for c in self.P_atn:
                resnum = list(self.P_atn[c].keys())
                resnum.sort()
                quadruplets = [tuple([self.P_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.P_atn[c]]
                if len(quadruplets)!=0:
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
                    if len(quadruplets)!=0:
                        quadruplets = np.int_(quadruplets)
                        T = self.__torsions__(quadruplets=quadruplets)
                        self.sc_dihedrals.append((quadruplets,T))
        
        return
    
    def Pairs(self,cmap,group="all",writefile=True):
        # Getting Non-bonded contact pairs info from the pre-supplied data
        temp_p,temp_c,temp_w,temp_d = [],[],[],[]
        pairs,chains,weights,distances = [],[],[],[]

        #identify what group to determine contacts for
        if group.lower() in ("p","prot","protein"): tagforfile,group="prot","prot"
        elif group.lower() in ("n","nucl","nucleic","rna","dna"): tagforfile,group="nucl","nucl"
        elif group.lower() == ("a","all"): tagforfile,group="all","all"
        elif group.lower() in ("i","inter"): tagforfile,group="inter","inter"
        tagforfile+=self.file_ndx

        if cmap.type == -1: return  # Generating top without pairs 

        elif cmap.type == 0:        # Use pairs from user input in format cid_i, atnum_i, cid_j, atnum_j, weight_ij (opt), dist_ij (opt)
            if self.file_ndx=="": file_ndx=0
            else: file_ndx=int(self.file_ndx)
            if group=="inter": infile=cmap.file
            else: infile=cmap.file[file_ndx]
            assert infile != ""
            print ("> Using cmap file (c1 a1 c2 a2 w d)",cmap.file[file_ndx])
            with open(infile) as fin:
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
                    elif len(line)>=6:
                        pairs.append((a1,a2));chains.append((c1,c2))
                        if cmap.func!=7:
                            w,d=np.float_(line[4:])
                            weights.append(w);distances.append(d)
                        elif cmap.func==7:
                            w= np.float_(line[4])
                            d=np.float_(line[5:])
                            weights.append(w);distances.append(d)

            if len(temp_p)!=0: 
                if group!="inter": temp_d = list(self.__distances__(pairs=np.int_(temp_p)))
                elif group=="inter": 
                    temp_d = list(self.__distances__(pairs=np.int_(temp_p),xyz0=self.cgpdb_n.xyz,xyz1=self.cgpdb_p.xyz))
                if cmap.func==7: temp_d=[tuple([x]) for x in temp_d]
            pairs += temp_p; chains += temp_c; weights += temp_w; distances += temp_d
            del (temp_p,temp_c,temp_d,temp_w)
            if writefile:
                print (tagforfile,cmap.file[file_ndx])
                fcg=open(tagforfile+".CGcont","w+")
                temp_c=list()
                for c in chains:
                    c=[y.split("_") for y in c]
                    if len(c[0])==2:
                        assert tagforfile.startswith(c[0][0])
                        c[0][0]=tagforfile
                    else: c[0]=[tagforfile,c[0]]
                    if len(c[1])==2:
                        assert tagforfile.startswith(c[1][0])
                        c[1][0]=tagforfile
                    else: c[1]=[tagforfile,c[1]]
                    c=tuple(["_".join(y) for y in c])
            if cmap.func!=7 and len(pairs)!=0:
                pairs=np.int_(pairs); weights=np.float_(weights); distances=np.float_(distances)
                self.contacts.append((pairs,chains,distances,weights))
                if writefile:
                    for x in range(pairs.shape[0]):
                        c,a = chains[x],pairs[x]+1
                        w,d = weights[x],distances[x]
                        fcg.write("%s %d %s %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
                    fcg.close()
            else:
                for max_dist_values in set([len(x) for x in distances]):
                    temp_p=np.int_([pairs[i] for i in range(len(pairs)) if len(distances[i])==max_dist_values])
                    if len(temp_p)==0: continue
                    temp_c=[chains[i] for i in range(len(chains)) if len(distances[i])==max_dist_values]
                    temp_w=np.float_([weights[i] for i in range(len(weights)) if len(distances[i])==max_dist_values])
                    temp_d=np.float_([distances[i] for i in range(len(distances)) if len(distances[i])==max_dist_values])
                    self.contacts.append((temp_p,temp_c,temp_d,temp_w))
                    if writefile:
                        for x in range(temp_p.shape[0]):
                            c,a = temp_c[x],temp_p[x]+1
                            w,d = temp_w[x],temp_d[x]
                            fcg.write("%s %d %s %d %.3f"%(c[0],a[0],c[1],a[1],w))
                            fcg.write(len(d)*" %.3f"%tuple(d)+"\n")
                    del (temp_p,temp_c,temp_d,temp_w)
                if writefile: fcg.close()

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
            assert len(atomgroup)!=0
            cid,rnum,bb_sc = np.transpose(np.array(atomgroup))
            del (atomgroup)

            aa2cg = {0:self.CA_atn,1:self.CB_atn,\
                     5:self.P_atn,2:self.S_atn,3:self.B_atn}

            cutoff = cmap.cutoff*cmap.scale
            resgap = 4 
            contacts_dict = dict()
            
            mol_id = np.int_(mol_id)
            str_cid  = np.array([["nucl%s_"%self.file_ndx,"prot%s_"%self.file_ndx][mol_id[x]]+\
                                    str(cid[x]+1) for x in range(mol_id.shape[0])])

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
                    pair_set = (str_cid0[i],str_cid[x]),(cg_a1,cg_a2)
                    if pair_set not in contacts_dict: contacts_dict[pair_set] = 0
                    contacts_dict[pair_set] += 1
            contacts_dict = {y:(x,contacts_dict[(x,y)]) for x,y in contacts_dict}
            pairs = list(contacts_dict.keys()); pairs.sort()
            weights = np.float_([contacts_dict[x][1] for x in pairs])
            weights = weights*(weights.shape[0]/np.sum(weights))
            if not cmap.W: weights = np.ones(weights.shape)
            #cid = np.int_([contacts_dict[x][0] for x in pairs])
            chains = [contacts_dict[x][0] for x in pairs]
            pairs = np.int_(pairs)
            if len(pairs)>0:
                if group!="inter": distances = self.__distances__(pairs=pairs)
                elif group=="inter": distances = self.__distances__(pairs=pairs,xyz0=self.cgpdb_n.xyz,xyz1=self.cgpdb_p.xyz)
                for x in range(pairs.shape[0]):
                    c,a = chains[x],pairs[x]+1
                    w,d = weights[x],distances[x]
                    fcg.write("%s %d %s %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
                self.contacts.append((pairs,chains,distances,weights))
            faa.close();fcg.close()

        elif cmap.type == 2:        # Calculating contacts from CG structure
            cid = []
            if len(self.CA_atn) != 0:
                tag="prot%s_"%self.file_ndx
                cacasep=4;cacbsep=3;cbcbsep=3
                for c1 in self.CA_atn:
                    pairs += [(self.CA_atn[c1][x],self.CA_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c1] if y-x>=cacasep]
                    if len(self.CB_atn) != 0:
                        pairs += [(self.CA_atn[c1][x],self.CB_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CA_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CB_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c1] if y-x>=cbcbsep]
                    cid += [(tag+str(c1+1),tag+str(c1+1)) for x in range(len(pairs)-len(cid))]
                    for c2 in self.CA_atn:
                        if c2>c1: 
                            pairs += [(self.CA_atn[c1][x],self.CA_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c2]]
                            if len(self.CB_atn)!=0: 
                                pairs += [(self.CA_atn[c1][x],self.CB_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CA_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CB_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c2]]
                            cid += [(tag+str(c1+1),tag+str(c2+1)) for x in range(len(pairs)-len(cid))]
            if len(self.P_atn)!=0:
                tag="nucl%s_"%self.file_ndx
                ppsep,ressep=1,1
                if len(self.S_atn)==0:
                    for c1 in self.P_atn:
                        pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c1] if y-x>=ppsep]
                        cid += [(tag+str(c1+1),tag+str(c1+1)) for x in range(len(pairs)-len(cid))]
                        for c2 in self.P_atn: 
                            if c2>c1:
                                pairs += [(self.P_atn[c1][x],self.P_atn[c1][y]) for x in self.P_atn[c1] for y in self.P_atn[c2]]
                                cid += [(tag+str(c1+1),tag+str(c2+1)) for x in range(len(pairs)-len(cid))]
                else:
                    assert len(self.B_atn)!=0
                    for c1 in self.S_atn:
                        all_atn_c1 = list(self.P_atn[c1].items())+list(self.S_atn[c1].items())+list(self.B_atn[c1].items())
                        pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c1 if ry-rx>=ressep]
                        cid += [(tag+str(c1+1),tag+str(c1+1)) for x in range(len(pairs)-len(cid))]
                        for c2 in self.S_atn:
                            if c2>c1:
                                all_atn_c2 = list(self.P_atn[c2].items())+list(self.S_atn[c2].items())+list(self.B_atn[c2].items())
                                pairs += [(ax,ay) for rx,ax in all_atn_c1 for ry,ay in all_atn_c2]
                                cid += [(tag+str(c1+1),tag+str(c2+1)) for x in range(len(pairs)-len(cid))]

            pairs = np.int_(pairs)
            if len(pairs)>0:
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
    def __init__(self,proc_data,Nprot,Nnucl,topfile,opt,excl_volume,excl_rule,fconst,cmap,coul):
        self.data = proc_data
        self.topfile=topfile
        self.nucl_cmap = cmap["nucl"]
        self.prot_cmap = cmap["prot"]
        self.inter_cmap= cmap["inter"]
        self.opt = opt
        self.excl_volume = excl_volume
        self.atomtypes = {}
        self.excl_rule = excl_rule
        self.fconst = fconst
        #hold data for functions
        self.atoms_in_mol = list()
        self.mol_surface_atoms = list()
        self.old2new_atomtypes = {k:[k] for k in self.excl_volume}
        self.molatnum2new_atomtypes = {}
        self.atoms_section=str()
        if self.opt.opensmog:
            self.xmlfile=OpenSMOGXML(xmlfile=self.opt.xmlfile,coulomb=coul)
        self.__merge__(Nprot=Nprot,Nnucl=Nnucl,opt=opt,excl_volume=excl_volume)
        
    def nPlaces(self,n,count2str):
        return "0"*(n-len(str(count2str)))+str(count2str)

    def __topParse(self,topfile_tag):
        topfile="%s_%s"%(topfile_tag,self.topfile)
        print ("> Parsing",topfile)
        top={x.split("]")[0].strip():x.split("]")[1].strip().split("\n") for x in open(topfile).read().split("[") if len(x.split("]"))==2}
        extras=[x.split() for x in open(topfile).read().split("[") if len(x.split("]"))<2][0]
        order=[x.split("]")[0].strip() for x in open(topfile).read().split("[") if len(x.split("]"))==2]
        return top,extras,order

    def __smogxmlParse(self,xmlfile_tag):
        xmlfile="%s_%s"%(xmlfile_tag,self.opt.xmlfile)
        data=dict()
        for line in open(xmlfile):
            indent=len(line)-len(line.lstrip())
            if indent==0: continue
            elif indent==1: 
                if "/" in line: continue
                tag=line
                if tag not in data: data[tag]={}
            elif indent==2: 
                if "/" in line: continue
                subtag=line
                if subtag not in data[tag]:
                    data[tag][subtag]={"expr":str(),"params":list(),"data":list()}
            elif indent==3:
                if "expr" in line: data[tag][subtag]["expr"]=line
                elif "parameter" in line: data[tag][subtag]["params"].append(line)
                else: data[tag][subtag]["data"].append(line)
        return data

    def __getSurfaceAtoms(self,contacts,tag):
        surface=list()
        for psirs,chains,eps,sig in contacts:
            I,J = 1+np.transpose(psirs)
            surface += [I[x] for x in range(I.shape[0]) if chains[x][0].split("_")[0]==tag]
            surface += [J[x] for x in range(J.shape[0]) if chains[x][1].split("_")[0]==tag]
        surface=list(set(surface))
        surface.sort()
        return surface

    def __getAtomsSection(self,inp,nmol,tag,prev_at_count=0,surface=[]):
        #writing merged atoms section
        section = str()
        offset = 0
        for y in inp:
            if y.strip().startswith(";") or y.strip() == "": continue
            offset = 1 - int(y.split()[0])
            break
        atoms_in_mol = offset
        assert tag not in self.atomtypes
        self.atomtypes[tag]=[]
        assert tag not in self.molatnum2new_atomtypes
        self.molatnum2new_atomtypes[tag]={}
        for x in range(0,nmol):
            section+=";%s_%s\n"%(tag,self.nPlaces(n=3,count2str=x+1))
            for y in inp:
                if y.strip() == "": continue
                elif y.strip().startswith(";"): section+=y+"\n"
                else:
                    i=y.strip().split()
                    if int(i[0]) in surface:
                        new_type="%s_%s%s"%(i[1],i[0],tag[0]+tag[4:])
                        #self.excl_volume[new_type]=self.excl_volume[i[1]]
                        if new_type not in self.old2new_atomtypes[i[1]]:
                            self.old2new_atomtypes[i[1]].append(new_type)
                        self.molatnum2new_atomtypes[tag][int(i[0])]=new_type
                        i[1]=new_type
                    a0 = offset + int(i[0]) + prev_at_count + x*atoms_in_mol
                    a5 = offset + int(i[5]) + prev_at_count + x*atoms_in_mol
                    section+="  %5d %10s %4s %5s %5s %5d %5s %5s\n" % (a0, i[1].center(10), i[2], i[3], i[4], a5, i[6], i[7])
                    self.atomtypes[tag].append(i[1])
                    if x==0: atoms_in_mol = int(i[0])
        self.atoms_in_mol.append(atoms_in_mol)
        self.atoms_section += section
        return atoms_in_mol*nmol

    def __writeAtomtypesSection(self,fsec,inp,exclude=[]):
        temp = []
        for data in inp:
            for line in data:
                if line.strip() != "" and line not in temp:
                    temp.append(line)
                    if line.strip()[0]==";": 
                        fsec.write("%s\n"%line)
                        continue
                    line=line.split()
                    bead=line[0]
                    for new_bead in self.old2new_atomtypes[bead]:
                        fsec.write(" %10s"%new_bead.center(10))
                        fsec.write(len(line[1:])*" %s"%tuple(line[1:])+"\n")

    def __writeSymPairs2Nonbond(self,fsec,inp,cmap_func):
        nonbond_pairs={}
        for pairs,chains,eps,sig in inp:
            C1,C2 = np.transpose(chains)
            C1,C2 = [x.split("_")[0] for x in C1],[x.split("_")[0] for x in C2]
            I,J=1+np.transpose(pairs)
            I=[self.molatnum2new_atomtypes[C1[x]][I[x]] for x in range(I.shape[0])]
            J=[self.molatnum2new_atomtypes[C2[x]][J[x]] for x in range(J.shape[0])]
            func=1
            if cmap_func==1: values = 2*eps*((sig)**6),1*eps*((sig)**12)
            elif cmap_func==2: values = 6*eps*((sig)**10),5*eps*((sig)**12)
            elif cmap_func in (5,6,7):
                assert self.opt.opensmog
                func,sd = 6,0.05
                I_rad = np.float_([self.excl_volume[I[x].split("_")[0]] for x in range(len(I))])
                J_rad = np.float_([self.excl_volume[J[x].split("_")[0]] for x in range(len(J))])
                if self.excl_rule == 1: C12 = ((I_rad**12.0)*(J_rad**12.0))**0.5
                elif self.excl_rule == 2: C12 = ((I_rad+J_rad)/2.0)**12.0
                values = eps,sig,np.ones(C12.shape[0])*sd,C12
            values = np.transpose(values)
            for x in range(pairs.shape[0]):
                p=[I[x],J[x]]; p.sort(); p=tuple(p)
                ptype=[y.split("_")[0] for y in p]; ptype.sort();ptype=ptype=tuple(ptype)
                if ptype not in nonbond_pairs: nonbond_pairs[ptype]={}
                nonbond_pairs[ptype][p]=0
                if self.opt.opensmog:
                    self.xmlfile.write_nonbond_param_entries(pairs=[p],params={"C12":[values[x][-1]],"epsA":[eps[x]],"sig":[sig[x]]})
                    continue
                fsec.write(" %10s %10s\t %5d\t"%(p[0].center(10),p[1].center(10),func))
                fsec.write(len(values[x])*" %e"%tuple(values[x])+"\n")
            
        return nonbond_pairs

    def __writeNucProtParams(self,fsec,exclude={}):
        print ("> Writing user given custom nonbond_params:",self.opt.interface)
        eps,sig = self.data[0].Interactions(interface=self.opt.interface)
        cmap_func=self.inter_cmap.func

        pairs = []
        Krep = (self.fconst.Kr_prot+self.fconst.Kr_nucl)*0.5
        for x in self.excl_volume:
            if not x.startswith(("CA","CB")):
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        func=1
                        if self.excl_rule==1: C12 = Krep*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                        elif self.excl_rule==2: C12 = Krep*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                        if cmap_func in (1,2): values = 0,C12
                        elif cmap_func in (5,6,7):
                            assert self.opt.opensmog
                            func,sd = 6,0.05
                            values = 0,0,sd,C12
                        p = [x,y]; ptype=p; p=tuple(p)
                        ptype.sort();ptype=tuple(ptype)
                        if p not in pairs and p not in eps:
                            pairs.append(p)
                            for i in self.old2new_atomtypes[p[0]]:
                                for j in self.old2new_atomtypes[p[1]]:
                                    a=[i,j]; a.sort(); a=tuple(a)
                                    if ptype in exclude and a in exclude[ptype]: continue
                                    if self.opt.opensmog:
                                        self.xmlfile.write_nonbond_param_entries(pairs=[(i,j)],params={"C12":[values[-1]],"epsA":[1],"sig":[0]})
                                        continue
                                    fsec.write(" %10s %10s\t %d\t"%(i.center(10),j.center(10),func))
                                    fsec.write(len(values)*" %e"%tuple(values))
                                    fsec.write("\n")

        if len(eps)>0:
            fsec.write("; Custom Protein-RNA/DNA interactions\n")
            if self.nucl_cmap.func in (5,6,7):
                fsec.write(";%5s %5s %5s %5s %5s %5s %5s\n"%("i","j","func","eps","r0","sd","C12(Rep)"))
            for p in eps:
                if p[0].startswith(("CA","CB")): continue
                if not p[1].startswith(("CA","CB")): continue
                if p[0] not in self.excl_volume: continue
                if p[1] not in self.excl_volume: continue
                p=list(p); p.sort(); p=tuple(p)
                if p in pairs: continue
                pairs.append(p); ptype=p
                if p not in sig: cmap_func=-1
                else: sig[p]=sig[p][0] #to be changed for Gaussian
                func=1
                if cmap_func ==-1:
                    if self.excl_rule==1: c12 = eps[p]*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif self.excl_rule==2: C12 = eps[p]*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = 0.0,C12
                elif cmap_func==1: values = 2*eps[p]*((sig[p])**6),1*eps[p]*((sig[p])**12)
                elif cmap_func==2: values = 6*eps[p]*((sig[p])**10),5*eps[p]*((sig[p])**12)
                elif cmap_func in (5,6,7):
                    assert self.opt.opensmog
                    func,sd = 6,0.05
                    if self.excl_rule==1: c12 = (((self.excl_volume[x.split("_")[0]]**12)*(self.excl_volume[y]**12))**0.5)
                    elif self.excl_rule==2: C12 = ((self.excl_volume[x.split("_")[0]]+self.excl_volume[y])/2.0)**12                
                    values = eps[p],sig[p],sd,C12
                for x in self.old2new_atomtypes[p[0]]:
                    for y in self.old2new_atomtypes[p[1]]:
                        a=[x,y]; a.sort(); a=tuple(a)
                        if ptype in exclude and a in exclude[ptype]:continue
                        if self.opt.opensmog:
                            self.xmlfile.write_nonbond_param_entries(pairs=[(x,y)],params={"C12":[values[-1]],"epsA":[eps[p]],"sig":[sig[p]]})
                            continue
                        fsec.write(" %10s %10s\t %d\t"%(x.center(10),y.center(10),func))
                        fsec.write(len(values)*" %e"%tuple(values))
                        fsec.write("\n")

        return pairs

    def __writeNonbondParams(self,fsec,inp,exclude={}):
        print ("> Ignoring allready added %d contact pair-types."%len(exclude))
        temp=[]
        for data in inp:
            for line in data:
                if line.strip().startswith(";"): fsec.write(line+"\n")
                elif line.strip() != "":
                    line=line.split()
                    beads=line[:2]; beads.sort(); beads=tuple(beads)
                    c12=line[-1]
                    if beads in temp: continue
                    temp.append(beads)
                    b1 = self.old2new_atomtypes[beads[0]]
                    b2 = self.old2new_atomtypes[beads[1]]
                    for i in range(len(b1)):
                        for j in range(len(b2)):
                            if j<i: continue
                            new_beads=[b1[i],b2[j]];new_beads.sort();new_beads=tuple(new_beads)
                            if beads in exclude and new_beads in exclude[beads]: continue
                            fsec.write(2*" %10s"%(new_beads[0].center(10),new_beads[1].center(10)))
                            fsec.write("\t"+len(line[2:])*" %5s"%tuple(line[2:])+"\n")
        return

    def __writeNonbondParamsXML(self,fsec,inp,exclude={}):
        print ('> Ignoring allready added %d contact pair-types.'%len(exclude))
        temp=[]
        for data in inp:
            for line in data:
                line=line.split('"')
                assert line[0].endswith('type1=') and line[2].endswith('type2=')
                beads=[line[1],line[3]]; beads.sort(); beads=tuple(beads)
                if beads in temp: continue
                temp.append(beads)
                b1 = self.old2new_atomtypes[beads[0]]
                b2 = self.old2new_atomtypes[beads[1]]
                for i in range(len(b1)):
                    for j in range(len(b2)):
                        if j<i: continue
                        new_beads=[b1[i],b2[j]];new_beads.sort();new_beads=tuple(new_beads)
                        if beads in exclude and new_beads in exclude[beads]: continue
                        fsec.write(line[0]+3*'"%s'%(new_beads[0],line[2],new_beads[1]))
                        fsec.write(len(line[4:])*'"%s'%tuple(line[4:]))
        return

    def __writeInteractions(self,fsec,nparticles,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset=0):
        for x in range(0,nmol):
            fsec.write(";%s_%s\n"%(tag,self.nPlaces(n=3,count2str=x+1)))
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

    def __writeSymPaIrs(self,fsec,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset=0):
        fsec.write(";%s_symmetrized_interactions_%s\n"%(tag,self.nPlaces(n=3,count2str=nmol)))
        for y in inp:
            if y.strip() == "": continue
            elif y.strip().startswith(";"): fsec.write(y+"\n")
            else:
                line=y.strip() .split()
                I = [atnum_offset + int(line[0]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
                J = [atnum_offset + int(line[1]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
                for i in range(len(I)):
                    for j in range(len(J)): 
                        fsec.write(2*" %5d"%(I[i],J[j]))
                        fsec.write(len(line[2:])*" %5s"%tuple(line[2:]))
                        fsec.write("; %s(%d,%d) %5s %5s\n"%tuple([tag,i,j]+line[:2]))
        return

    def __writeInteractionsXML(self,fsec,nparticles,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset=0):
        for x in range(0,nmol):
            #fsec.write(';%s_%s\n'%(tag,self.nPlaces(n=3,count2str=x+1)))
            for line in inp:
                line=line.split('"')
                assert line[0].endswith('i=') and line[2].endswith('j=')
                a = [atnum_offset + int(line[2*i+1]) + prev_at_count + x*atoms_in_mol for i in range(nparticles)]
                fsec.write(line[0])
                for i in range(nparticles): fsec.write('"%d"%s'%(a[i],line[2*(i+1)]))
                fsec.write(len(line[2*nparticles+1:])*'"%s'%tuple(line[2*nparticles+1:]))            
        return

    def __writeSymPaIrsXML(self,fsec,inp,nmol,prev_at_count,atoms_in_mol,tag,atnum_offset=0):
        fsec.write(";%s_symmetrized_interactions_%s\n"%(tag,self.nPlaces(n=3,count2str=nmol)))
        for line in inp:
            line=line.split('"')
            group = []
            I = [atnum_offset + int(line[1]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
            J = [atnum_offset + int(line[3]) + prev_at_count + x*atoms_in_mol for x in range(nmol)]
            for i in range(len(I)):
                for j in range(len(J)): 
                    fsec.write('%s"%d"%s"%d'%(line[0],I[i],line[2],J[j]))
                    fsec.write(len(line[4:])*'"%s'%tuple(line[4:]))
        return

    def __writeInterPairs(self,fsec,inp,nmol,prev_at_count,atoms_in_mol,tag,sym=True,neighlist=False,atnum_offset=0):
        print ("> Determining Inter pairs")
        atoms_in_mol=self.atoms_in_mol
        cmap_func=self.inter_cmap.func
        if not sym: 
            assert len(nmol)==sum(nmol),"Multplte units not supported if not symmeterized"
        prev_at_count={tag[x]:int(sum(prev_at_count[:x+1])) for x in range(len(tag))}
        nmol={tag[x]:nmol[x] for x in range(len(tag))}
        atoms_in_mol={tag[x]:atoms_in_mol[x] for x in range(len(tag))}
        inter_exclusions=[";Inter-molecule symmetrized_interactions\n"]
        fsec.write(";Inter-molecule symmetrized_interactions\n")
        xml_pairs_data=list()
        for pairs,chains,eps,sig in inp:
            I,J = np.transpose(pairs)
            C1,C2 = np.transpose(chains)
            C1,C2 = [x.split("_")[0] for x in C1],[x.split("_")[0] for x in C2]
            sym_I = [1 + np.int_([I[y] + x*atoms_in_mol[C1[y]] for x in range(nmol[C1[y]])]) + \
                        atnum_offset + prev_at_count[C1[y]] for y in range(I.shape[0])]
            sym_J = [1 + np.int_([J[y] + x*atoms_in_mol[C2[y]] for x in range(nmol[C2[y]])]) + \
                        atnum_offset + prev_at_count[C2[y]] for y in range(J.shape[0])]
            func=1
            if cmap_func==1: values = 2*eps*((sig)**6),1*eps*((sig)**12)
            elif cmap_func==2: values = 6*eps*((sig)**10),5*eps*((sig)**12)
            elif cmap_func in (5,6,7):
                func,sd = 6,0.05
                I_rad = np.float_([self.excl_volume[self.atomtypes[C1[x]][I[x]].split("_")[0]] for x in range(I.shape[0])])
                J_rad = np.float_([self.excl_volume[self.atomtypes[C2[x]][J[x]].split("_")[0]] for x in range(J.shape[0])])
                if self.excl_rule == 1: C12 = ((I_rad**12.0)*(J_rad**12.0))**0.5
                elif self.excl_rule == 2: C12 = ((I_rad+J_rad)/2.0)**12.0
                values = eps,sig,np.ones(C12.shape[0])*sd,C12
            values = np.transpose(values)
            I,J = 1 + np.transpose(pairs)
            for x in range(pairs.shape[0]):
                for i in range(len(sym_I[x])):
                    for j in range(len(sym_J[x])):
                        if neighlist:
                            if C1[x]==C2[x] and i==j and sym_I[x][0]!=sym_J[x][0]:
                                inter_exclusions.append(" %5s %5s ; %s(%d)-%s(%d) %5d %5d\n"%(sym_I[x][i],sym_J[x][i],C1[x],i,C2[x],j,I[x],J[x]))
                            continue
                        if C1[x]==C2[x] and i==j: continue
                        inter_exclusions.append(" %5s %5s ; %s(%d)-%s(%d) %5d %5d\n"%(sym_I[x][i],sym_J[x][j],C1[x],i,C2[x],j,I[x],J[x]))
                        if self.opt.opensmog:
                            xml_pairs_data.append((sym_I[x][i],sym_J[x][j],eps[x],sig[x],values[x][-1]))
                            continue
                        fsec.write(" %5s %5s\t%d\t"%(sym_I[x][i],sym_J[x][j],func))
                        fsec.write(len(values[x])*" %e"%tuple(values[x]))
                        fsec.write("; %s(%d)-%s(%d) %5d %5d\n"%(C1[x],i,C2[x],j,I[x],J[x]))

        if self.opt.opensmog and len(xml_pairs_data)!=0:
            I,J,eps,sig,C12=np.transpose(xml_pairs_data)
            pairs=-1+np.int_([(I[x],J[x]) for x in range(len(pairs))])
            params={"r0":sig,"eps":eps}

            print (cmap_func);exit()
            if cmap_func==1: expr,name="eps*( 1*((r0/r)^12) - 2*((r0/r)^10) )","symInter_contacts_LJ-06-12"
            elif cmap_func==2: expr,name="eps*( 5*((r0/r)^12) - 6*((r0/r)^10) )","symInter_contacts_LJ-10-12"
            elif cmap_func in (5,6,7):
                sd=0.05;params["C12"]=C12
                expr="eps*((1+(C12/(r^12)))*(1-exp(-((r-r0)^2)/(2*(sd^2))))-1); sd=%e"%sd
                name="symInter_contacts_Gaussian-12"
            self.xmlfile.write_pairs_xml(pairs=pairs,name=name,params=params,expression=expr)

        return inter_exclusions

    def __merge__(self,Nprot,Nnucl,opt,excl_volume):
        #merting  top file

        #for every input PDB there is an Nprot and Nnucl value (=0 if either if not present)
        assert len(Nprot)==len(Nnucl)
        Ninp=len(Nprot)         #number of input PDBs (and hence .top/.xml files)
        outfile=self.topfile    #combined topfile name

        parsed_top,parsed_xml=[],[]     #list of parsed top/xml files
        tag_list,nmol_list=[],[]        #order of nucl or prot top files
        temp_data=list()                #order of data to be added to merfed top/xml files.
        pdb_input_ndx=list()            #index of input PDB file (to be written to molecule_order.list file for reference)
        if Ninp==1:
            # top and xml files do-not have index after prot or nucl tags if a single PDB input
            if Nnucl[0]>0:  #nucl molecule data is added first
                pdb_input_ndx.append(0)
                if self.opt.opensmog: 
                        parsed_xml.append(self.__smogxmlParse(xmlfile_tag="nucl"))
                parsed_top.append(self.__topParse(topfile_tag="nucl"))
                tag_list.append("nucl");nmol_list.append(Nnucl[0])
                temp_data.append(self.data[0])
            if Nprot[0]>0: #prot molecule data is added after nucl data
                pdb_input_ndx.append(0)
                if self.opt.opensmog: 
                        parsed_xml.append(self.__smogxmlParse(xmlfile_tag="prot"))
                parsed_top.append(self.__topParse(topfile_tag="prot"))
                tag_list.append("prot");nmol_list.append(Nprot[0])
                temp_data.append(self.data[0])
        else:        
            # top and xml files have indices (0,1,2,..) after prot or nucl tags for multiple PDB inputs
            for i in range(Ninp):   #All nucl molecules are added first
                if Nnucl[i]>0: 
                    pdb_input_ndx.append(i)
                    if self.opt.opensmog: 
                        parsed_xml.append(self.__smogxmlParse(xmlfile_tag="nucl%d"%i))
                    parsed_top.append(self.__topParse(topfile_tag="nucl%d"%i))
                    tag_list.append("nucl%d"%i);nmol_list.append(Nnucl[i])
                    temp_data.append(self.data[i])
            for i in range(Ninp): #All prot molecules are added after nucl molecules
                if Nprot[i]>0:
                    pdb_input_ndx.append(i)
                    if self.opt.opensmog: 
                        parsed_xml.append(self.__smogxmlParse(xmlfile_tag="prot%d"%i))
                    parsed_top.append(self.__topParse(topfile_tag="prot%d"%i))
                    tag_list.append("prot%d"%i);nmol_list.append(Nprot[i])
                    temp_data.append(self.data[i])
        
        #counting nucl and prot data as separate inputs (all nucl first followed by all prots in their input order)
        if opt.control_run: assert(len(self.data)==1)
        Ninp,self.data=len(parsed_top),temp_data
        del(temp_data)
        
        inter_contacts=list()   #list for loading inter-PDB contacts 
        self.mol_surface_atoms = [[] for x in range(Ninp)] #lsit of atoms in interface contacts
        if len(nmol_list)>1:    #more than 1 input files
            assert self.prot_cmap.nbfunc==self.nucl_cmap.nbfunc==self.inter_cmap.nbfunc
            nbfunc=self.inter_cmap.nbfunc
            if self.inter_cmap.type>=0:
                if not opt.control_run:     #determining contacts only supported with control runs
                    assert self.inter_cmap.type == 0, \
                        "Error, calculating inter molecule contacts only supported for --control runs with single input PDB. Provide custom file --cmap_i" 

            self.data[0].Pairs(cmap=self.inter_cmap,group="inter",writefile=False)
            inter_contacts = self.data[0].contacts.copy()
            self.data[0].contacts=[]
            opt.inter_symmetrize=True
            # for N>3 molecules, symmeterized contacts can be added to nonbond-paramns if nbfunc is same as contact func
            if sum(nmol_list)>5 and self.inter_cmap.nbfunc==self.inter_cmap.func:
                add_inter_2_neighlist=True
                for x in range(Ninp):
                    self.mol_surface_atoms[x] += self.__getSurfaceAtoms(contacts=inter_contacts,tag=tag_list[x])
            else: add_inter_2_neighlist=False
            if self.opt.opensmog: 
                #since all interactions have same nbfunc, there should be only 1 identical equations in all xml files
                #if this is not true (custom added model not suited for merging), then the program wiil terminate
                uniq_expr=set([parsed_xml[i][tag][subtag]["expr"] for i in range(Ninp) \
                                for tag in parsed_xml[i] for subtag in parsed_xml[i][tag]\
                                if "nonbond" in tag and "nonbond" in subtag])
                assert len(uniq_expr)==1, "Error, Cannot use custom expressions while merging files"
        else:
           opt.inter_symmetrize=False  
        if opt.intra_symmetrize:
            add_intra_2_neighlist=[nmol_list[i]>=3 for i in range(Ninp)]
            assert self.prot_cmap.nbfunc==self.nucl_cmap.nbfunc==self.inter_cmap.nbfunc
            nbfunc=self.inter_cmap.nbfunc
            for x in range(Ninp):
                if "prot" in tag_list[x]:cmap=self.prot_cmap
                elif "nucl" in tag_list[x]:cmap=self.nucl_cmap
                else: print (tag_list[x])
                cmap.file[pdb_input_ndx[x]]=tag_list[x]+".CGcont"
                cmap.type=0
                self.data[x].Pairs(cmap=cmap,group=tag_list[x][:4],writefile=False)
                # if nbfunc and contfunc are not same, then cannot add sym contacts to neighbout list
                if cmap.func != nbfunc: add_intra_2_neighlist[x]=False
                #if adding to neighbour list get surface atoms to be excluded from rep term in nonbond params
                if add_intra_2_neighlist[x]:
                    self.mol_surface_atoms[x] += self.__getSurfaceAtoms(contacts=self.data[x].contacts.copy(),tag=tag_list[x])

        if opt.opensmog: 
            #nonbond tag and subtag
            nb_tag=[tag for tag in parsed_xml[0] if "nonbond" in tag][0] 
            nb_subtag=[subtag for subtag in parsed_xml[0][nb_tag]][0]
            #contacts tag and subtag
            ct_tag=list(set([tag for i in range(Ninp) for tag in parsed_xml[i] if "contacts" in tag]))
            assert len(ct_tag)<=1
            if len(ct_tag)==1: ct_tag=ct_tag[0]

        print (">> Detected %d topology file(s)"%Ninp)
        with open("molecule_order.list","w+") as fout:
            fout.write("#inp_ndx mol_typ num_mol\n")
            for i in range(Ninp): 
                print (pdb_input_ndx[i],tag_list[i],nmol_list[i])
                fout.write(" %7d %7s %7d\n"%(pdb_input_ndx[i],tag_list[i].rjust(7),nmol_list[i]))
        
        print (">>> writing Combined GROMACS toptology", outfile)
        with open(outfile,"w+") as fout:
            ##
            Nparticles={"bonds":2,"angles":3,"pairs":2,"dihedrals":4,"exclusions":2,"nonbond_params":2,"atomtypes":1}
            fout.write("\n; Topology file generated by SuBMIT.\n")
            order=parsed_top[0][2]
            ##
            atom_list = [parsed_top[x][0]["atoms"] for x in range(Ninp)]
            prev_natoms=np.zeros(Ninp+1)
            for i in range(Ninp):
                prev_natoms[i+1] = self.__getAtomsSection(inp=atom_list[i],nmol=nmol_list[i],tag=tag_list[i], \
                                            prev_at_count=sum(prev_natoms),surface=self.mol_surface_atoms[i])
            ##
            for header in order:
                data_list = [parsed_top[x][0][header] for x in range(Ninp)]
                print ("> Writing",outfile,header,"section.")
                fout.write("\n[ "+header+" ]\n")
                ##
                if header == "atomtypes":
                    self.__writeAtomtypesSection(fsec=fout,inp=data_list)
                elif header == "nonbond_params":
                    if opt.opensmog:
                        print ("> Writing %s nonbond section."%opt.xmlfile)
                        expr=parsed_xml[0][nb_tag][nb_subtag]["expr"]
                        self.xmlfile.fxml.write(nb_tag+nb_subtag+expr)
                        for param in parsed_xml[0][nb_tag][nb_subtag]["params"]: self.xmlfile.fxml.write(param)
                    exclude_nonbond_pairs = {}
                    if opt.intra_symmetrize:
                        for i in range(Ninp):
                            if not add_intra_2_neighlist[i]: continue
                            exclude_nonbond_pairs.update(self.__writeSymPairs2Nonbond(fsec=fout,cmap_func=nbfunc,inp=self.data[i].contacts))
                    if opt.inter_symmetrize and add_inter_2_neighlist:
                        exclude_nonbond_pairs.update(self.__writeSymPairs2Nonbond(fsec=fout,cmap_func=nbfunc,inp=inter_contacts))
                    if sum(Nprot)>0 and sum(Nnucl)>0: self.__writeNucProtParams(fsec=fout,exclude=exclude_nonbond_pairs)
                    self.__writeNonbondParams(fsec=fout,inp=data_list,exclude=exclude_nonbond_pairs)
                    if opt.opensmog:
                        xml_data_list=[parsed_xml[i][nb_tag][nb_subtag]["data"] for i in range(Ninp)]
                        self.__writeNonbondParamsXML(fsec=self.xmlfile.fxml,inp=xml_data_list,exclude=exclude_nonbond_pairs)
                        self.xmlfile.fxml.write("%s</%s"%tuple(nb_subtag.split("<")))
                        self.xmlfile.fxml.write("%s</%s"%tuple(nb_tag.split("<")))
                        if len(ct_tag)!=0: self.xmlfile.fxml.write(ct_tag)
                elif header == "atoms":
                    fout.write(self.atoms_section)
                elif header in ["bonds","angles","dihedrals"]:
                    for i in range(Ninp):
                        self.__writeInteractions(fsec=fout,nparticles=Nparticles[header],inp=data_list[i],tag=tag_list[i], \
                                    nmol=nmol_list[i],prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i])
                        if header=="bonds" and opt.opensmog and len(ct_tag)!=0:
                            if ct_tag not in parsed_xml[i]: continue
                            ct_subtags=[subtag for subtag in parsed_xml[i][ct_tag] \
                                            if "contacts" not in subtag.split("=")[-1].lower()]
                            for subtag in ct_subtags:
                                print ("> Writing %s bonds to %s contacts section."%(subtag.split('"')[1],opt.xmlfile))
                                new_subtag=subtag.split('"');new_subtag[1]="%s_%s"%(tag_list[i],new_subtag[1])
                                new_subtag='"'.join(new_subtag)
                                expr=parsed_xml[i][ct_tag][subtag]["expr"]
                                self.xmlfile.fxml.write(new_subtag+expr)
                                for param in parsed_xml[i][ct_tag][subtag]["params"]: self.xmlfile.fxml.write(param)
                                xml_data=parsed_xml[i][ct_tag][subtag]["data"]
                                self.__writeInteractionsXML(fsec=self.xmlfile.fxml,nparticles=Nparticles[header],inp=xml_data,tag=tag_list[i],\
                                                                nmol=nmol_list[i],prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i])
                                self.xmlfile.pairs_count+=1
                                self.xmlfile.fxml.write("%s</%s>\n"%tuple(subtag.split()[0].split("<")))
                elif header in ["pairs","exclusions"]:
                    for i in range(Ninp):
                        if not opt.intra_symmetrize: 
                            self.__writeInteractions(fsec=fout,nparticles=Nparticles[header],inp=data_list[i],tag=tag_list[i], \
                                        nmol=nmol_list[i],prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i])                    
                        elif opt.intra_symmetrize: 
                            if not add_intra_2_neighlist[i]:
                                self.__writeSymPaIrs(fsec=fout,inp=data_list[i],nmol=nmol_list[i], \
                                    prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i],tag=tag_list[i])
                        if header=="pairs" and opt.opensmog and len(ct_tag)!=0:
                            if ct_tag not in parsed_xml[i]: continue
                            ct_subtags=[subtag for subtag in parsed_xml[i][ct_tag] \
                                        if "contacts" in subtag.split("=")[-1].lower()]
                            for subtag in ct_subtags:
                                print ("> Writing %s pairs to %s contacts section."%(subtag.split('"')[1],opt.xmlfile))
                                new_subtag=subtag.split('"');new_subtag[1]="%s_%s"%(tag_list[i],new_subtag[1])
                                new_subtag='"'.join(new_subtag)
                                expr=parsed_xml[i][ct_tag][subtag]["expr"]
                                self.xmlfile.fxml.write(new_subtag+expr)
                                for param in parsed_xml[i][ct_tag][subtag]["params"]: self.xmlfile.fxml.write(param)
                                xml_data=parsed_xml[i][ct_tag][subtag]["data"]
                                if not opt.intra_symmetrize:
                                    self.__writeInteractionsXML(fsec=self.xmlfile.fxml,nparticles=Nparticles[header],inp=xml_data,tag=tag_list[i],\
                                                                nmol=nmol_list[i],prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i])
                                elif opt.intra_symmetrize:
                                    if not add_intra_2_neighlist[i]:
                                        self.__writeSymPaIrsXML(fsec=self.xmlfile.fxml,inp=xml_data,nmol=nmol_list[i],tag=tag_list[i], \
                                                                prev_at_count=sum(prev_natoms[:i+1]),atoms_in_mol=self.atoms_in_mol[i])
                                self.xmlfile.pairs_count+=1
                                self.xmlfile.fxml.write("  %s</%s>\n"%tuple(subtag.split()[0].split("<")))
                    if header == "pairs":
                        if len(inter_contacts)>0:
                            if add_inter_2_neighlist:
                                inter_exclusions_section = self.__writeInterPairs(fsec=fout,inp=inter_contacts,nmol=nmol_list,tag=tag_list,\
                                                                prev_at_count=prev_natoms,atoms_in_mol=self.atoms_in_mol[i],neighlist=add_inter_2_neighlist)

                            else:
                                inter_exclusions_section = self.__writeInterPairs(fsec=fout,inp=inter_contacts,nmol=nmol_list,tag=tag_list,\
                                                                prev_at_count=prev_natoms,atoms_in_mol=self.atoms_in_mol[i],neighlist=add_inter_2_neighlist)
                        else: inter_exclusions_section = []
                    else: status = [fout.write(line) for line in inter_exclusions_section]
                else:
                    status=[fout.write(i+"\n") for i in data_list[0] if i.strip() != ""]
                    
class OpenSMOGXML:
    def __init__(self,xmlfile,coulomb) -> None:
        self.fxml=open(xmlfile,"w+")
        self.fxml.write('<OpenSMOGforces>\n')
        self.nb_count,self.pairs_count,self.dihed_count=0,0,0
        self.add_electrostatics=False
        self.coulomb=coulomb
        if coulomb.P or coulomb.CA or coulomb.CB or coulomb.inter:
            self.add_electrostatics=True
            K_elec,D=coulomb.inv_4pieps/coulomb.caltoj,coulomb.dielec
            T=Tables(); T.__electrostatics__(coulomb=coulomb,r=np.float_([]))
            self.elec_expr="(Kelec/D)*Bk*exp(-inv_dl*r)*q1q2(type1,type2)/r"
            self.elec_const="Bk=%e; inv_dl=%e; D=%d; Kelec=%e"%(T.Bk,T.inv_dl,D,K_elec)
            
    def write_nonbond_xml(self,pairs=[],func=1,C12=[],epsA=[],sig=[],expression="C12 - 2*epsA*(sig/r)^6",params={}):
        self.fxml.write(' <nonbond>\n')
        self.fxml.write('  <nonbond_bytype>\n')
        if len(expression)==0:
            if func==1: expression="C12(type1,type2)/(r^12) - 2*epsA(type1,type2)*(sig(type1,type2)/r)^6"
            elif func==2: expression="C12(type1,type2)/(r^12) - 6*epsA(type1,type2)*(sig(type1,type2)/r)^10"
            elif func in (5,6): 
                expression="epsA(type1,type2)*((1+(C12(type1,type2)/(r^12)))*(1-exp(-((r-r0(type1,type2))^2)/(2*(sd(type1,type2)^2))))-1); sd=%e"%sd
        if len(C12)!=0: params["C12"]=C12
        if len(epsA)!=0: params["epsA"]=epsA
        if len(sig)!=0: params["sig"]=sig
        if self.add_electrostatics:
            expression="%s + %s ; %s"%(self.elec_expr,expression,self.elec_const)
            params["q1q2"]=[]
        self.fxml.write('   <expression expr="%s"/>\n'%expression)
        for p in params: self.fxml.write('   <parameter>%s</parameter>\n'%p)
        self.write_nonbond_param_entries(pairs=pairs,params=params)
        self.fxml.write('  </nonbond_bytype>\n')
        self.fxml.write(' </nonbond>\n') 
        self.nb_count+=1
        return

    def write_nonbond_param_entries(self,pairs,params):
        if self.add_electrostatics:
            if "q1q2" not in params:params["q1q2"]=[]
            neg,pos=[],[]
            if self.coulomb.CB: neg,pos=["CBD","CBE"],["CBK","CBR","CBH"]
            if self.coulomb.P:neg+=["P"]
            neg,pos=tuple(neg),tuple(pos)
            for p in pairs:
                if (p[0].startswith(neg) or p[0].endswith(neg)) and "CBP" in p[0]:
                    if (p[1].startswith(neg) or p[1].endswith(neg)) and "CBP" not in p[1]:
                        params["q1q2"].append(1)
                    elif p[1].startswith(pos) or p[1].endswith(pos): params["q1q2"].append(-1)
                    else: params["q1q2"].append(0)
                elif p[0].startswith(pos) or p[0].endswith(pos):
                    if p[1].startswith(pos) or p[1].endswith(pos): params["q1q2"].append(1)
                    elif p[1].startswith(neg) or p[1].endswith(neg): params["q1q2"].append(-1)
                    else: params["q1q2"].append(0)
                else: params["q1q2"].append(0)
        for x in range(len(pairs)):
            self.fxml.write('   <nonbond_param type1="%s" type2="%s"'%tuple(pairs[x]))
            for p in params: self.fxml.write(' %s="%e"'%(p,params[p][x]))
            self.fxml.write('/>\n')
        return

    def write_pairs_xml(self,pairs=[],params={},name="contacts_LJ-10-12",\
                            expression="eps*( 5*((sig/r)^12) - 6*((sig/r)^10) )"):
        if len(pairs)==0: return
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

    def write_dihedrals_xml(self,quads=[],params={},name="CustomDihedrals",\
                            expression="Kd*(1+cos(n*theta-theta0));theta0=theta0_deg*3.141592653589793/180"):
        if len(quads)==0: return
        if self.pairs_count>0:
            self.fxml.write(' </contacts>\n')
            self.pairs_count=0
        if self.dihed_count==0: self.fxml.write(' <dihedrals>\n')
        expression="select(check,V,V_pi);check=floor(1000*sin(angle(p1,p2,p3))*sin(angle(p2,p3,p4))),V_pi=0.0;V="+expression
        self.fxml.write('  <dihedrals_type name="%s">\n'%name)
        self.fxml.write('   <expression expr="%s"/>\n'%expression)
        for p in params: self.fxml.write('   <parameter>%s</parameter>\n'%p)
        I,J,K,L = 1+np.transpose(quads)
        for x in range(quads.shape[0]): 
            self.fxml.write('   <interaction i="%d" j="%d" k="%d" l="%d"'%(I[x],J[x],K[x],L[x]))
            for p in params: self.fxml.write(' %s="%e"'%(p,params[p][x]))
            self.fxml.write('/>\n')
        self.fxml.write('  </dihedrals_type>\n')
        self.dihed_count+=1
        return

    def __del__(self):
        if self.pairs_count>0:self.fxml.write(' </contacts>\n')
        elif self.dihed_count>0: self.fxml.write(' </dihedrals>\n')
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
        self.excl_volume,self.excl_volume_set = dict(),dict()
        self.atomtypes = []
        self.tableb_ndx = 0

    def __write_header(self,fout,combrule) -> None:
            print (">> Writing header section")
            fout.write("\n; Topology file generated by SuBMIT. \n")
            fout.write("\n[ defaults  ]\n")
            fout.write("; nbfunc comb-rule gen-pairs\n")
            fout.write("  1      1         no   \n\n")
            return

    def __write_footer(self,fout) -> None:
            print (">> Writing tail section")
            fout.write("\n%s\n"%("[ system ]"))
            fout.write("%s\n"%(";name"))
            fout.write("  %s\n"%("Macromolecule"))
            fout.write("\n%s\n"%("[ molecules ]"))
            fout.write("%s\n"%(";name    #molec"))
            fout.write("%s\n"%("Macromolecule     1"))
            return

    def __write_protein_atomtypes(self,fout,type,rad,seq,data):
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
        return 
    
    def __write_nucleicacid_atomtypes(self,fout,type,rad,seq,data):
        print (">> Writing atomtypes section")
        #1:CA model or 2:CA+CB model
        fout.write('%s\n'%("[ atomtypes ]"))
        fout.write(6*"%s".ljust(5)%("; name","mass","charge","ptype","C6(or C10)","C12"))
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
                                fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,bead[0]))
        return

    def __write_protein_nonbondparams(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))

        eps,sig = data.Interactions(nonbond=self.opt.nonbond)
        pairs,repul_C12 = [],[]
        self.atomtypes = []

        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))

        if len(data.CA_atn) > 0:
            cmap_func=self.cmap["prot"].nbfunc
            #if excl_rule == 2 and type == 2:
            for x in self.excl_volume:
                if x.startswith(("CA","CB")):
                    for y in self.excl_volume:
                        if y.startswith(("CA","CB")):
                            func=1
                            if excl_rule==1: C12 = self.fconst.Kr_prot*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                            elif excl_rule==2: C12=self.fconst.Kr_prot*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                            values = 0,C12
                            if cmap_func in (5,6,7):
                                assert self.opt.opensmog
                                func,sd = 6,0.05
                                values = 0,0,sd,C12
                            p = [x,y]; p.sort(); p=tuple(p)
                            if p not in pairs:
                                if p in eps: continue
                                pairs.append(p)
                                repul_C12.append(C12) 
                                if self.opt.opensmog: continue
                                fout.write(" %s %s\t%d\t"%(p[0].ljust(5),p[1].ljust(5),func))
                                fout.write(len(values)*" %e"%tuple(values)+"\n")

        if len(eps)>0:
            fout.write("; Custom Nnobond interactions\n")
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
                elif cmap_func in (5,6,7):
                    assert self.opt.opensmog
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
            params={"eps_att":[],"sig":[],"C12":repul_C12}
            for p in pairs: 
                if p in eps: params["eps_att"].append(eps[p])
                else: params["eps_att"].append(1)
                if p in sig: params["sig"].append(sig[p])
                else: params["sig"].append(0)
            assert len(params["eps_att"])==len(params["C12"])
            if cmap_func==1: expr="C12 - 2*epsA*(sig/r)^6"
            elif cmap_func==2: expr="C12 - 6*epsA*(sig/r)^10"
            elif cmap_func in (5,6,7):expr="epsA*((1+(C12/(r^12)))*(1-exp(-((r-r0)^2)/(2*(sd^2))))-1); sd=%e"%sd
            if len(data.CA_atn)!=0:
                self.prot_xmlfile.write_nonbond_xml(func=cmap_func,pairs=pairs,expression=expr, \
                        C12=params["C12"],epsA=params["eps_att"],sig=params["sig"])
            if len(data.P_atn)!=0: 
                self.nucl_xmlfile.write_nonbond_xml(func=cmap_func,pairs=pairs,expression=expr, \
                        C12=params["C12"],epsA=params["eps_att"],sig=params["sig"])
        return 0

    def __write_nucleicacid_nonbondparams(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))

        eps,sig = data.Interactions(nonbond=self.opt.nonbond)
        pairs,repul_C12 = [],[]

        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))

        if len(data.P_atn) > 0:
            cmap_func=self.cmap["nucl"].nbfunc
            #if excl_rule == 2 and type in (3,5):
            for x in self.excl_volume:
                if x.startswith(("P","S","B")):
                    for y in self.excl_volume:
                        if y.startswith(("P","S","B")):
                            func=1
                            if excl_rule==1: C12 = self.fconst.Kr_nucl*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                            elif excl_rule==2: C12=self.fconst.Kr_nucl*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12
                            values = 0,C12
                            if cmap_func in (5,6,7):
                                assert self.opt.opensmog
                                func,sd = 6,0.05
                                values = 0,0,sd,C12
                            p = [x,y]; p.sort(); p=tuple(p)
                            if p not in pairs:
                                if p in eps: continue
                                pairs.append(p)
                                repul_C12.append(C12) 
                                if self.opt.opensmog: continue
                                fout.write(" %s %s\t%d\t"%(p[0].ljust(5),p[1].ljust(5),func))
                                fout.write(len(values)*" %e"%tuple(values)+"\n")

        if len(eps)>0:
            fout.write("; Custom Nnobond interactions\n")
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
                    if excl_rule==1: C12 = eps[p]*(((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif excl_rule==2: C12 = eps[p]*((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = 0.0,C12
                elif cmap_func==1: values = 2*eps[p]*((sig[p])**6),1*eps[p]*((sig[p])**12)
                elif cmap_func==2: values = 6*eps[p]*((sig[p])**10),5*eps[p]*((sig[p])**12)
                elif cmap_func in (5,6,7):
                    assert self.opt.opensmog
                    func,sd = 6,0.05
                    if excl_rule==1: C12 = (((self.excl_volume[x]**12)*(self.excl_volume[y]**12))**0.5)
                    elif excl_rule==2: C12 = ((self.excl_volume[x]+self.excl_volume[y])/2.0)**12                
                    values = eps[p],sig[p],sd,C12
                repul_C12.append(values[-1])
                if self.opt.opensmog: continue
                fout.write(" %5s %5s\t%d\t"%(p[0],p[1],func))
                fout.write(len(values)*" %e"%tuple(values))
                fout.write("\n")

        if self.opt.opensmog and len(repul_C12)!=0:
            params={"eps_att":[],"sig":[],"C12":repul_C12}
            for p in pairs: 
                if p in eps: params["eps_att"].append(eps[p])
                else: params["eps_att"].append(1)
                if p in sig: params["sig"].append(sig[p])
                else: params["sig"].append(0)
            assert len(params["eps_att"])==len(params["C12"])
            if cmap_func==1: expr="C12 - 2*epsA*(sig/r)^6"
            elif cmap_func==2: expr="C12 - 6*epsA*(sig/r)^10"
            elif cmap_func in (5,6,7):expr="epsA*((1+(C12/(r^12)))*(1-exp(-((r-r0)^2)/(2*(sd^2))))-1); sd=%e"%sd
            if len(data.CA_atn)!=0:
                self.prot_xmlfile.write_nonbond_xml(func=cmap_func,pairs=pairs,expression=expr, \
                        C12=params["C12"],epsA=params["eps_att"],sig=params["sig"])
            if len(data.P_atn)!=0: 
                self.nucl_xmlfile.write_nonbond_xml(func=cmap_func,pairs=pairs,expression=expr, \
                        C12=params["C12"],epsA=params["eps_att"],sig=params["sig"])
        return 0

    def __write_moleculetype(self,fout):
        print (">> Writing moleculetype section")
        fout.write("\n%s\n"%("[ moleculetype ]"))
        fout.write("%s\n"%("; name            nrexcl"))
        fout.write("%s\n"%("  Macromolecule   3"))

    def __write_protein_atoms(self,fout,type,cgfile,seq,inc_charge):
        print (">> Writing atoms section")
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = dict()
        if inc_charge: 
            Q.update({x:1 for x in ["CBK","CBR","CBH"]})
            Q.update({x:-1 for x in ["CBD","CBE"]})

        prev_resnum,seqcount,rescount="",0,0
        assert ".prot." in cgfile
        seq=["_%s_"%x for x in seq.split()]
        self.atomtypes = []
        with open(cgfile) as fin:
            for line in fin:
                if line.startswith("ATOM"):
                    atnum=hy36decode(5,line[6:11])
                    atname=line[12:16].strip()
                    resname=line[17:20].strip()
                    resnum=hy36decode(4,line[22:26])
                    atype=atname
                    if resnum !=prev_resnum: prev_resnum,rescount=resnum,1+rescount
                    if atype=="CB": atype+=seq[seqcount][rescount]
                    if atype not in Q: Q[atype] = 0
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
                elif line.startswith("TER"): seqcount,rescount=1+seqcount,0
        return

    def __write_nucleicacid_atoms(self,fout,type,cgfile,seq,inc_charge):
        print (">> Writing atoms section")
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = {"P":-1}

        prev_resnum,seqcount,rescount="",0,0
        assert ".nucl." in cgfile
        seq=["5%s3"%x for x in seq.split()]
        self.atomtypes = []
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
                    if atype.endswith("'"): atype="S"+codon
                    elif atype.startswith("N"): atype="B"+codon
                    if atype not in Q: Q[atype] = 0
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
                elif line.startswith("TER"): seqcount,rescount=1+seqcount,0
        return

    def __write_protein_bonds(self,fout,data,func):
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

    def __write_protein_angles(self,fout,data):
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

    def __write_protein_dihedrals(self,fout,data,chiral):
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

        data.Impropers()
        func = 1
        for c in range(len(data.bb_dihedrals)):
            quads,diheds = data.bb_dihedrals[c]
            #if self.opt.opensmog:
            #    self.prot_xmlfile.write_dihedrals_xml(quads=quads,name="bb_dihedrals%d_n1"%c,\
            #                            expression="Kd*(1-cos(theta-theta0)) + (Kd/f)*(1-cos(3*(theta-theta0)));theta0=theta0_deg*3.141592653589793/180",\
            #                            params={"theta0_deg":diheds,"Kd":Kd_bb*np.ones(quads.shape[0]),"fc":mfac*np.ones(quads.shape[0])})
            #    continue
            I,J,K,L = 1+np.transpose(quads)
            diheds += phase
            for x in range(quads.shape[0]):
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_bb,1))
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_bb/mfac,3))
        if chiral and len(data.CB_atn) != 0:
            func = 2
            fout.write("; %5s %5s %5s %5s %5s %5s %5s \n" % (";ai","aj","ak","al","func","phi0(deg)","Kd"))
            for c in range(len(data.sc_dihedrals)):
                quads,diheds = data.sc_dihedrals[c]
                #if self.opt.opensmog:
                #    self.prot_xmlfile.write_dihedrals_xml(quads=quads,name="sc_dihedrals%d_n1"%c,\
                #                        expression="Kd*((theta-theta0)^2);theta0=theta0_deg*3.141592653589793/180",\
                #                        params={"theta0_deg":diheds,"Kd":Kd_sc*np.ones(quads.shape[0])})
                #    continue
                I,J,K,L = 1+np.transpose(quads)
                for x in range(quads.shape[0]):fout.write(" %5d %5d %5d %5d %5d %e %e\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_sc))
        return

    def __write_protein_pairs(self,fout,data,excl_rule,charge):
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
        elif cmap.func in (5,6,7):
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
                if cmap.func==6:
                    if self.opt.opensmog:
                        self.prot_xmlfile.write_pairs_xml(
                            pairs=pairs,params={"r0":dist,"eps":eps,"C12":c12},\
                            name="contacts%d_Gaussian-12"%c,\
                            expression="eps*((1+(C12/(r^12)))*(1-exp(-((r-r0)^2)/(2*(sd^2))))-1); sd=%e"%sd)
                        continue
                    I,J = 1+np.transpose(pairs)
                    for x in range(pairs.shape[0]): 
                        fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(I[x],J[x],func,eps[x],dist[x],sd,c12[x]))
                elif cmap.func==7:
                    if self.opt.opensmog:
                        N_dist_values=dist.shape[1]
                        expr="(1+(C12/(r^12)))"
                        for i in range(N_dist_values): expr+="*(1-G%d)"%i
                        expr="eps*(%s - 1)"%expr
                        for i in range(N_dist_values):
                            expr="%s;G%d=exp(-((r-r0%d)^2)/(2*(sd%d^2)))"%(expr,i,i,i)
                        params={"eps":eps,"C12":c12}
                        dist=np.transpose(dist)
                        for i in range(N_dist_values):
                            params["r0%d"%i]=dist[i]
                            params["sd%d"%i]=sd*np.ones(len(dist[i]))
                        self.prot_xmlfile.write_pairs_xml(pairs=pairs,params=params,expression=expr,\
                                                    name="contacts%d_%dGaussian-12"%(c,N_dist_values))
                        continue
                    I,J = 1+np.transpose(pairs)
                    for x in range(pairs.shape[0]):
                        assert len(dist[x])<=2, "Error GROMACS 4.5.4 SBM version supports maximim two distances for Gaussians"
                        if len(dist[x])==1:func=6
                        else: func=7
                        fout.write(" %5d %5d %5d %.3f"%(I[x],J[x],func,eps[x]))
                        for r0 in dist[x]: fout.write(" %e %e"%(r0,sd))
                        fout.write(" %e\n"%c12[x])
        return 

    def __write_nucleicacid_bonds(self,fout,data,func):
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

    def __write_nucleicacid_angles(self,fout,data):
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

    def __write_nucleicacid_dihedrals(self,fout,data,chiral):
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

        data.Impropers()

        func = 1
        for c in range(len(data.bb_dihedrals)):
            quads,diheds = data.bb_dihedrals[c]
            if self.opt.P_stretch: diheds=180*np.ones(diheds.shape)
            #if self.opt.opensmog:
            #    self.nucl_xmlfile.write_dihedrals_xml(quads=quads,name="bb_dihedrals%d_n1"%c,\
            #                            expression="Kd*(1-cos(theta-theta0)) + (Kd/f)*(1-cos(3*(theta-theta0)));theta0=theta0_deg*3.141592653589793/180",\
            #                            params={"theta0_deg":diheds,"Kd":Kd_bb*np.ones(quads.shape[0]),"fc":mfac*np.ones(quads.shape[0])})
            #    continue
            I,J,K,L = 1+np.transpose(quads)
            diheds += phase
            for x in range(quads.shape[0]):
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_bb,1))
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_bb/mfac,3))
        if len(data.S_atn):
            for c in range(len(data.sc_dihedrals)):
                quads,diheds = data.sc_dihedrals[c]
                #if self.opt.opensmog:
                #   self.nucl_xmlfile.write_dihedrals_xml(quads=quads,name="sc_dihedrals%d_n1"%c,\
                #                        expression="Kd*(1-cos(theta-theta0)) + (Kd/f)*(1-cos(3*(theta-theta0)));theta0=theta0_deg*3.141592653589793/180",\
                #                        params={"theta0_deg":diheds,"Kd":Kd_sc*np.ones(quads.shape[0]),"fc":mfac*np.ones(quads.shape[0])})
                #   continue
                I,J,K,L = 1+np.transpose(quads)
                diheds += phase
                for x in range(quads.shape[0]):
                    fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,diheds[x],Kd_sc,1))
                    fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(I[x],J[x],K[x],L[x],func,3*diheds[x],Kd_sc/mfac,3))
        return

    def __write_nucleicacid_pairs(self,fout,data,excl_rule,charge):
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
        elif cmap.func in (5,6,7):
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
                if cmap.func==6:
                    if self.opt.opensmog:
                        self.nucl_xmlfile.write_pairs_xml(
                                pairs=pairs,params={"r0":dist,"eps":eps,"C12":c12},\
                                name="contacts%d_Gaussian-12"%c,\
                                expression="eps*((1+(C12/(r^12)))*(1-exp(-((r-r0)^2)/(2*(sd^2))))-1); sd=%e"%sd)
                    I,J = 1+np.transpose(pairs)
                    for x in range(pairs.shape[0]): 
                        fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(I[x],J[x],func,eps[x],dist[x],sd,c12[x]))
                elif cmap.func==7:
                    if self.opt.opensmog:
                        N_dist_values=dist.shape[1]
                        expr="(1+(C12/(r^12)))"
                        for i in range(N_dist_values): expr+="*(1-G%d)"%i
                        expr="eps*(%s - 1)"%expr
                        for i in range(N_dist_values):
                            expr="%s;G%d=exp(-((r-r0%d)^2)/(2*(sd%d^2)))"%(expr,i,i,i)
                        params={"eps":eps,"C12":c12}
                        dist=np.transpose(dist)
                        for i in range(N_dist_values):
                            params["r0%d"%i]=dist[i]
                            params["sd%d"%i]=sd*np.ones(len(dist[i]))
                        self.prot_xmlfile.write_pairs_xml(pairs=pairs,params=params,expression=expr,\
                                                    name="contacts%d_%dGaussian-12"%(c,N_dist_values))
                        continue
                    I,J = 1+np.transpose(pairs)
                    for x in range(pairs.shape[0]):
                        assert len(dist[x])<=2, "Error GROMACS 4.5.4 SBM version supports maximim two distances for Gaussians"
                        if len(dist[x])==1:func=6
                        else: func=7
                        fout.write(" %5d %5d %5d %.3f"%(I[x],J[x],func,eps[x]))
                        for r0 in dist[x]: fout.write(" %e %e"%(r0,sd))
                        fout.write(" %e\n"%c12[x])

        return 

    def __write_exclusions(self,fout,data):
        print (">> Writing exclusions section")
        fout.write("\n%s\n"%("[ exclusions ]"))
        fout.write("; %5s %5s\n"%("i","j"))
        for pairs,chains,dist,eps in data.contacts:
            I,J = 1+np.transpose(pairs)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d\n"%(I[x],J[x]))
        return

    def __next(self):
        self.excl_volume_set.update(self.excl_volume.copy())
        self.excl_volume = dict()
        self.atomtypes = []

    def write_topfile(self,outtop,excl,charge,bond_function,CBchiral,rad):
        nfiles=len(self.allatomdata)
        Nmol,Data = self.Nmol,list()
        total_mol_p,total_mol_n= sum(Nmol["prot"]),sum(Nmol["nucl"])
        for i in range(nfiles):
            cgpdb = PDB_IO()
            fileindex=[str(i),str()][int(nfiles==1)]
            Data.append(Preprocess(aa_pdb=self.allatomdata[i],pdbindex=fileindex))
            if len(self.allatomdata[i].prot.lines) > 0 and self.CGlevel["prot"] in (1,2):
                if self.CGlevel["prot"]==1: cgpdb.loadfile(infile=self.allatomdata[i].prot.bb_file,refine=False)
                elif self.CGlevel["prot"]==2: cgpdb.loadfile(infile=self.allatomdata[i].prot.sc_file,refine=False)
                prot_topfile = "prot%s_%s"%(fileindex,outtop)
                if self.opt.opensmog: self.prot_xmlfile=OpenSMOGXML(xmlfile="prot%s_%s"%(fileindex,self.opt.xmlfile),coulomb=charge)
                with open(prot_topfile,"w+") as ftop:
                    print (">>> writing Protein GROMACS toptology", prot_topfile)
                    proc_data_p = Preprocess(aa_pdb=self.allatomdata[i],pdbindex=fileindex)
                    proc_data_p.processData(data=cgpdb.prot)
                    self.__write_header(fout=ftop,combrule=excl)
                    self.__write_protein_atomtypes(fout=ftop,data=proc_data_p,type=self.CGlevel["prot"],seq=cgpdb.prot.seq,rad=rad)
                    self.__write_protein_nonbondparams(fout=ftop,data=proc_data_p,type=self.CGlevel["prot"],excl_rule=excl)
                    self.__write_moleculetype(fout=ftop)
                    self.__write_protein_atoms(fout=ftop,type=self.CGlevel["prot"],cgfile=cgpdb.pdbfile,seq=cgpdb.prot.seq,inc_charge=(charge.CB or charge.CA)*(not self.opt.opensmog))
                    self.__write_protein_pairs(fout=ftop, data=proc_data_p,excl_rule=excl,charge=charge)
                    self.__write_protein_bonds(fout=ftop, data=proc_data_p,func=bond_function)
                    self.__write_protein_angles(fout=ftop, data=proc_data_p)
                    self.__write_protein_dihedrals(fout=ftop, data=proc_data_p,chiral=CBchiral)
                    self.__write_exclusions(fout=ftop,data=proc_data_p)
                    self.__write_footer(fout=ftop)
                    Data[i].CA_atn,Data[i].CB_atn=proc_data_p.CA_atn,proc_data_p.CB_atn
                    Data[i].cgpdb_p=proc_data_p.cgpdb
                if self.opt.opensmog: del self.prot_xmlfile
            if len(self.allatomdata[i].nucl.lines) > 0 and self.CGlevel["nucl"] in (1,3,5):
                if self.CGlevel["nucl"]==1: cgpdb.loadfile(infile=self.allatomdata[i].nucl.bb_file,refine=False)
                elif self.CGlevel["nucl"] in (3,5): cgpdb.loadfile(infile=self.allatomdata[i].nucl.sc_file,refine=False)
                nucl_topfile = "nucl%s_%s"%(fileindex,outtop)
                if self.opt.opensmog: self.nucl_xmlfile=OpenSMOGXML(xmlfile="nucl%s_%s"%(fileindex,self.opt.xmlfile),coulomb=charge)
                with open(nucl_topfile,"w+") as ftop:
                    print (">>> writing RNA/DNA GROMACS toptology", nucl_topfile)
                    proc_data_n = Preprocess(aa_pdb=self.allatomdata[i],pdbindex=fileindex)
                    proc_data_n.processData(data=cgpdb.nucl)
                    self.__write_header(fout=ftop,combrule=excl)
                    self.__write_nucleicacid_atomtypes(fout=ftop,type=self.CGlevel["nucl"],data=proc_data_n,seq=cgpdb.nucl.seq,rad=rad)
                    self.__write_nucleicacid_nonbondparams(fout=ftop,data=proc_data_n,type=self.CGlevel["nucl"],excl_rule=excl)
                    self.__write_moleculetype(fout=ftop)
                    self.__write_nucleicacid_atoms(fout=ftop,type=self.CGlevel["nucl"],cgfile=cgpdb.pdbfile,seq=cgpdb.nucl.seq,inc_charge=charge.P*(not self.opt.opensmog))
                    self.__write_nucleicacid_pairs(fout=ftop, data=proc_data_n,excl_rule=excl,charge=charge)
                    self.__write_nucleicacid_bonds(fout=ftop, data=proc_data_n,func=bond_function)
                    self.__write_nucleicacid_angles(fout=ftop, data=proc_data_n)
                    self.__write_nucleicacid_dihedrals(fout=ftop, data=proc_data_n,chiral=CBchiral)
                    self.__write_exclusions(fout=ftop,data=proc_data_n)
                    self.__write_footer(fout=ftop)
                    Data[i].P_atn,Data[i].S_atn,Data[i].B_atn=proc_data_n.P_atn,proc_data_n.S_atn,proc_data_n.B_atn
                    Data[i].cgpdb_n=proc_data_n.cgpdb
                if self.opt.opensmog: del self.nucl_xmlfile
            self.__next()
        if sum(Nmol["prot"])==0: self.CGlevel["prot"],self.cmap["prot"].func=0,-1
        if sum(Nmol["nucl"])==0: self.CGlevel["nucl"],self.cmap["nucl"].func=0,-1
        
        #if sum(Nmol["prot"])+sum(Nmol["nucl"])>1:
        merge=MergeTop(proc_data=Data,Nprot=Nmol["prot"],Nnucl=Nmol["nucl"],topfile=outtop,opt=self.opt, \
                coul=charge,excl_volume=self.excl_volume_set,excl_rule=excl,fconst=self.fconst,cmap=self.cmap)

        if self.opt.opensmog: return #don't write table
        table = Tables()
        if self.cmap["prot"].func == 2 or self.cmap["nucl"].func == 2 or self.cmap["inter"].func == 2 :
            table.write_pair_table(coulomb=charge,ljtype=2)
        if self.cmap["prot"].func == 1 or self.cmap["nucl"].func == 1 or self.cmap["inter"].func == 1:
            if charge.debye: table.write_pair_table(coulomb=charge,ljtype=1)

        return 0

class Clementi2000(Topology):
    def __del__(self):
        pass

class Banerjee2023(Clementi2000):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume,self.excl_volume_set = dict(),dict()
        self.atomtypes = []
        self.tableb_ndx = 0
        self.pdb_counter = 0

    def __write_inter_pairs_file__(self):
        pass

    def __next(self):
        self.pdb_counter+=1
        self.excl_volume_set.update(self.excl_volume.copy())
        if self.pdb_counter==2:
            self.__write_inter_pairs_file__()
            exit()
        self.excl_volume = dict()
        self.atomtypes = []

class Pal2019(Topology):
    def __write_nucleicacid_pairs(self,fout,data,excl_rule,charge):
        print (">> Writing pairs section")
        cmap = self.cmap["nucl"]
        fout.write("\n%s\n"%("[ pairs ]"))
        assert cmap.func==2
        print ("> Using LJ C10-C12 for Stackubg. Note: Require Table file(s)")
        fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
        func = 1
        epsmat,stack = data.Interactions(pairs=True)

        B_atn = {data.B_atn[c][r]:self.atomtypes[data.B_atn[c][r]] for c in data.B_atn for r in data.B_atn[c]}
        for p in stack:stack[p]=stack[p][0] #multiple distances not supported
        for c in data.B_atn:
            resnum = list(data.B_atn[c].keys())
            resnum.sort()
            pairs = np.int_([(data.B_atn[c][x],data.B_atn[c][x+1]) for x in resnum if x+1 in data.B_atn[c]])
            if len(pairs)==0: continue
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

class Denesyuk2013_Chakraborty2018(Topology):
    def __get_atn_typeletter__(self,data):
        P_atn={data.P_atn[c][r]:self.atomtypes[data.P_atn[c][r]] for c in data.P_atn for r in data.P_atn[c]}
        S_atn={data.S_atn[c][r]:self.atomtypes[data.S_atn[c][r]][0] for c in data.S_atn for r in data.S_atn[c]}
        B_atn={data.B_atn[c][r]:self.atomtypes[data.B_atn[c][r]] for c in data.B_atn for r in data.B_atn[c]}
        all_atn={}
        for k in P_atn: assert P_atn[k]=="P"
        for k in S_atn: assert S_atn[k]=="S"
        for k in B_atn: 
            assert B_atn[k][0]=="B"
            B_atn[k]=[x for x in B_atn[k][2:] if x.isupper()][0]
        return [P_atn,S_atn,B_atn]

    def __get_force_constants__(self,type,deoxy):
        if not deoxy: #Denesyuk2013
            Kb={("P","S"):23.00, ("S","P"):64.00, \
                ("S","A"):10.00, ("S","G"):10.00, \
                ("S","U"):10.00, ("S","C"):10.00}
            Ka={("P","S","P"):20.00, ("S","P",'S'):20.00, ("P","S","A"):05.00, \
                ("A","S","P"):05.00, ("G","S","P"):05.00, ("P","S","G"):05.00, \
                ("P","S","U"):05.00, ("P","S","C"):05.00, \
                ("U","S","P"):05.00, ("C","S","P"):05.00 }
        elif deoxy:#Chakraborty2018
            Kb={("P","S"):17.63, ("S","P"):62.59, \
                ("S","A"):44.31, ("S","G"):48.98, \
                ("S","T"):46.56, ("S","C"):43.25}
            Ka={("P","S","P"):25.67, ("S","P",'S'):67.50, ("P","S","A"):29.53, \
                ("A","S","P"):67.32, ("G","S","P"):62.94, ("P","S","G"):26.28, \
                ("P","S","T"):39.56, ("P","S","C"):35.25, \
                ("T","S","P"):93.99, ("C","S","P"):77.78 }

        #Kcal/(mol A2) to KJ/(mol nm2)
        for k in Kb: Kb[k]*=(100*self.fconst.caltoj) 
        #Kcal/(mol rad2) to KJ/(mol rad2)
        for k in Ka: Ka[k]*=self.fconst.caltoj

        if type in ("bonds","bond","b"): return Kb.copy()
        elif type in ("angles","angle","a"): return Ka.copy()

    def __write_nucleicacid_bonds(self,fout,data,func):
        print (">> Writing bonds section")
        #GROMACS IMPLEMENTS Ebonds = (Kx/2)*(r-r0)^2
        #Input units KJ mol-1 A-2 GROMACS units KJ mol-1 nm-1 (100 times the input value) 

        all_atn=dict()
        for x in self.__get_atn_typeletter__(data=data): all_atn.update(x) 


        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "r0(nm)", "Kb"))


        #TIS force bond force constants for RNA or DNA 
        deoxy=data.allatpdb.nucl.deoxy
        data.Bonds()
        for c in range(len(data.bonds)):
            pairs,dist=data.bonds[c]
            Kb=self.__get_force_constants__(type="bonds",deoxy=deoxy[c])
            I,J = 1+np.transpose(pairs)
            pairs = [(all_atn[pairs[x][0]],all_atn[pairs[x][1]]) for x in range(pairs.shape[0])]
            for x in range(len(pairs)): 
                fout.write(" %5d %5d %5d %e %e"%(I[x],J[x],func,dist[x],Kb[pairs[x]]))
                fout.write("; %s-%s\n"%pairs[x])
        return 

    def __write_nucleicacid_angles(self,fout,data):
        print (">> Writing angless section")
        #V_ang = (Ktheta/2)*(r-r0)^2
        #Input units KJ mol-1 #GROMACS units KJ mol-1 

        all_atn=dict()
        for x in self.__get_atn_typeletter__(data=data): all_atn.update(x) 
        #TIS force bond force constants

        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))

        func = 1
        #TIS force bond force constants for RNA or DNA 
        deoxy=data.allatpdb.nucl.deoxy
        data.Angles()
        for c in range(len(data.angles)):
            triplets,angles=data.angles[c]
            Ka=self.__get_force_constants__(type="angles",deoxy=deoxy[c])
            I,J,K = 1+np.transpose(triplets)
            triplets=[(all_atn[triplets[x][0]],all_atn[triplets[x][1]],all_atn[triplets[x][2]]) \
                                                            for x in range(triplets.shape[0])]
            for x in range(len(triplets)): 
                fout.write(" %5d %5d %5d %5d %e %e"%(I[x],J[x],K[x],func,angles[x],Ka[triplets[x]]))
                fout.write("; %s-%s-%s\n"%triplets[x])
        return

    def __write_nucleicacid_pairs(self,fout,data,excl_rule,charge):
        print (">> Writing stacking interactions section to contscts")
        cmap = self.cmap["nucl"]
        
        all_atn=self.__get_atn_typeletter__(data=data)
        data.Pairs(cmap=cmap,group="nucl")

        opensmog_dlp_fork=False
        print ("NOTE: TIS stacking interactions includes B1-S1-P2-S2 and S1-P2-S2-B2 torsions, requiring 5-particle function which is not supported in current official version of OpenSMOG.")
        print ("A fork of OpenSMOGv1.1.1 with N-particle interaction support is available here: ")
        opensmog_dlp_fork=bool(int("0"+input("\n\t 0) Implement B1-S1-S2-B2 torsion instead.\n\t 1) Use default TIS implementation with unofficial fork of OpenSMOG.\nSelect an options (0/1, default:0: ")))
        print ("Use OpenSMOG fork: ",opensmog_dlp_fork)
        if opensmog_dlp_fork: data.Dihedrals()
        else: data.Impropers()        

        assert self.opt.opensmog
        for c in data.B_atn:
            resnum = list(data.B_atn[c].keys())
            resnum.sort()
            stack = {(data.B_atn[c][x],data.B_atn[c][x+1]):[] for x in resnum if x+1 in data.B_atn[c]}
            for pairs,chains,sig,eps in data.contacts:
                for x in range(pairs.shape[0]):
                    if tuple(pairs[x]) in stack:
                        stack[tuple(pairs[x])].append(sig[x])
            phi1 = {k[0]:0.0 for k in stack}
            phi2 = {k[1]:0.0 for k in stack}
            grp1,grp2 = phi1.copy(),phi2.copy()
            for quads,torsions in data.sc_dihedrals:
                for x in range(quads.shape[0]):
                    if quads[x][0] in phi1:
                        phi1[quads[x][0]]=torsions[x]
                        grp1[quads[x][0]]=quads[x]
                    if quads[x][3] in phi2:
                        phi2[quads[x][3]]=torsions[x]
                        grp2[quads[x][3]]=quads[x]
            group=stack.copy()
            for k1,k2 in stack:
                stack[(k1,k2)]+=[phi1[k1],phi2[k2]]
                group[(k1,k2)]=[grp1[k1],grp2[k2]]

            quads1,quads2,values=[],[],[]
            for k in stack: 
                quads1.append(group[k][0])
                quads2.append(group[k][1])
                values.append(stack[k])

            Kl,Kd=1.45,3.00
            if not opensmog_dlp_fork:
                quads1=np.int_(quads1)
                values=np.transpose(values)
                self.nucl_xmlfile.write_dihedrals_xml(quads=quads1,name="base_stacking%d_"%c,\
                                params={"r0":values[0],"theta0":values[1]},\
                                 expression="eps/(Kl*(r-r0)^2 + Kd*(theta-theta0)^2;Kd=%d'Kl=%d"%(Kl,Kd))

        return 

class Reddy2017(Denesyuk2013_Chakraborty2018):
    def __init__(self,allatomdata,fconst,CGlevel,Nmol,cmap,opt) -> None:
        self.allatomdata = allatomdata
        #for data in range(len(allatomdata)) self.__check_H_atom__(data)
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume,self.excl_volume_set = dict(),dict()
        self.atomtypes = []
        self.tableb_ndx = 0

    def __check_H_atom__(self,data):
        # checking presence of H-atom. Important for adding GLY CB
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

    def __write_protein_nonbondparams(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        assert self.cmap["prot"].nbfunc==1
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
            #self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"excl_r1":excl_rad1,"excl_r2":excl_rad2},\
            #        expression='Krep*(sig/r)^6;sig=0.5*(excl_r1(type1,type2)+excl_r2(type1,type2));Krep=%e'%self.fconst.Kr_prot)
            sig=[0.5*(excl_rad1[i]+excl_rad2[i]) for i in range(len(pairs))]
            Krep=(-0.5)*self.fconst.Kr_prot*np.ones(len(pairs))
            repul_12=np.zeros(len(pairs))
            self.prot_xmlfile.write_nonbond_xml(func=1,pairs=pairs,C12=repul_12,epsA=Krep,sig=sig)
        return 0

    def __write_protein_bonds(self,fout,data,func):
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
        data.Bonds()
        table_idx = dict()
        if self.opt.opensmog:
            fout.write(";%5s %5s %5s %5s %5s; for excl\n"%("ai", "aj", "func", "r0", "Kb=0.0"))
            for c in range(len(data.bonds)):
                pairs,dist=data.bonds[c]
                I,J = 1+np.transpose(pairs) 
                print (">Writing chain %d Bonds as OpenSMOG contacts"%c)
                self.prot_xmlfile.write_pairs_xml( pairs=pairs,params={"r0":dist},\
                            name="FENE_bonds%d_R=0.2"%c,\
                            expression="-(K/2)*(R^2)*log(1-((r-r0)/R)^2); R=%.2f; K=%e"%(R,K))
                for i in range(pairs.shape[0]): 
                    r0 = np.round(dist[i],3)
                    fout.write(" %5d %5d %5d %.3f 0.0; dummy_entry\n"%(I[i],J[i],1,r0))
            return
        #else:
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "table_no.", "Kb"))
        for c in range(len(data.bonds)):
            pairs,dist=data.bonds[c]
            I,J = 1+np.transpose(pairs) 
            for i in range(pairs.shape[0]): 
                r0 = np.round(dist[i],3)
                if r0 not in table_idx: table_idx[r0]=len(table_idx)+self.tableb_ndx
                if r0-R>0:r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                else: r=0.001*np.int_(range(int(1000*(0+0.001)),int(1000*(r0+R-0.001))))
                V = -0.5*(R**2)*np.log(1-((r-r0)/R)**2)
                #V_1 = -0.5*(R**2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                V_1 = (R**2)*(r-r0)/(R**2-(r-r0)**2)
                Tables().write_bond_table(X=r,index=table_idx[r0],V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e; d=%.3f\n"%(I[i],J[i],func,table_idx[r0],K,r0))
        self.tableb_ndx=max(table_idx.values())
        return 

    def __write_protein_angles(self,fout,data):
        print (">> Not Writing angless section")
        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))
        return

    def __write_protein_dihedrals(self,fout,data,chiral):
        print (">> Not Writing dihedrals section")
        fout.write("\n%s\n"%("[ dihedrals ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s %5s %5s\n" % (";ai","aj","ak","al","func","phi0(deg)","Kd","mult"))
        return

    def __write_protein_pairs(self,fout,data,excl_rule,charge):
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
                continue
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
            #data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
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

class Baul2019(Reddy2017):
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
        self.excl_volume,self.excl_volume_set = dict(),dict()
        self.atomtypes = []
        self.tableb_ndx = 0

    def __write_protein_nonbondparams(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        assert self.cmap["prot"].nbfunc==1
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
        
        if self.opt.opensmog and len(pairs)!=0:
            excl_rad1,excl_rad2=np.transpose([(self.excl_volume[x],self.excl_volume[y]) for x,y in pairs])
            epsmat=[epsmat[p] for p in pairs]
            eps=[eps[(x[:2],y[:2])] for x,y in pairs]
            #self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"eps":eps,"f":epsmat,"r1":excl_rad1,"r2":excl_rad2},\
                                            #expression='eps(type1,type2)*f(type1,type2)*((s/r)^12 - 2*(s/r)^6); s=0.5*(r1(type1,type2)+r2(type1,type2))')
            sig=[0.5*(excl_rad1[i]+excl_rad2[i]) for i in range(len(pairs))]
            eps=[eps[i]*epsmat[i] for i in range(len(pairs))]
            repul_12=[1*eps[i]*(sig[i])**12 for i in range(len(pairs))]
            self.prot_xmlfile.write_nonbond_xml(func=1,pairs=pairs,C12=repul_12,epsA=eps,sig=sig)

        return 0  

    def __write_protein_pairs(self,fout,data,excl_rule,charge):
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
    
        pairs=[]
        for c in data.CA_atn:
            resnum = list(data.CA_atn[c].keys())
            resnum.sort()
            for x in resnum:
                if x+2 in data.CA_atn[c]:pairs.append((data.CA_atn[c][x],data.CA_atn[c][x+2]))
                for y in (x+1,x+2):
                    if y in data.CB_atn[c]:pairs.append((data.CA_atn[c][x],data.CB_atn[c][y]))
                    if x in data.CB_atn[c]:
                        if y in data.CA_atn[c]:
                            pairs.append((data.CB_atn[c][x],data.CA_atn[c][y]))
                        if x in data.CB_atn[c] and y in data.CB_atn[c]:
                            pairs.append((data.CB_atn[c][x],data.CB_atn[c][y]))
        
        if len(pairs)==0: return
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
        #data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
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
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.Nmol = Nmol
        self.cmap = {"prot":cmap[0],"nucl":cmap[1],"inter":cmap[2]}
        self.opt = opt
        self.eps_idr_bbbb = 0.12*self.fconst.caltoj 
        self.eps_idr_bbsc = 0.24*self.fconst.caltoj 
        self.eps_idr_scsc = 0.18*self.fconst.caltoj 
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume,self.excl_volume_set = dict(),dict()
        self.atomtypes = []
        self.tableb_ndx = 0
        self.bonds = []

    def __write_unfolded_cgpdb__(self,rad,data):
        print ("> Writing unfolded CG-PDB file")
        self.ordered=data.allatpdb.prot
        residues=set(self.idrdata.res+self.ordered.res)
        residues = [x for x in residues if "CA" in x]        
        residues.sort()
        chain,prev_rnum=0,0
        outfasta="unfolded.fa"

        if len(self.ordered.cid)>len(self.idrdata.cid):
            CID=self.ordered.cid
            assert tuple(CID[:len(self.idrdata.cid)])==tuple(self.idrdata.cid)
        else:
            CID=self.idrdata.cid
            assert tuple(CID[:len(self.ordered.cid)])==tuple(self.ordered.cid)

        self.new2old_res,ch_count=dict(),-1
        with open(outfasta,"w+") as fout:
            for x in range(len(residues)):
                cnum,rnum,rname,atname = residues[x]
                assert atname=="CA"
                if x==0 or CID[cnum]!=CID[chain] or rnum-prev_rnum not in (0,1):
                    fout.write("\n\n>chain:%s:%s\n"%(CID[cnum],rnum))
                    ch_count+=1
                fout.write(self.idrdata.amino_acid_dict[rname])
                chain = cnum
                self.new2old_res[(ch_count,rnum)]=cnum,rnum
                prev_rnum=rnum
        #get unfolded CG-pdb
        unfolded = PDB_IO()
        unfolded.buildProtIDR(fasta=outfasta,rad=rad)
        self.unfolded_cgpdb=unfolded.prot
        #get unfolded CG-pdb processed data
        unfolded = Preprocess(aa_pdb=self.allatomdata)
        unfolded.processData(data=self.unfolded_cgpdb)
        self.unfolded_data=unfolded
        return 0

    def __write_protein_atomtypes(self,fout,type,rad,seq,data):
        print (">> Writing atomtypes section")
        self.__write_unfolded_cgpdb__(rad=rad,data=data)
        #1:CA model or 2:CA+CB model
        fout.write('%s\n'%("[ atomtypes ]"))
        fout.write(6*"%s".ljust(5)%("; name","mass","charge","ptype","C6(or C10)","C12\n"))

        assert len(data.CA_atn)!=0 and type==2
        self.excl_volume["CA"] = 2*rad["CA"]
        self.excl_volume["CAi"] = self.excl_volume["CA"]
        C12 = self.fconst.Kr_prot*(2*rad["CA"])**12.0
        fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%("CA".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CA"))
        fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%("CAi".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CA"))
        for s in seq+self.idrdata.seq:
            bead = "CB"+s
            if bead in self.excl_volume or s == " ": continue
            C12 = self.fconst.Kr_prot*(2*rad[bead])**12.0
            self.excl_volume[bead] = 2*rad[bead]
            fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead.ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CB"))
            if s in self.idrdata.seq:
                self.excl_volume[bead+"i"]=self.excl_volume[bead]
                fout.write(" %s %8.3f %8.3f %s %e %e; %s\n"%(bead+"i".ljust(4),1.0,0.0,"A".ljust(4),0,C12,"CB"))
        
        return 0

    def __write_protein_nonbondparams(self,fout,type,excl_rule,data):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))
        fout.write(";C6 based repulsion term\n")
        assert type==2 and excl_rule == 2
        pairs,values = [],[]
        for x in self.excl_volume:
            if x.startswith(("CA","CB")) and x[-1]!="i":
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")) and y[-1]!="i":
                        sig=(self.excl_volume[x]+self.excl_volume[y])/2.0
                        eps = -1*self.fconst.Kr_prot
                        C06 = eps*(sig)**6
                        C12 = 0.0
                        p = [x,y]; p.sort(); p=tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            values.append((0.5*eps,sig,C12))  #values.append((C06,C12))
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
            if x.startswith(("CA","CB")) and x[-1]=="i":
                for y in self.excl_volume:
                    if y.startswith(("CA","CB")):
                        sig = (self.excl_volume[x]+self.excl_volume[y])/2.0
                        p = [x[:2],y[:2]]; p.sort(); p = tuple(p)
                        q=(x.strip("i"),y.strip("i"))
                        C06 = 2*eps[p]*epsmat[q]*(sig)**6
                        C12 = 1*eps[p]*epsmat[q]*(sig)**12
                        ptype=tuple(p)
                        p = [x,y]; p.sort(); p = tuple(p)
                        if p not in pairs:
                            pairs.append(p)
                            values.append((eps[ptype]*epsmat[q],sig,C12))  #values.append((C06,C12))
                            if self.opt.opensmog: continue
                            fout.write(" %s %s\t1\t%e %e "%(x.ljust(5),y.ljust(5),C06,C12))
                            if y.startswith("i"): fout.write("; IDR-IDR\n")
                            else: fout.write("; OR-IDR\n")

        if self.opt.opensmog and len(values)!=0:
            #C06,C12 = np.transpose(values)
            #self.prot_xmlfile.write_nonbond_xml(pairs=pairs,params={"C12":C12,"C6":C06},\
            #            expression='C12(type1,type2)/(r^12) - C6(type1,type2)/(r^6)')
            eps,sig,C12=np.transpose(values)
            self.prot_xmlfile.write_nonbond_xml(func=1,pairs=pairs,C12=C12,epsA=eps,sig=sig)

        return 0  

    def __write_protein_atoms(self,fout,type,cgfile,seq,inc_charge):
        print (">> Writing atoms section")
        fout.write("\n%s\n"%("[ atoms ]"))
        fout.write("%s\n"%(";nr  type  resnr residue atom  cgnr"))
        Q = dict()
        if inc_charge: 
            Q.update({x:1 for x in ["CBK","CBR","CBH"]})
            Q.update({x:-1 for x in ["CBD","CBE"]})
        
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
                    seqcount0,resnum0 = self.new2old_res[(seqcount,resnum)]
                    atinfo = (seqcount0,resnum0,resname,atname)
                    atype=atname
                    if resnum !=prev_resnum: prev_resnum,rescount=resnum,1+rescount
                    if atype=="CB": atype+=seq[seqcount][rescount]
                    if atype not in Q: Q[atype] = 0
                    if atinfo in self.idrdata.res:
                        Q[atype+"i"]=Q[atype]
                        atype=atype+"i"
                    fout.write("  %5d %5s %4d %5s %5s %5d %5.2f %5.2f\n"%(atnum,atype,resnum,resname,atname,atnum,Q[atype],1.0))
                    self.atomtypes.append(atype)
                elif line.startswith("TER"): seqcount,rescount=1+seqcount,0
        return

    def __get_idr_bonds__(self,data):
        unfolded=self.unfolded_data
        new2old_atn={}
        for c in unfolded.CA_atn:
            for r in unfolded.CA_atn[c]:
                c0,r0=self.new2old_res[(c,r)]
                assert r0==r
                if c0 in data.CA_atn and r0 in data.CA_atn[c0]:
                    new2old_atn[unfolded.CA_atn[c][r]]=data.CA_atn[c0][r0] 
        for c in unfolded.CB_atn:
            for r in unfolded.CB_atn[c]:
                c0,r0=self.new2old_res[(c,r)]
                assert r0==r
                if c0 in data.CB_atn and r0 in data.CB_atn[c0]:
                    new2old_atn[unfolded.CB_atn[c][r]]=data.CB_atn[c0][r0]

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

    def __write_protein_bonds(self,fout,data,func):
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

        data.Bonds()
        adjusted_data=self.__get_idr_bonds__(data=data)
        data.bonds=adjusted_data.bonds
        self.bonds.append(adjusted_data.bonds)
        table_idx = dict()
        if self.opt.opensmog:
            fout.write(";%5s %5s %5s %5s %5s; for excl\n"%("ai", "aj", "func", "r0", "Kb=0.0"))
            for c in range(len(adjusted_data.bonds)):
                pairs,dist=adjusted_data.bonds[c]
                print (">Writing chain %d Bonds as OpenSMOG contacts"%c)
                self.prot_xmlfile.write_pairs_xml( pairs=pairs,params={"r0":dist},\
                            name="FENE_bonds%d_R=0.2"%c,\
                            expression="-(K/2)*(R^2)*log(1-((r-r0)/R)^2); R=%.2f; K=%e"%(R,K))
                I,J = 1+np.transpose(pairs) 
                for i in range(pairs.shape[0]): 
                    r0 = np.round(dist[i],3)
                    fout.write(" %5d %5d %5d %.3f 0.0; dummy_entry\n"%(I[i],J[i],1,r0))
            return
        #else:
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "table_no.", "Kb"))
        for c in range(len(adjusted_data.bonds)):
            pairs,dist=adjusted_data.bonds[c]
            I,J = 1+np.transpose(pairs) 
            for i in range(pairs.shape[0]): 
                r0 = np.round(dist[i],3)
                if r0 not in table_idx: table_idx[r0]=len(table_idx)+self.tableb_ndx
                if r0-R>0:
                    r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                else:
                    r=0.001*np.int_(range(int(1000*(0+0.001)),int(1000*(r0+R-0.001))))
                V = -0.5*(R**2)*np.log(1-((r-r0)/R)**2)
                #V_1 = -0.5*(R**2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                V_1 = (R**2)*(r-r0)/(R**2-(r-r0)**2)
                Tables().write_bond_table(X=r,index=table_idx[r0],V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e; d=%.3f\n"%(I[i],J[i],func,table_idx[r0],K,r0))
        self.tableb_ndx=max(table_idx.values())
        return 

    def __write_protein_pairs(self,fout,data,excl_rule,charge):
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
        
        old2new_atn=dict()
        for c in self.unfolded_data.CA_atn:
            for r in self.unfolded_data.CA_atn[c]:
                c0,r0=self.new2old_res[(c,r)]
                assert r0==r
                if c0 in data.CA_atn and r0 in data.CA_atn[c0]:
                    old2new_atn[data.CA_atn[c0][r0]]=self.unfolded_data.CA_atn[c][r]
        for c in self.unfolded_data.CB_atn:
            for r in self.unfolded_data.CB_atn[c]:
                c0,r0=self.new2old_res[(c,r)]
                assert r0==r
                if c0 in data.CB_atn and r0 in data.CB_atn[c0]:
                    old2new_atn[data.CB_atn[c0][r0]]=self.unfolded_data.CB_atn[c][r]
            
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
                    if x in u_data.CB_atn[c] and x+1 in u_data.CA_atn[c]:
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
        assert len(pairs)!=0, "Error, No structured region?"
        
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
        #data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
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

        if len(idr_pairs)==0: return
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
        #data.contacts.append((pairs,np.zeros(pairs.shape),np.zeros(pairs.shape[0]),np.zeros(pairs.shape[0])))
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
