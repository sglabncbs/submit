import numpy as np
from tqdm import trange
from PDB_IO import *
from hybrid_36 import hy36encode,hy36decode

class Tables:
    def __init__(self) -> None:
        pass
    def __write_bond_table__(self,index,X,V,V_1):
        with open("table_b"+str(index)+".xvg","w+") as fout:
            step = 0.002 #nm
            for i in range(X.shape[0]):
                fout.write("%e %e %e\n"%(X[i],V[i],-V_1[i]))
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
        return

    def Bonds(self):
        # Getting Bond length info from the pre-supplied data
        pairs = []
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

    def Angles(self):
        # Getting Bond angle info from the pre-supplied data
        triplets = []
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
            
    def Dihedrals(self):
        # Getting torsion angle info from the pre-supplied data
        if len(self.CA_atn) != 0:
            quadruplets = []
            for c in self.CA_atn:
                resnum = list(self.CA_atn[c].keys())
                resnum.sort()
                quadruplets += [tuple([self.CA_atn[c][x+i] for i in range(4)]) for x in resnum if x+3 in self.CA_atn[c]]
            quadruplets = np.int_(quadruplets)
            T = self.__torsions__(quadruplets=quadruplets)
            self.bb_dihedrals.append((quadruplets,T))
            if len(self.CB_atn)!=0:
                quadruplets = []
                for x in self.CA_atn: quadruplets += [(self.CA_atn[c][x-1],self.CA_atn[c][x+1],self.CA_atn[c][x],self.CB_atn[c][x]) for x in resnum if x+1 in self.CA_atn[c] and x-1 in self.CA_atn[c] and x in self.CB_atn[c]]
                quadruplets = np.int_(quadruplets)
                T = self.__torsions__(quadruplets=quadruplets)
                self.sc_dihedrals.append((quadruplets,T))

    def Pairs(self,cmap,aa_data):
        # Getting Non-bonded contact pairs info from the pre-supplied data
        temp_p,temp_w,temp_d = [],[],[]
        pairs,weights,distances = [],[],[]
        if cmap.type == -1: return  # Generating top without pairs 
        elif cmap.type == 0:        # Use pairs from user input in format cid_i, atnum_i, cid_j, atnum_j, weight_ij (opt), dist_ij (opt)
            assert cmap["file"] != ""
            print ("> Using cmap file (c1 a1 c2 a2 w d)",cmap["file"])
            with open(cmap["file"]) as fin:
                for line in fin:
                    line = line.split()
                    c1,a1,c2,a2 = np.int_(line[:4])
                    if len(line) < 6:
                        w,d = 1.0,0.0
                        if len(line)==5: w = np.float(line[4])
                        temp_p.append((a1,a2));temp_w.append(w)
                    elif len(line)==6: 
                        w,d = np.float_(line[4:])
                        pairs.append((a1,a2));weights.append(w);distances.append(d)
            if len(temp_p)!=0: temp_d = list(self.__distances__(pairs=np.int_(temp_p)-1))
            pairs += temp_p; weights += temp_w; distances += temp_d
            pairs = np.int_(pairs); weights = np.float_(weights); distances = np.float_(distances)
        elif cmap.type == 1:        # Calculating contacts from all-atom structure and maping to CG structure
            group = []
            if len(self.CA_atn) != 0:
                if len(self.CB_atn) == 0:
                    for r in aa_data.res: group.append(tuple(list(r[:2])+[0]))
                else:
                    for r in aa_data.res: group.append(tuple(list(r[:2])+[int(r[-1] not in ("N","C","CA","O"))]))
            faa = open(aa_data.pdbfile+".AAcont","w+")
            fcg = open(aa_data.pdbfile+".CGcont","w+")
            cid,rnum,bb_sc = np.transpose(np.array(group))
            aa2cg = {0:self.CA_atn,1:self.CB_atn}
            del (group)
            cutoff = cmap.cutoff*cmap.scale
            resgap = 4 
            contacts_dict = dict()
            for i in range(aa_data.xyz.shape[0]):
                gap=resgap-np.int_(bb_sc+bb_sc[i]>0)
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
            cid = np.int_([contacts_dict[x][0] for x in pairs])
            pairs = np.int_(pairs)
            distances = self.__distances__(pairs=pairs)
            for x in range(pairs.shape[0]):
                c,a = cid[x]+1,pairs[x]+1
                w,d = weights[x],distances[x]
                fcg.write("%d %d %d %d %.3f %.3f\n"%(c[0],a[0],c[1],a[1],w,d))
            faa.close();fcg.close()
            if not cmap.W: weights = np.ones(weights.shape)
        elif cmap.type == 2:        # Calculating contacts from CG structure
            cacasep=4;cacbsep=3;cbcbsep=3
            if len(self.CA_atn) != 0:
                for c1 in self.CA_atn:
                    pairs += [(self.CA_atn[c1][x],self.CA_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c1] if y-x>=cacasep]
                    if len(self.CB_atn) != 0:
                        pairs += [(self.CA_atn[c1][x],self.CB_atn[c1][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CA_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c1] if y-x>=cacbsep]
                        pairs += [(self.CB_atn[c1][x],self.CB_atn[c1][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c1] if y-x>=cbcbsep]
                    for c2 in self.CA_atn:
                        if c2>c1: 
                            pairs += [(self.CA_atn[c1][x],self.CA_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CA_atn[c2]]
                            if len(self.CB_atn)!=0: 
                                pairs += [(self.CA_atn[c1][x],self.CB_atn[c2][y]) for x in self.CA_atn[c1] for y in self.CB_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CA_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CA_atn[c2]]
                                pairs += [(self.CB_atn[c1][x],self.CB_atn[c2][y]) for x in self.CB_atn[c1] for y in self.CB_atn[c2]]

            pairs = np.int_(pairs)
            distances = self.__distances__(pairs)
            weights = np.ones(pairs.shape[0])
        self.contacts.append((pairs,distances,weights))

class Topology:
    def __init__(self,allatomdata,fconst,CGlevel,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.cmap = cmap
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

    def __write_atomtypes__(self,fout,type,rad,seq):
        print (">> Writing atomtypes section")
        #1:CA model or 2:CA+CB model
        assert type<=2
        self.excl_volume["CA"] = 2*rad["CA"]
        C12 = self.fconst.Kr_prot*(2*rad["CA"])**12.0
        fout.write('%s\n'%("[ atomtypes ]"))
        fout.write(6*"%s".ljust(5)%("; name","mass","charge","ptype","C6(or C10)","C12"))
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
        return

    def __write_nonbond_params__(self,fout,type,excl_rule):
        print (">> Writing nonbond_params section")
        ##add non-bonded r6 term in sop-sc model for all non-bonded non-native interactions.
        fout.write("\n%s\n"%("[ nonbond_params ]"))
        fout.write('%s\n' % ('; i    j     func C6(or C10)  C12'))
        if type == 1 or excl_rule == 1: return
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
            Q.update({x:-1 for x in ["CBD","CBE"]})
        with open(cgfile) as fin:
            for line in fin:
                if line.startswith("ATOM"):
                    atnum=hy36decode(5,line[6:11])
                    atname=line[12:16].strip()
                    resname=line[17:20].strip()
                    resnum=hy36decode(4,line[22:26])
                    atype=atname
                    if atype=="CB": atype+=data.prot.amino_acid_dict[resname]
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
        K = float(self.fconst.Kb_prot)*100.0

        #GROMACS 4.5.4 : FENE=7 AND HARMONIC=1
        #if dsb: func = 9
        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "r0(nm)", "Kb"))
        if func==1: R = str()
        else: R = str(0.2) #nm

        data.Bonds()
        for pairs,dist in data.bonds:
            pairs += 1 
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(pairs[x][0],pairs[x][1],func,dist[x],K))
        return 

    def __write_protein_angles__(self,fout,data):
        print (">> Writing angless section")
        #V_ang = (Ktheta/2)*(r-r0)^2
        #Input units KJ mol-1 #GROMACS units KJ mol-1 
        K = float(self.fconst.Ka_prot)

        fout.write("\n%s\n"%("[ angles ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s\n"%("ai", "aj", "ak","func", "th0(deg)", "Ka"))

        func = 1
        data.Angles()
        for triplets,angles in data.angles:
            triplets += 1 
            for x in range(triplets.shape[0]): 
                fout.write(" %5d %5d %5d %5d %e %e\n"%(triplets[x][0],triplets[x][1],triplets[x][2],func,angles[x],K))
        return

    def __write_protein_dihedrals__(self,fout,data,chiral):
        print (">> Writing dihedrals section")

        #GROMACS IMPLEMENTATION: Edihedrals Kphi*(1 + cos(n(phi-phi0)))
        #Our implementaion: Edihedrals = Kphi*(1 - cos(n(phi-phi0)))
        #The negative sign is included by adding phase = 180 to the phi0
        #Kphi*(1 + cos(n(phi-180-phi0))) = Kphi*(1 + cos(n180)*cos(n(phi-phi0)))
        #if n is odd i.e. n=1,3.... then cos(n180) = -1
        #hence Edihedrals = Kphi*(1 - cos(n(phi-phi0)))

        Kbb = float(self.fconst.Kd_prot["bb"])
        Ksc = float(self.fconst.Kd_prot["sc"])
        mfac = float(self.fconst.Kd_prot["mf"])

        phase = 180
        radtodeg = 180/np.pi

        fout.write("\n%s\n"%("[ dihedrals ]"))
        fout.write("; %5s %5s %5s %5s %5s %5s %5s %5s\n" % (";ai","aj","ak","al","func","phi0(deg)","Kd","mult"))

        data.Dihedrals()
        func = 1
        for quads,diheds in data.bb_dihedrals:
            quads+= 1 
            diheds += phase
            for x in range(quads.shape[0]):
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(quads[x][0],quads[x][1],quads[x][2],quads[x][3],func,diheds[x],Kbb,1))
                fout.write(" %5d %5d %5d %5d %5d %e %e %d\n"%(quads[x][0],quads[x][1],quads[x][2],quads[x][3],func,3*diheds[x],Kbb/mfac,3))
        if chiral and len(data.CB_atn) != 0:
            func = 2
            fout.write("; %5s %5s %5s %5s %5s %5s %5s \n" % (";ai","aj","ak","al","func","phi0(deg)","Kd"))
            for quads,diheds in data.sc_dihedrals:
                quads+= 1 
                for x in range(quads.shape[0]):fout.write(" %5d %5d %5d %5d %5d %e %e\n"%(quads[x][0],quads[x][1],quads[x][2],quads[x][3],func,diheds[x],Ksc))
        return

    def __write_protein_pairs__(self,fout,data,excl_rule):
        print (">> Writing pairs section")
        cmap = self.cmap
        data.Pairs(cmap=cmap,aa_data=self.allatomdata.prot)

        if cmap.scsc_custom:
            assert cmap.type in (1,2)
            with open("interactions.dat") as fin:
                scscmat = {line.split()[0]:float(line.split()[1]) for line in fin}
                scscmat.update({k[1]+k[0]:v for k,v in scscmat.items()})                
            CB_atn = {v:Prot_Data().amino_acid_dict[k[2]] \
                for k,v in self.allatomdata.prot.CB_atn.items()}
            all_atn = CB_atn.copy()
            all_atn.update({v:Prot_Data().amino_acid_dict[k[2]] \
                        for k,v in self.allatomdata.prot.CA_atn.items()})
            for index in range(len(data.contacts)):
                pairs,dist,eps = data.contacts[index]
                I,J = np.transpose(pairs)
                interaction_type = \
                        np.int_([x in CB_atn for x in I])+ \
                        np.int_([x in CB_atn for x in J])
                eps_scsc = np.float_([scscmat[all_atn[I[x]]+all_atn[J[x]]] for x in range(I.shape[0])])
                eps_bbsc = np.float_(eps)
                eps_bbbb = np.float_(eps)
                eps = eps_bbbb*np.int_(interaction_type==0) \
                    + eps_bbsc*np.int_(interaction_type==1) \
                    + eps_scsc*np.int_(interaction_type==2) 
                for x in range(eps.shape[0]): print(eps[x],interaction_type[x],new_eps[x])
                #eps = np.round(eps + (eps_IJ - eps)*interaction_type,3)
                data.contacts[index] = pairs,dist,eps

        fout.write("\n%s\n"%("[ pairs ]"))
        if cmap.func==1:
            print ("> Using LJ C6-C12 for contacts")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C06(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                pairs += 1 
                c06 = 2*eps*(dist**6.0)
                c12 = eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(pairs[x][0],pairs[x][1],func,c06[x],c12[x]))
        elif cmap.func==2:
            print ("> Using LJ C10-C12 for contacts. Note: Require Table file(s)")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C10(Att)","C12(Rep)"))
            func = 1
            for pairs,dist,eps in data.contacts:
                pairs += 1 
                c10 = 6*eps*(dist**10.0)
                c12 = 5*eps*(dist**12.0)
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %e %e\n"%(pairs[x][0],pairs[x][1],func,c10[x],c12[x]))
        elif cmap.func==3:
            print ("> Using LJ C12-C18 for contacts. Note: Require Table file(s) or ")
            fout.write(";%5s %5s %5s %5s %5s\n"%("i","j","func","C12(Att)","C18(Rep)"))
            func = 3
            assert func!=3, "Error, func 3 not encoded yes. WIP"
            for pairs,dist,eps in data.contacts:
                pairs += 1 
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
                pairs += 1
                for x in range(pairs.shape[0]): 
                    fout.write(" %5d %5d %5d %.3f %e %e %e\n"%(pairs[x][0],pairs[x][1],func,eps[x],dist[x],sd,c12[x]))
        return 

    def __write_protein_exclusions(self,fout,data):
        print (">> Writing exclusions section")
        fout.write("\n%s\n"%("[ exclusions ]"))
        fout.write("; %5s %5s\n"%("i","j"))
        for pairs,dist,eps in data.contacts:
            pairs += 1
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d\n"%(pairs[x][0],pairs[x][1]))
        return

    def write_topfile(self,outtop,excl,charge,bond_function,CBchiral,rad):
        if len(self.allatomdata.prot.lines) > 0:
            cgpdb = PDB_IO()
            if self.CGlevel["prot"]==1: cgpdb.loadfile(infile=self.allatomdata.prot.bb_file,refine=False)
            elif self.CGlevel["prot"]==2: cgpdb.loadfile(infile=self.allatomdata.prot.sc_file,refine=False)
            prot_topfile = "prot_"+outtop
            with open(prot_topfile,"w+") as ftop:
                print (">>> writing Protein GROMACS toptology", prot_topfile)
                proc_data = Calculate(pdb=cgpdb.prot)
                proc_data.processData(data=cgpdb.prot)
                self.__write_header__(fout=ftop,combrule=excl)
                self.__write_atomtypes__(fout=ftop,type=self.CGlevel["prot"],seq=cgpdb.prot.seq,rad=rad)
                self.__write_nonbond_params__(fout=ftop,type=self.CGlevel["prot"],excl_rule=excl)
                self.__write_moleculetype__(fout=ftop)
                self.__write_atoms__(fout=ftop,type=self.CGlevel["prot"], data=cgpdb,inc_charge=charge["CB"])
                self.__write_protein_pairs__(fout=ftop, data=proc_data,excl_rule=excl)
                self.__write_protein_bonds__(fout=ftop, data=proc_data,func=bond_function)
                self.__write_protein_angles__(fout=ftop, data=proc_data)
                self.__write_protein_dihedrals__(fout=ftop, data=proc_data,chiral=CBchiral)
                self.__write_protein_exclusions(fout=ftop,data=proc_data)
                self.__write_footer__(fout=ftop)
        return
    
class Clementi2000(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.cmap = cmap
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

class Pal2019(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.cmap = cmap
        self.opt = opt
        self.bfunc,self.afunc,self.pfunc,self.dfunc = 1,1,1,1
        self.excl_volume = dict()
        self.atomtypes = []

class Reddy2017(Topology):
    def __init__(self,allatomdata,fconst,CGlevel,cmap,opt) -> None:
        self.allatomdata = allatomdata
        self.__check_H_atom__()
        self.fconst = fconst
        self.CGlevel = CGlevel
        self.cmap = cmap
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

    def __write_nonbond_params__(self,fout,type,excl_rule):
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
        sopsc = False
        if sopsc: print ("WIP") 
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

        # V = - (K/2)*R*ln(1-((r-r0)/R)^2)
        # V_1 = dV/dr = -(K/2)*R*(1/(1-((r-r0)/R)^2))*(-2*(r-r0)/R^2)

        fout.write("\n%s\n"%("[ bonds ]"))
        fout.write(";%5s %5s %5s %5s %5s\n"%("ai", "aj", "func", "table_no.", "Kb"))
        data.Bonds()
        table = dict()
        for pairs,dist in data.bonds:
            pairs += 1 
            for i in range(pairs.shape[0]): 
                r0 = dist[i]
                r=0.001*np.int_(range(int(1000*(r0-R+0.001)),int(1000*(r0+R-0.001))))
                V = -(R*K/2)*np.log(1-((r-r0)/R)**2)
                V_1 = -(R*K/2)*(1/(1-((r-r0)/R)**2))*(-2*(r-r0)/R**2)
                Tables().__write_bond_table__(X=r,index=i,V=V,V_1=V_1)
                fout.write(" %5d %5d %5d %5d %e\n"%(pairs[i][0],pairs[i][1],func,i,K))
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

    def __write_protein_pairs__(self,fout,data,excl_rule):
        print (">> Writing SOP-SC pairs section")
        cmap = self.cmap
        data.Pairs(cmap=cmap,aa_data=self.allatomdata.prot)

        assert cmap.scsc_custom and cmap.type==2 and cmap.func==1
        with open("interactions.dat") as fin:
            scscmat = {line.split()[0]:float(line.split()[1]) for line in fin}
            scscmat.update({k[1]+k[0]:v for k,v in scscmat.items()})                
            
        CA_atn = {v:"CA"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CA_atn.items()}
        CB_atn = {v:"CB"+Prot_Data().amino_acid_dict[k[2]] for k,v in self.allatomdata.prot.CB_atn.items()}
        all_atn = CA_atn.copy()
        all_atn.update(CB_atn.copy())
        eps_bbbb = 0.5*self.fconst.kcalAtokjA
        eps_bbsc = 0.5*self.fconst.kcalAtokjA
        Kboltz = self.fconst.Kboltz #*self.fconst.kcalAtokjA/self.fconst.kcalAtokjA
        for index in range(len(data.contacts)):
            pairs,dist,eps = data.contacts[index]
            I,J = np.transpose(pairs)
            interaction_type = np.int_(\
                np.int_([x in CB_atn for x in I])+ \
                np.int_([x in CB_atn for x in J]))
            eps_scsc = np.float_([scscmat[all_atn[I[x]][-1]+all_atn[J[x]][-1]] for x in range(I.shape[0])])
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
            pairs += 1 
            c06 = 2*eps*(dist**6.0)
            c12 = eps*(dist**12.0)
            for x in range(pairs.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(pairs[x][0],pairs[x][1],func,c06[x],c12[x]))
        
        fout.write(";angle based rep temp\n;%5s %5s %5s %5s %5s\n"%("i","j","func","-C06(Rep)","C12 (N/A)"))        
        data.Angles()
        diam = self.excl_volume.copy()
        diam.update({"CA"+k[-1]:diam["CA"] for k in diam.keys() if k.startswith("CB")})
        eps_bbbb = 1.0*self.fconst.kcalAtokjA
        eps_bbsc = 1.0*self.fconst.kcalAtokjA
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
            I,K = I+1,K+1
            for x in range(triplets.shape[0]): 
                fout.write(" %5d %5d %5d %e %e\n"%(I[x],K[x],func,c06[x],0.0))
        return 

