import numpy as np
import scipy.sparse as sp
import time
from ..preprocessor.multiscale import get_dual_and_primal_1, get_local_problems_structure
from .assembler import Assembler
from packs.postprocessor.exporter import FieldVisualizer
visualize=FieldVisualizer()

class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.PVI=0
        self.alpha_lim=1.0
        self.GID_0=volumes['GID_0']
        self.wells=wells
        self.swns=np.zeros(len(self.GID_0))
        self.adjs=faces['adjacent_volumes']
        self.prep_time=[]
        t0=time.time()
        self.GID_1, self.DUAL_1 = get_dual_and_primal_1(volumes['centroids'])
        self.prep_time.append(time.time()-t0)#prep1
        t0=time.time()
        self.local_problems_structure, self.local_ID = get_local_problems_structure(self.DUAL_1, self.GID_1, faces['adjacent_volumes'],faces['permeabilities'])
        self.OP, self.OP_matrix = self.get_prolongation_operator(faces['permeabilities'])
        self.prep_time.append(time.time()-t0)#prep2
        self.OR_matrix = sp.csc_matrix((np.ones_like(self.GID_0), (self.GID_1, self.GID_0)), shape=(int(self.GID_1.max()+1),int(self.GID_0.max()+1)))
        self.beta_lim=4
        self.get_op_to_alpha()
        self.get_beta_groups()
        self.Assembler = Assembler(wells, faces, volumes)
        self.proc_cumulative=[]
        self.int_prim_flag=[]
        self.PVI=[]
        self.porosities=volumes['pore_volume']

    def dual_aglomerator(self):
        t0=time.time()
        JP=self.Assembler.Jpp*self.OP_matrix
        RJP=self.OR_matrix*JP
        self.prep_time.append(time.time()-t0)#prep3
        l, c, d = sp.find(RJP)
        off_diags=l!=c
        l, c, d=l[off_diags], c[off_diags], d[off_diags]
        diags=RJP.diagonal()
        d=d/diags[l] #turns dd into neta
        upper=l>c
        d[upper][d[upper]<d[~upper]]=d[~upper][d[upper]<d[~upper]]
        l1, c1, d1 = l[upper], c[upper], d[upper]
        IJ=np.vstack([l1,c1])
        neta_IJ=np.vstack([d1,np.zeros_like(d1)]).max(axis=0)

    def get_operators(self):
        self.get_finescale_vols()
        t0=time.time()
        self.update_NU_ADM_mesh()
        self.update_averager()
        self.proc_temp.append(time.time()-t0) # time3
        t0=time.time()
        self.update_NU_ADM_operators()
        self.proc_temp.append(time.time()-t0) # time4
        self.update_R_and_P()

        rar=self.P.toarray()
        # visualize.plot_labels(self.NU_ADM_ID)
        # visualize.plot_labels(self.GID_0)
        # visualize.plot_labels(rar[:,5][:-36])
        return self.R, self.P

    def get_finescale_vols(self):
        flag=-np.ones_like(self.GID_1)
        swns=self.swns
        ws_p=self.wells['ws_prod']
        ws_i=self.wells['ws_inj']
        adjs=self.adjs
        deltas=abs(swns[adjs][:,0]-swns[adjs][:,1])
        fs=np.arange(len(deltas))[deltas>0.1]
        # import pdb; pdb.set_trace()
        vols=adjs[fs].flatten()
        vols=vols[swns[vols]<0.3]
        for i in range(2):
            flag[vols]=1
            viz=np.unique(adjs[flag[adjs].sum(axis=1)==0])
            vols=np.unique(np.concatenate([vols,viz]))

        t0=time.time()
        self.update_alpha()
        self.proc_temp.append((time.time()-t0)/10)#time1
        alpha_vols=self.GID_0[self.alphas>self.alpha_lim]
        # import pdb; pdb.set_trace()
        c_ws_p=self.GID_0[self.GID_1==self.GID_1[ws_p]]
        fs_vs=np.unique(np.concatenate([c_ws_p, ws_i, vols, alpha_vols]))
        bs=self.beta_ind[fs_vs]
        binds=np.unique(bs[bs>-1])

        if len(binds)>0:
            bvols=np.concatenate(self.beta_groups[binds])
            fs_vs=np.unique(np.concatenate([fs_vs,bvols]))
        self.fs_vols=fs_vs.copy()

    def get_finescale_vols_haji(self):
        flag=-np.ones_like(self.GID_1)
        swns=self.swns
        ws_p=self.wells['ws_prod']
        ws_i=self.wells['ws_inj']
        adjs=self.adjs        
        deltas=abs(swns[adjs][:,0]-swns[adjs][:,1])
        fs=np.arange(len(deltas))[deltas>0.1]


    def update_alpha_stabls(self):
        np.set_printoptions(5)
        nf=int(len(self.q)/2)
        # import pdb; pdb.set_trace()
        Jpp=self.J[0:nf,0:nf]

        JP=Jpp*self.OP_matrix
        RJP=self.OR_matrix*JP
        self.GID_0
        self.GID_1
        # JP=JP.tocsr().tocsc()
        l, c, d=sp.find(JP)
        same=c==self.GID_1[l]
        diag=RJP.diagonal()[self.GID_1]
        lines=l[~same]
        maxs=np.zeros(nf)
        np.maximum.at(maxs,lines,d[~same])
        # import pdb; pdb.set_trace()
        self.alphas=maxs/abs(diag)
    # @ profile
    def get_op_to_alpha(self):
        self.OP_to_alpha=self.OP_matrix.copy()
        l, c, _ =sp.find(self.OP_matrix)
        ad1s=self.GID_1[self.adjs]
        self.bound_prim=np.repeat(False,len(self.GID_1))
        self.bound_prim[ad1s[ad1s[:,0]!=ad1s[:,1]]]=True
        self.int_prim_flag=c==self.GID_1[l] | self.bound_prim[l]
        self.OP_to_alpha.data[self.int_prim_flag]=0

    def update_alpha(self):
        np.set_printoptions(5)
        nf=int(self.OP_matrix.shape[0]/2)

        JP=self.Assembler.Ta*self.OP_to_alpha
        RJP=self.OR_matrix*JP

        l, c, d=sp.find(JP)
        same=c==self.GID_1[l]
        diag=RJP.diagonal()[self.GID_1]
        lines=l[~same]
        maxs=np.zeros(l.max()+1)
        # import pdb; pdb.set_trace()
        np.maximum.at(maxs,lines,d[~same])
        # import pdb; pdb.set_trace()
        self.alphas=maxs/abs(diag)

    def get_beta_groups(self):
        pos=self.GID_1[self.OP[0]]==self.OP[1]
        phis=self.OP[2][pos][np.argsort(self.OP[0][pos])]
        # import pdb; pdb.set_trace()
        self.betas=(1-phis)/phis
        beta_facs=self.betas[self.adjs].max(axis=1)
        ads=self.adjs[beta_facs>self.beta_lim]
        map=np.arange(self.adjs.max())
        uads=np.unique(ads)
        n=len(uads)
        map[uads]=np.arange(n)
        adjs=map[ads]
        adjs=np.vstack([adjs,np.array([adjs[:,1],adjs[:,0]]).T])
        graph = sp.csc_matrix((np.ones_like(adjs[:,0]), (adjs[:,0], adjs[:,1])),shape=(n,n))
        n_l,labels=sp.csgraph.connected_components(graph)
        self.beta_ind=-np.ones_like(self.GID_0)
        self.beta_ind[uads]=labels
        self.beta_groups=np.array([uads[labels==l] for l in range(n_l)])

    def get_prolongation_operator_fast(self, ts):
        i=-1
        ops = []
        glines=[]
        gcols=[]
        gdata=[]
        prob=0
        if len(self.local_problems_structure[-1][0][0])==0:
            self.local_problems_structure=self.local_problems_structure[:-1]
        for structure in self.local_problems_structure:
            ops.append([])
            i+=1
            for local_problem in structure:
                internal_matrix = local_problem[0][0]
                off_diagonal_entries = local_problem[0][1]
                diagonal_entries = local_problem[0][2]
                acumulator = local_problem[0][3]
                internal_gids = local_problem[0][4]
                new_data = np.concatenate([ts[off_diagonal_entries], -ts[diagonal_entries]])
                sums=np.bincount(acumulator,weights=new_data)
                internal_matrix=internal_matrix.tocsc()
                # internal_matrix.data=np.arange(len(internal_matrix.data))
                # if prob==1:
                # print(internal_matrix.toarray())
                internal_matrix.data=sums[sums!=0]
                # print(internal_matrix.toarray())
                # import pdb; pdb.set_trace()


                for local_external_problem in local_problem[1]:
                    external_matrix = local_external_problem[0]
                    entries = local_external_problem[1]
                    external_gids = local_external_problem[2]
                    entity_up_ids = local_external_problem[3]
                    matrix_connection = local_external_problem[4]
                    columns= local_external_problem[5]
                    if entity_up_ids.max()>-1:
                        d=[]
                        for e in entity_up_ids:
                            d.append(ops[i-1][e])
                        d=np.concatenate(d)
                        # import pdb; pdb.set_trace()
                        matrix_connection.data=d[matrix_connection.data-1]
                        external_matrix.data = ts[entries]
                        external_matrix = external_matrix*matrix_connection
                    else:
                        external_matrix.data = ts[entries]
                        entity_up_ids=external_gids
                op=-sp.linalg.spsolve(internal_matrix, external_matrix)
                # if ((op.sum(axis=1)<0.999) | (op.sum(axis=1)>1.001)).sum()!=0:
                #     # import pdb; pdb.set_trace()
                #     prob=1

                # import pdb; pdb.set_trace()
                fop=sp.find(op)
                glines.append(internal_gids[fop[0]])
                gcols.append(columns[fop[1]])
                gdata.append(fop[2])
                data=op.data
                ops[i].append(data)

        all_volumes=np.arange(len(self.DUAL_1))
        vertices=self.DUAL_1==3
        glines.append(all_volumes[vertices])
        gcols.append(all_volumes[vertices])
        gdata.append(np.ones_like(all_volumes[vertices]))

        glines=np.concatenate(glines)
        gcols=np.concatenate(gcols)
        mapg=-np.ones(gcols.max()+1).astype(int)
        mapg[np.unique(gcols)]=np.arange(len(np.unique(gcols)))
        gcols=mapg[gcols]
        gdata=np.concatenate(gdata)
        op1=[glines, gcols, gdata]
        OP_AMS=sp.csc_matrix((gdata, (glines, gcols)),shape=(int(glines.max()+1), int(gcols.max())+1))
        return op1, OP_AMS

    def get_prolongation_operator(self,ts):
        # _,_1=self.get_prolongation_operator_fast(ts )
        # import pdb; pdb.set_trace()
        dual_1=self.DUAL_1
        n=len(dual_1)
        nv=(dual_1==3).sum()
        ne=(dual_1==2).sum()
        nf=(dual_1==1).sum()
        wire=-np.ones_like(dual_1)
        wire[dual_1==1]=range(nf)
        wire[dual_1==2]=nf+range(ne)
        # import pdb; pdb.set_trace()
        wire[dual_1==3]=nf+ne+range(nv)
        # G = sp.csc_matrix((np.ones_like(dual_1), (wire, self.GID_0)),shape=(n,n))
        a=wire[self.adjs.T]

        lines=np.concatenate([a[0],a[1],a[0],a[1]])
        cols=np.concatenate([a[1],a[0],a[0],a[1]])
        data=np.concatenate([ts,ts,-ts,-ts])

        T = sp.csc_matrix((data, (lines, cols)),shape=(n,n))
        # W=G*T*G.T
        Tff=T[0:nf,0:nf]
        Tfe=T[0:nf,nf:ne+nf]
        Tee=T[nf:ne+nf,nf:ne+nf]
        Tev=T[nf:ne+nf,ne+nf:n]
        Tvv=T[ne+nf:n,ne+nf:n]
        Tvv.data=np.ones(nv)
        # import pdb; pdb.set_trace()
        aux=sp.csc_matrix((np.array(Tfe.sum(axis=0))[0], (range(ne), range(ne))),shape=(ne,ne))
        Tee+=aux
        ope=sp.linalg.spsolve(Tee,-Tev*Tvv)
        opf=sp.linalg.spsolve(Tff,-Tfe*ope)
        prol=sp.vstack([opf,ope,Tvv])

        # import pdb; pdb.set_trace()
        G=sp.csc_matrix((np.ones(n), (wire.astype(int), self.GID_0)),shape=(n, n))
        fp=sp.find(G.T*prol)
        lp=fp[0]
        cp=fp[1]
        dp=fp[2]
        op1=[lp, cp, dp]
        OP_AMS=sp.csc_matrix((dp, (lp, cp)),shape=(int(lp.max()+1), int(cp.max())+1))

        return op1, OP_AMS

    def update_averager(self):
        # self.fs_vols=fs_vols
        levels=np.ones_like(self.GID_1)
        NU_ADM_ID = -self.levels
        levels[self.fs_vols]=0
        gid1_adjs=self.GID_1[self.adjs]
        same_gid=gid1_adjs[:,0]==gid1_adjs[:,1]
        cc_adjs=self.levels[self.adjs].sum(axis=1)==2
        adjs=self.adjs[same_gid & cc_adjs]
        fines=np.tile(self.fs_vols,(2,1)).T
        adjs=np.vstack([fines,adjs])
        adjs=np.tile(adjs,(2,1))
        data = np.ones(len(adjs))
        n=len(self.levels)
        graph = sp.csc_matrix((data, (adjs[:,0], adjs[:,1])),shape=(n,n))
        n,labels=sp.csgraph.connected_components(graph)
        '''
        self.NU_ADM_ID=labels
        gid1=self.GID_1[self.GID_0[self.DUAL_1==3]]
        self.coarse_id_NU_ADM=gid1
        '''
        cols=self.GID_0
        lines=labels
        data=np.ones_like(cols)
        # averager=sp.csc_matrix(())
        # import pdb; pdb.set_trace()
        self.averager=sp.csc_matrix((data, (lines, cols)), shape=(lines.max()+1, cols.max()+1))

    def update_NU_ADM_mesh(self):
        # self.fs_vols=fs_vols
        self.levels=np.ones_like(self.GID_1)
        self.NU_ADM_ID = -self.levels
        self.levels[self.fs_vols]=0
        coarse_volumes =  self.levels==1
        self.NU_ADM_ID[coarse_volumes]=self.GID_1[coarse_volumes]
        all_cvs=np.unique(self.NU_ADM_ID)
        if all_cvs.min()==-1:
            all_cvs=all_cvs[1:]
        remaining_ids=np.setdiff1d(np.unique(self.GID_1), all_cvs)
        nids=len(self.fs_vols)-len(remaining_ids)
        ids=np.concatenate([remaining_ids, self.GID_1.max()+np.arange(nids)+1])
        # import pdb; pdb.set_trace()
        self.NU_ADM_ID[self.fs_vols]=ids

        self.vertices=self.GID_0[self.DUAL_1==3]
        gid1=self.GID_1[self.vertices]
        for rgid in remaining_ids:
            if rgid in gid1:
                gid1[gid1==rgid]=self.NU_ADM_ID[self.vertices[self.GID_1[self.vertices]==rgid]]
        self.coarse_id_NU_ADM=gid1

    def update_NU_ADM_operators(self):
        l, c, d=self.OP
        coarse=self.levels[l]==1
        # import pdb; pdb.set_trace()
        # mapc = self.NU_ADM_ID[self.DUAL_1==3] #aqui trocar por linha abaixo
        mapc = self.coarse_id_NU_ADM
        lines = l[coarse]
        cols = mapc[c[coarse]]
        # import pdb; pdb.set_trace()
        same=self.GID_1[lines]==c[coarse]
        cols[same]=self.NU_ADM_ID[lines[same]]
        # import pdb; pdb.set_trace()
        # cols = self.NU_ADM_ID[lines]
        data = d[coarse]
        ls = self.fs_vols
        cs = self.NU_ADM_ID[self.fs_vols]
        ds = np.ones_like(cs)

        lines = np.concatenate([lines, ls])
        cols = np.concatenate([cols, cs])
        data = np.concatenate([data, ds])

        self.NU_ADM_OP = [lines, cols, data]
        # import pdb; pdb.set_trace()
        # visualize.plot_labels(self.OP[:,4].T.toarray()[0])
        # visualize.plot_labels(self.NU_ADM_OP[:,12].T.toarray()[0])
        # visualize.plot_labels(self.levels)
        # import pdb; pdb.set_trace()
        cols = self.GID_0
        lines = self.NU_ADM_ID
        data = np.ones(len(lines))
        self.NU_ADM_OR = [lines,cols, data]

    def update_R_and_P(self):
        lp, cp, dp = self.NU_ADM_OP
        lr, cr, dr = self.NU_ADM_OR
        n_f, n_ADM=lp.max()+1, cp.max()+1
        lP=np.concatenate([lp, cr+n_f])
        cP=np.concatenate([cp, lr+n_ADM])
        dP=np.concatenate([dp, dr])

        lR=np.concatenate([lr, lr+n_ADM])
        cR=np.concatenate([cr, cr+n_f])
        dR=np.concatenate([dr, dr])
        self.R=sp.csc_matrix((dR, (lR, cR)), shape=(2*n_ADM, 2*n_f))
        self.P=sp.csc_matrix((dP, (lP, cP)), shape=(2*n_f, 2*n_ADM))
    # @profile
    def newton_iteration_ADM(self, p, s, time_step, rel_tol=1e-5):
        pressure = p.copy()
        swns = s.copy()
        swn1s = s.copy()
        converged=False
        count=0
        dt=time_step
        while not converged:
            self.proc_temp=[]
            swns[self.Assembler.wells['ws_inj']]=1
            self.swns=swns

            self.J, self.q=self.Assembler.get_jacobian_matrix(swns, swn1s, pressure, time_step)

            self.proc_temp.append(self.Assembler.time_Jpp) #prep_time1
            R, P = self.get_operators()
            t0=time.time()
            sol=-P*sp.linalg.spsolve(R*self.J*P, R*self.q)
            self.proc_temp.append(time.time()-t0) #time5
            self.proc_cumulative.append(self.proc_temp)
            n=int(len(self.q)/2)
            pressure+=sol[0:n]
            sol[n+self.wells['ws_prod']]=0
            swns+=sol[n:]
            swns[self.Assembler.wells['ws_inj']]=1
            # import pdb; pdb.set_trace()
            # swns[self.Assembler.wells['ws_prod']]=0

            self.PVI=(self.swns*self.porosities).sum()/self.porosities.sum()
            # import pdb; pdb.set_trace()
            # posics=np.arange(len(swns))
            # posics=posics[posics!=
            # converged=max(abs(sol[n:]))<rel_tol
            converged=max(abs(sol))<rel_tol
            print(max(abs(sol)),'fs')
            count+=1
            if count>20:
                print('excedded maximum number of iterations finescale')
                # import pdb; pdb.set_trace()
                return False, count, pressure, swns
        # saturation[wells['ws_prod']]=saturation[wells['viz_prod']].sum()/len(wells['viz_prod'])
        na, nf=int(self.R.shape[0]/2), int(self.R.shape[1]/2)
        # OR_ADM=self.R[na:,nf:]
        OR_ADM=self.averager

        swns=OR_ADM.T*(OR_ADM.tocsr()*swns/np.array(OR_ADM.sum(axis=1)).T[0])

        # swns=(Rs.T*(Rs*swns))#/Rs.sum(axis=0)
        return True, count, pressure, swns
