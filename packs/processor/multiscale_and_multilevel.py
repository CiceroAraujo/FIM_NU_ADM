import numpy as np
import scipy.sparse as sp
from ..preprocessor.multiscale import get_dual_and_primal_1, get_local_problems_structure
from .assembler import Assembler
from packs.postprocessor.exporter import FieldVisualizer
visualize=FieldVisualizer()

class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.GID_0=volumes['GID_0']
        self.adjs=faces['adjacent_volumes']
        self.GID_1, self.DUAL_1 = get_dual_and_primal_1(volumes['centroids'])
        self.local_problems_structure, self.local_ID = get_local_problems_structure(self.DUAL_1, self.GID_1, faces['adjacent_volumes'])
        self.OP = self.get_prolongation_operator(faces['permeabilities'])
        self.Assembler = Assembler(wells, faces, volumes)

    def get_operators(self):
        self.get_finescale_vols()
        self.update_NU_ADM_mesh()
        self.update_NU_ADM_operators()
        self.update_R_and_P()
        return self.R, self.P

    def get_finescale_vols(self):
        self.fs_vols=np.array([0,48,30,24,18])
        # self.fs_vols=self.GID_0


    def get_prolongation_operator(self, ts):
        np.set_printoptions(3)
        i=-1
        ops = []
        glines=[]
        gcols=[]
        gdata=[]
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
                internal_matrix.data=sums[sums!=0]
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
                        matrix_connection.data=d[matrix_connection.data]
                        external_matrix.data = ts[entries]
                        external_matrix = external_matrix*matrix_connection
                    else:
                        external_matrix.data = ts[entries]
                        entity_up_ids=external_gids
                op=-sp.linalg.spsolve(internal_matrix, external_matrix)
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
        op=sp.csc_matrix((gdata, (glines, gcols)), shape=(glines.max()+1, gcols.max()+1))
        return op

    def update_NU_ADM_mesh(self):
        # self.fs_vols=fs_vols
        self.levels=np.ones_like(self.GID_1)
        self.NU_ADM_ID = -self.levels
        self.levels[self.fs_vols]=0
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
        self.NU_ADM_ID=labels
        gid1=self.GID_1[self.GID_0[self.DUAL_1==3]]
        self.coarse_id_NU_ADM=gid1

    def update_NU_ADM_mesh_dep(self):
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
        self.NU_ADM_ID[self.fs_vols]=ids

        self.vertices=self.GID_0[self.DUAL_1==3]
        gid1=self.GID_1[self.vertices]
        for rgid in remaining_ids:
            if rgid in gid1:
                gid1[gid1==rgid]=self.NU_ADM_ID[self.vertices[self.GID_1[self.vertices]==rgid]]
        self.coarse_id_NU_ADM=gid1

    def update_NU_ADM_operators(self):
        l, c, d=sp.find(self.OP)
        coarse=self.levels[l]==1
        mapc = self.NU_ADM_ID[self.DUAL_1==3]
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

        self.NU_ADM_OP = sp.csc_matrix((data, (lines, cols)), shape=(lines.max()+1, cols.max()+1))
        # import pdb; pdb.set_trace()
        visualize.plot_labels(self.OP[:,4].T.toarray()[0])
        visualize.plot_labels(self.NU_ADM_OP[:,12].T.toarray()[0])
        visualize.plot_labels(self.NU_ADM_ID)
        import pdb; pdb.set_trace()
        cols = self.GID_0
        lines = self.NU_ADM_ID
        data = np.ones(len(lines))
        self.NU_ADM_OR = sp.csc_matrix((data,(lines,cols)),shape=(lines.max()+1,cols.max()+1))

    def update_R_and_P(self):
        lp, cp, dp = sp.find(self.NU_ADM_OP)
        lr, cr, dr = sp.find(self.NU_ADM_OR)
        n_f, n_ADM=self.NU_ADM_OP.shape
        lP=np.concatenate([lp, cr+n_f])
        cP=np.concatenate([cp, lr+n_ADM])
        dP=np.concatenate([dp, dr])

        lR=np.concatenate([lr, lr+n_ADM])
        cR=np.concatenate([cr, cr+n_f])
        dR=np.concatenate([dr, dr])

        self.R=sp.csc_matrix((dR, (lR, cR)), shape=(2*n_ADM, 2*n_f))
        self.P=sp.csc_matrix((dP, (lP, cP)), shape=(2*n_f, 2*n_ADM))

    def newton_iteration_ADM(self, p, s, time_step, rel_tol=1e-3):
        pressure = p.copy()
        swns = s.copy()
        swn1s = s.copy()
        converged=False
        count=0
        dt=time_step
        while not converged:
            swns[self.Assembler.wells['ws_inj']]=1
            J, q=self.Assembler.get_jacobian_matrix(swns, swn1s, pressure, time_step)
            R, P = self.get_operators()
            sol=-P*sp.linalg.spsolve(R*J*P, R*q)
            n=int(len(q)/2)
            pressure+=sol[0:n]
            swns+=sol[n:]
            swns[self.Assembler.wells['ws_inj']]=1
            converged=max(abs(sol[n:]))<rel_tol
            print(max(abs(sol)),max(abs(sol)),'fs')
            count+=1
            if count>20:
                print('excedded maximum number of iterations finescale')
                return False, count, pressure, swns
        # saturation[wells['ws_prod']]=saturation[wells['viz_prod']].sum()/len(wells['viz_prod'])
        return True, count, pressure, swns
