import numpy as np
import scipy.sparse as sp
from ..preprocessor.multiscale import get_dual_and_primal_1, get_local_problems_structure
class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.GID_0=volumes['GID_0']
        self.GID_1, self.DUAL_1 = get_dual_and_primal_1(volumes['centroids'])
        self.local_problems_structure, self.local_ID = get_local_problems_structure(self.DUAL_1, self.GID_1, faces['adjacent_volumes'])
        self.OP = self.get_prolongation_operator(faces['permeabilities'])
        self.update_NU_ADM_mesh(np.array([1, -2]))
        self.update_NU_ADM_OP()

    @profile
    def get_prolongation_operator(self, ts):
        np.set_printoptions(3)
        i=-1
        ops = []
        glines=[]
        gcols=[]
        gdata=[]
        # import pdb; pdb.set_trace()
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

    def update_NU_ADM_mesh(self, fs_vols):
        self.fs_vols=fs_vols
        self.levels=np.ones_like(self.GID_1)
        self.NU_ADM_ID = -self.levels
        self.levels[fs_vols]=0
        coarse_volumes =  self.levels==1
        self.NU_ADM_ID[coarse_volumes]=self.GID_1[coarse_volumes]
        all_cvs=np.unique(self.NU_ADM_ID)
        if all_cvs.min()==-1:
            all_cvs=all_cvs[1:]
        remaining_ids=np.setdiff1d(np.unique(self.GID_1), all_cvs)
        nids=len(fs_vols)-len(remaining_ids)
        ids=np.concatenate([remaining_ids, self.NU_ADM_ID.max()+np.arange(nids)+1])
        self.NU_ADM_ID[fs_vols]=ids

    def update_NU_ADM_OP(self):
        l, c, d=sp.find(self.OP)
        coarse=self.levels[l]==1
        mapc = np.unique(self.NU_ADM_ID[self.DUAL_1==3])
        lines = l[coarse]
        cols = mapc[c[coarse]]
        data = d[coarse]

        ls = self.GID_0[self.fs_vols]
        cs = self.NU_ADM_ID[self.fs_vols]
        ds = -np.ones_like(cs)
        # import pdb; pdb.set_trace()
        lines = np.concatenate([lines, ls])
        cols = np.concatenate([cols, cs])
        data = np.concatenate([data, ds])
        self.NU_ADM_OP = sp.csc_matrix((data, (lines, cols)), shape=(lines.max()+1, cols.max()+1))
        # import pdb; pdb.set_trace()
