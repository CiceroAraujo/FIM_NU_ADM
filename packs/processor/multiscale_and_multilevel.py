import numpy as np
import scipy.sparse as sp
from ..preprocessor.multiscale import get_dual_and_primal_1, get_local_problems_structure
class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.GID_1, self.DUAL_1 = get_dual_and_primal_1(volumes['centroids'])
        self.local_problems_structure, self.local_ID = get_local_problems_structure(self.DUAL_1, self.GID_1, faces['adjacent_volumes'])
        self.OP = self.get_prolongation_operator(faces['permeabilities'])

    @profile
    def get_prolongation_operator(self, ts):
        i=-1
        ops = []
        for structure in self.local_problems_structure:
            ops.append([])
            i+=1
            for local_problem in structure:
                try:
                    internal_matrix = local_problem[0][0]
                except:
                    import pdb; pdb.set_trace()
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
                    external_acumulator = local_external_problem[5]
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

                op=-sp.linalg.spsolve(internal_matrix, external_matrix)
                # import pdb; pdb.set_trace()
                data=op.data
                ops[i].append(data)
