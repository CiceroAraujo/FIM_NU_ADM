import numpy as np
import scipy.sparse as sp
from ..preprocessor.multiscale import get_dual_and_primal_1, get_local_problems_structure
class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.GID_1, self.DUAL_1 = get_dual_and_primal_1(volumes['centroids'])
        self.local_problems_structure = get_local_problems_structure(self.DUAL_1, self.GID_1, faces['adjacent_volumes'])
        self.OP = self.get_prolongation_operator(faces['permeabilities'])

    @profile
    def get_prolongation_operator(self, ts):
        count=0
        for structure in self.local_problems_structure:
            count2=0
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
                    external_matrix.data = ts[entries]

                op=-sp.linalg.spsolve(internal_matrix, external_matrix)
                data=op.data
                g_lines=np.tile(internal_gids,len(external_gids))
                g_cols=np.repeat(external_gids,len(internal_gids))
                # import pdb; pdb.set_trace()
                count2+=1
            import pdb; pdb.set_trace()
            print(count, count2)
            count+=1
