import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from ..postprocessor.exporter import FieldVisualizer
from .assembler import Assembler
visualize=FieldVisualizer()

class NewtonIterationFinescale():
    def __init__(self, wells, faces, volumes):
        self.Assembler = Assembler(wells, faces, volumes)

    # @profile
    def newton_iteration_finescale(self, p, s, time_step, rel_tol=1e-3):
        pressure = p.copy()
        swns = s.copy()
        swn1s = s.copy()
        converged=False
        count=0
        dt=time_step
        # data_impress['swn1s']=data_impress['swns'].copy()
        # all_ids=GID_0
        # not_prod=np.setdiff1d(all_ids,wells['all_wells'])
        while not converged:
            swns[self.Assembler.wells['ws_inj']]=1
            J, q=self.Assembler.get_jacobian_matrix(swns, swn1s, pressure, time_step)
            sol=-linalg.spsolve(J, q)
            n=int(len(q)/2)
            pressure+=sol[0:n]
            swns+=sol[n:]
            # visualize.plot_field(pressure)
            swns[self.Assembler.wells['ws_inj']]=1
            converged=max(abs(sol[n:]))<rel_tol
            print(max(abs(sol)),max(abs(sol)),'fs')
            count+=1
            if count>20:
                print('excedded maximum number of iterations finescale')
                return False, count, pressure, swns
        # saturation[wells['ws_prod']]=saturation[wells['viz_prod']].sum()/len(wells['viz_prod'])
        return True, count, pressure, swns
