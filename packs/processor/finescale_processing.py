import numpy as np
import scipy.sparse as sp
import time
from scipy.sparse import linalg
from ..postprocessor.exporter import FieldVisualizer
from .assembler import Assembler
visualize=FieldVisualizer()

class NewtonIterationFinescale():
    def __init__(self, wells, faces, volumes):
        self.Assembler = Assembler(wells, faces, volumes)
        self.time_solve=[]
        self.porosities=volumes['pore_volume']
        self.wells=wells

    # @profile
    def newton_iteration_finescale(self, p, s, time_step, rel_tol=1e-5):
        pressure = p.copy()
        swns = s.copy()
        swn1s = s.copy()
        converged=False
        count=0
        dt=time_step
        while not converged:
            # swns[self.Assembler.wells['ws_inj']]=1
            self.Assembler.iteration=count
            J, q=self.Assembler.get_jacobian_matrix(swns, swn1s, pressure, time_step)
            t0=time.time()
            sol=-linalg.spsolve(J, q)
            self.time_solve.append(time.time()-t0)
            n=int(len(q)/2)
            # sol[self.wells['ws_p']]=0
            pressure+=sol[0:n]
            # sol[n+self.wells['ws_inj']]=0
            # sol[np.array([13421,13422])]=0
            swns+=sol[n:]

            # if count>8:
            #     print(np.where(abs(sol)>0.0000001))
            #     import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # swns[self.wells['ws_prod']]=0
            # visualize.plot_field(pressure)
            # swns[self.Assembler.wells['ws_inj']]=1
            converged=max(abs(sol))<rel_tol
            print(max(abs(sol[n:])),max(abs(sol[n:])),'fs')
            # import pdb; pdb.set_trace()
            self.PVI=(swns*self.porosities).sum()/self.porosities.sum()
            count+=1
            # swns[swns<0]=0
            # swn1s[swn1s>1]=1
            if count>20:
                print('excedded maximum number of iterations finescale')
                return False, count, pressure, swns
        # saturation[wells['ws_prod']]=saturation[wells['viz_prod']].sum()/len(wells['viz_prod'])
        return True, count, pressure, swns
