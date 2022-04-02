
from packs.preprocessor.finescale_mesh import finescale_preprocess
from packs.processor.finescale_processing import NewtonIterationFinescale
from packs.processor.multiscale_and_multilevel import NewtonIterationMultilevel
import numpy as np
from packs.postprocessor.exporter import FieldVisualizer

import time

visualize=FieldVisualizer()

wells, faces, volumes = finescale_preprocess()

finescale=NewtonIterationFinescale(wells, faces, volumes)
multilevel=NewtonIterationMultilevel(wells, faces, volumes)

p=np.zeros_like(volumes['GID_0']).astype(np.float64)
p[wells['ws_p']]=wells['values_p']
s=p.copy()
time_step=0.0005
wells['count']=0
'''
multilevel.get_finescale_vols()
multilevel.update_NU_ADM_mesh()
multilevel.update_NU_ADM_operators()
'''
# import pdb; pdb.set_trace()
# # visualize.plot_labels(multilevel.GID_1)
# visualize.plot_labels(multilevel.NU_ADM_ID)
# import pdb; pdb.set_trace()
# visualize.plot_labels(multilevel.GID_1)
# visualize.plot_labels(multilevel.GID_0)
# import pdb; pdb.set_trace()
#
# visualize.plot_field_plt(np.log10(volumes['Kxx']))
# visualize.plot_field(np.log10(volumes['Kxx']))
# visualize.plot_field

# visualize.plot_labels(multilevel.OP_matrix[:,1].T.toarray()[0])
# visualize.plot_labels(ope[:,4].T.toarray()[0])

# visualize.plot_labels(multilevel.NU_ADM_OP[:,0].T.toarray()[0])
# visualize.plot_labels(multilevel.NU_ADM_OP[:,-1].T.toarray()[0])
# visualize.plot_field(multilevel.OP[:,1].T.toarray()[0])
# visualize.plot_labels(np.arange(len(volumes['GID_0'])))
# import pdb; pdb.set_trace()
# visualize.plot_field(volumes['GID_0'])
count=0
plots=np.arange(0,1000,10)

# while True:
tadm=[]
tfs=[]
converged=False
time_steps=[]
while not converged:
    conv=False
    while not conv:
        t0=time.time()
        conv, fs_iters, p1, s1=multilevel.newton_iteration_ADM(p, s , time_step)
        time_steps.append(time_step)
        t1=time.time()
        # conv, fs_iters, p1, s1=finescale.newton_iteration_finescale(p, s , time_step)
        tadm.append(t1-t0)
        tfs.append(time.time()-t1)
        if count in plots:
            visualize.plot_field(s)
        if fs_iters<5:
            print('increasing time_step from: {}, to: {}'.format(time_step, 1.5*time_step))
            time_step*=1.3
        elif fs_iters>15:
            print('reducing time_step from: {}, to: {}'.format(time_step, 0.8*time_step))
            time_step*=0.8

        try:
            if multilevel.PVI>0.15:
                converged=True
            print(multilevel.PVI,"PVI")
        except:
            if finescale.PVI>0.15:
                converged=True
            print(finescale.PVI,"PVI")


    p=p1.copy()
    s=s1.copy()
    count+=1

# np.save('results/time_steps.npy',np.array(time_steps))
# np.save('results/times/proc_'+str(len(multilevel.GID_0))+'.npy',np.array(multilevel.proc_cumulative))
# np.save('results/times/prep_'+str(len(multilevel.GID_0))+'.npy',np.array(multilevel.prep_time))
# np.save('results/times/fs_'+str(len(multilevel.GID_0))+'.npy',np.array(finescale.time_solve))



# import pdb; pdb.set_trace()
