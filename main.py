
from packs.preprocessor.finescale_mesh import finescale_preprocess
from packs.processor.finescale_processing import NewtonIterationFinescale
from packs.processor.multiscale_and_multilevel import NewtonIterationMultilevel
import numpy as np
from packs.postprocessor.exporter import FieldVisualizer
visualize=FieldVisualizer()

wells, faces, volumes = finescale_preprocess()

finescale=NewtonIterationFinescale(wells, faces, volumes)
multilevel=NewtonIterationMultilevel(wells, faces, volumes)

p=np.zeros_like(volumes['GID_0']).astype(np.float64)
p[wells['ws_p']]=wells['values_p']
s=p.copy()
time_step=0.0005
wells['count']=0

multilevel.get_finescale_vols()
multilevel.update_NU_ADM_mesh()
multilevel.update_NU_ADM_operators()
# import pdb; pdb.set_trace()
# # visualize.plot_labels(multilevel.GID_1)
# visualize.plot_labels(multilevel.NU_ADM_ID)
# import pdb; pdb.set_trace()
# visualize.plot_labels(multilevel.GID_1)
# visualize.plot_labels(multilevel.GID_0)
# import pdb; pdb.set_trace()

# visualize.plot_field_plt(np.log10(volumes['Kxx']))
# visualize.plot_field(np.log10(volumes['Kxx']))
# visualize.plot_field
# visualize.plot_labels(multilevel.NU_ADM_OP[:,1].T.toarray()[0])
# visualize.plot_labels(ope[:,4].T.toarray()[0])

# visualize.plot_labels(multilevel.NU_ADM_OP[:,0].T.toarray()[0])
# visualize.plot_labels(multilevel.NU_ADM_OP[:,-1].T.toarray()[0])
# visualize.plot_field(multilevel.OP[:,1].T.toarray()[0])
# visualize.plot_labels(np.arange(len(volumes['GID_0'])))
# import pdb; pdb.set_trace()
# visualize.plot_field(volumes['GID_0'])
count=0
plots=np.arange(0,1000,20)

while True:
    conv=False
    while not conv:
        conv, fs_iters, p1, s1=multilevel.newton_iteration_ADM(p, s , time_step)
        # conv, fs_iters, p1, s1=finescale.newton_iteration_finescale(p, s , time_step)
        if fs_iters<5:
            print('increasing time_step from: {}, to: {}'.format(time_step, 1.5*time_step))
            time_step*=1.5
        elif fs_iters>15:
            print('reducing time_step from: {}, to: {}'.format(time_step, 0.8*time_step))
            time_step*=0.8
    p=p1.copy()
    s=s1.copy()
    count+=1
    if count in plots:
        # import pdb; pdb.set_trace()
        visualize.plot_field(multilevel.levels)

        visualize.plot_field(s)
# import pdb; pdb.set_trace()
