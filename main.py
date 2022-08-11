
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
time_step=0.000005
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
# plots=np.arange(0,10000,50)
vpi_max=0.7
pvis=np.arange(0,vpi_max+.00001,0.0005)
ind_pvi=0
# while True:
tadm=[]
tfs=[]
converged=False
time_steps=[]
simulation='multilevel'
# resolution='homogeneous'
case='label_1'
np.save('results/'+case+'/porosities.npy',volumes['pore_volume'])

ws_prod=wells['ws_prod']
ads=faces['adjacent_volumes']
viz=np.setdiff1d(np.unique(ads[(ads==ws_prod).sum(axis=1)==1]),ws_prod)
np.save('results/'+case+'/viz.npy',viz)
# import pdb; pdb.set_trace()
# vpis_for_npy=np.arange(0,0.70001,0.001)
saturations=[]
pressures=[]
n1_adm=[]
ind_npy=0
# import pdb; pdb.set_trace()
s[(volumes['centroids'][:,0]>30)&(volumes['centroids'][:,0]<37)]=1
s[(volumes['centroids'][:,0]<5)]=1
while not converged:
    conv=False
    while not conv:
        t0=time.time()
        if simulation=='multilevel':
            conv, fs_iters, p1, s1=multilevel.newton_iteration_ADM(p, s , time_step)
            act_pvi=multilevel.PVI
        time_steps.append(time_step)
        t1=time.time()
        if simulation=='finescale':
            conv, fs_iters, p1, s1=finescale.newton_iteration_finescale(p, s , time_step)
            act_pvi=finescale.PVI
            if conv:
                saturations.append(s1)
                pressures.append(p1)
            if (finescale.PVI>vpi_max) and conv:
                saturations=np.array(saturations)
                pressures=np.array(pressures)
                np.save('results/'+case+'/pressures_'+simulation+'.npy',pressures)
                np.save('results/'+case+'/saturations_'+simulation+'.npy',saturations)
                converged=True
        if simulation=='multilevel':
            conv, fs_iters, p1, s1=multilevel.newton_iteration_ADM(p1, s1 , time_step)
            act_pvi=multilevel.PVI
            if conv:
                saturations.append(s1)
                pressures.append(p1)
                n1_adm.append(multilevel.NU_ADM_ID.max())
            if (multilevel.PVI>vpi_max) and conv:
                saturations=np.array(saturations)
                pressures=np.array(pressures)
                n1_adm=np.array(n1_adm)
                np.save('results/'+case+'/pressures_'+simulation+'.npy',pressures)
                np.save('results/'+case+'/saturations_'+simulation+'.npy',saturations)
                np.save('results/'+case+'/n1_adm_'+simulation+'.npy',n1_adm)
                converged=True
        tadm.append(t1-t0)
        tfs.append(time.time()-t1)

        if act_pvi > pvis[ind_pvi]:
            ind_pvi+=1
            visualize.plot_field(p1,'Pressure')
            visualize.plot_field(s1,'Saturation')
            # visualize.plot_field_plt(multilevel.DUAL_1)
            if simulation=='multilevel':
                visualize.plot_field(multilevel.alphas,'Alpha')
                visualize.plot_field(multilevel.betas,'Beta')
                visualize.plot_field(np.log10(volumes['Kxx']),'Kxx=Kyy')
                visualize.plot_field(multilevel.levels,'Levels')
                visualize.plot_field(multilevel.GID_1,'GID_1')
            # visualize.grid.save('results/'+case+'/arqs/'+simulation+'_'+str(int(100*pvis[ind_pvi-1]))+'.vtk') #ativar

        if fs_iters<5:
            print('increasing time_step from: {}, to: {}'.format(time_step, 1.5*time_step))
            time_step*=1.3
        elif fs_iters>15:
            print('reducing time_step from: {}, to: {}'.format(time_step, 0.8*time_step))
            time_step*=0.8

    p=p1.copy()
    s=s1.copy()
    visualize.grid.save('results/'+case+'/arqs/'+simulation+'_'+str(count)+'.vtk')
    count+=1

# np.save('results/time_steps.npy',np.array(time_steps))
# np.save('results/times/proc_'+str(len(multilevel.GID_0))+'.npy',np.array(multilevel.proc_cumulative))
# np.save('results/times/prep_'+str(len(multilevel.GID_0))+'.npy',np.array(multilevel.prep_time))
# np.save('results/times/fs_'+str(len(multilevel.GID_0))+'.npy',np.array(finescale.time_solve))



# import pdb; pdb.set_trace()
