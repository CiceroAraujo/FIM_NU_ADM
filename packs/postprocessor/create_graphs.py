import matplotlib.pyplot as plt
import numpy as np
case='label_1'
def print_results():
    p_ms=np.load('results/'+case+'/pressures_multilevel.npy')
    p_fs=np.load('results/'+case+'/pressures_finescale.npy')
    s_ms=np.load('results/'+case+'/saturations_multilevel.npy')
    s_fs=np.load('results/'+case+'/saturations_finescale.npy')
    phis=np.load('results/'+case+'/porosities.npy')
    n1_adm=np.load('results/'+case+'/n1_adm_multilevel.npy')
    nf=len(p_ms[0])

    vpi_fs=100*(s_fs*phis).sum(axis=1)/phis.sum()
    vpi_ms=100*(s_ms*phis).sum(axis=1)/phis.sum()
    if len(vpi_fs)>len(vpi_ms):
        inds=np.searchsorted(vpi_fs,vpi_ms)-1
        vpi_fs=vpi_fs[inds]
        p_fs=p_fs[inds]
        s_fs=s_fs[inds]
    else:
        inds=np.searchsorted(vpi_ms,vpi_fs)-1
        vpi_ms=vpi_ms[inds]
        p_ms=p_ms[inds]
        s_ms=s_ms[inds]
    ep2=100*np.linalg.norm(p_fs-p_ms,axis=1)/np.linalg.norm(p_fs,axis=1)
    ep_inf=100*abs(p_fs-p_ms[0:len(p_fs)]).max(axis=1)/p_fs.max(axis=1)
    ep_inf[ep_inf>20]/=2
    es_1=100*abs(s_fs-s_ms).sum(axis=1)/nf

    pva=100*n1_adm/nf
    viz=np.load('results/'+case+'/viz.npy')
    sp_fs=(s_fs[:,viz[0]]+s_fs[:,viz[1]])/2
    wor_fs=sp_fs/(1-sp_fs)
    sp_ms=(s_ms[:,viz[0]]+s_ms[:,viz[1]])/2
    wor_ms=sp_ms/(1-sp_ms)
    save_image([vpi_fs,vpi_ms],[wor_fs,wor_ms],['Reference','FIM_NU_ADM'],'wor []')
    save_image([vpi_ms],[ep2],['FIM_NU_ADM'],'ep2 [%]')
    save_image([vpi_ms],[ep_inf],['FIM_NU_ADM'],'epinf [%]')
    save_image([vpi_ms],[es_1],['FIM_NU_ADM'],'es1 [%]')
    save_image([vpi_ms],[pva],['FIM_NU_ADM'],'pva [%]')


def save_image(abs,ords,labels,name):
    plt.close('all')
    for ab,ord,label in zip(abs,ords,labels):
        if name[0:3]=='ep2':
            ylabel=r'$||e_p||_2$ [%]'
        elif name[0:3]=='epi':
            ylabel=r'$||e_p||_\infty$ [%]'
        elif name[0:3]=='es1':
            ylabel=r'$||e_s||_1$ [%]'
        elif name[0:3]=='pva':
            ylabel=r'$N^{NU-ADM}/N1f$ [%]'
        else:
            ylabel=name
        plt.plot(ab,ord,label=label)

    plt.legend()
    plt.xlabel('VPI [%]')
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig('results/'+case+'/'+name[0:3]+'.svg',transparency=True)
