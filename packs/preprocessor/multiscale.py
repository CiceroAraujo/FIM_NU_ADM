import numpy as np
from scipy.sparse import csc_matrix, csgraph
import scipy.sparse as sp
from .. import inputs

def get_primal_and_dual_meshes(centroids, faces):
    GID_1, DUAL_1 = get_dual_and_primal_1(centroids)
    # dual_preps = get_dual_structure(DUAL_1, faces['adjacent_volumes'], 2)
    OP_AMS = get_prolongation_operator(DUAL_1, GID_1, faces)
    lcd_vertices = get_vertices_OP(DUAL_1, GID_1)
    # lcd_edges = get_edges_OP(DUAL_1, GID_1, lcd_vertices)
    # import pdb; pdb.set_trace()
    return GID_1, DUAL_1

def get_prolongation_operator(DUAL_1, GID_1, faces):
    adjs = faces['adjacent_volumes']
    ts = faces['permeabilities']
    lcd_vertices = get_vertices_OP(DUAL_1, GID_1)
    ds, local_ID= get_dual_structure(DUAL_1, faces['adjacent_volumes'], 2)
    lcd_edges = get_edges_OP(faces, ds, lcd_vertices, DUAL_1, local_ID)

@profile
def get_edges_OP(faces, entities, lcd_vertices, DUAL_1, local_ID):

    for i in range(len(entities[0])):
        group=entities[0][i]
        ev=entities[1][i]
        adjs_evs = faces['adjacent_volumes'][ev]
        edges_ev=DUAL_1[adjs_evs]==2
        adjs_ev = adjs_evs[edges_ev]
        ts_ev = faces['permeabilities'][ev]
        adjs = faces['adjacent_volumes'][group]
        ts = faces['permeabilities'][group]
        lines=local_ID[np.concatenate([adjs[:,0], adjs[:,1], adjs[:,0], adjs[:,1], adjs_ev])]
        cols=local_ID[np.concatenate([adjs[:,1], adjs[:,0], adjs[:,0], adjs[:,1], adjs_ev])]
        data=np.concatenate([ts, ts, -ts, -ts, -ts_ev])
        EE=csc_matrix((data, (lines, cols)), shape=(lines.max()+1,cols.max()+1))

        vertices_IDs=adjs_evs[~edges_ev]
        linesev=local_ID[adjs_ev]
        colsev=np.arange(len(linesev))
        dataev=ts_ev
        EV=csc_matrix((dataev, (linesev, colsev)), shape=(EE.shape[0],colsev.max()+1))
        Pev=-sp.linalg.spsolve(EE,EV)

        lcd_faces=sp.find(Pev)
        uadjs=np.unique(adjs)
        lines=uadjs[lcd_faces[0]]
        cols=vertices_IDs[lcd_faces[1]]
        data=lcd_faces[2]
        import pdb; pdb.set_trace()
# @profile
def get_dual_and_primal_1(centroids):
    maxs=centroids.max(axis=0)
    mins=centroids.min(axis=0)
    cr1=np.array(inputs.multiscale_and_multilevel_inputs['multiscale']['coarsening_ratio_1'])
    n_blocks=np.array(inputs.finescale_inputs['mesh_generation_parameters']['n_blocks'])
    block_size=np.array(inputs.finescale_inputs['mesh_generation_parameters']['block_size'])
    n_duals=n_blocks//cr1
    second_line = (n_blocks-n_duals*cr1)//2
    xd=[]
    for i in range(3):
        xd.append(np.arange(second_line[i]+cr1[i],n_duals[i]*cr1[i],cr1[i]))
        if len(xd[i])>1:
            xd[i]=np.unique(np.concatenate([[mins[i], maxs[i]],(xd[i]+0.5)*block_size[i]]))
        else:
            xd[i]=np.unique(np.array([mins[i], maxs[i]]))

    d=np.zeros(len(centroids))
    for i in range(3):
        for x in xd[i]:
            d+=centroids[:,i]==x
    DUAL_1=d

    volumes=np.arange(len(centroids))
    GID_1=-np.ones_like(volumes)
    xp=np.array(mins[0]-1)
    xp=np.append(xp,(xd[0][:-1]+xd[0][1:])/2)
    xp=np.append(xp,maxs[0]+1)

    yp=np.array(mins[1]-1)
    yp=np.append(yp,(xd[1][:-1]+xd[1][1:])/2)
    yp=np.append(yp,maxs[1]+1)

    zp=np.array(mins[2]-1)
    zp=np.append(zp,(xd[2][:-1]+xd[2][1:])/2)
    zp=np.append(zp,maxs[2]+1)
    count=0
    for i in range(len(xp)-1):
        vx=volumes[(centroids[:,0]>=xp[i]) & (centroids[:,0]<xp[i+1])]
        for j in range(len(yp)-1):
            vy=vx[(centroids[:,1][vx]>=yp[j]) & (centroids[:,1][vx]<yp[j+1])]
            for k in range(len(zp)-1):
                vz=vy[(centroids[:,2][vy]>=zp[k]) & (centroids[:,2][vy]<zp[k+1])]
                GID_1[vz]=count
                count+=1
    return GID_1, DUAL_1

def get_vertices_OP(DUAL_1, GID_1):
    all_volumes=np.arange(len(DUAL_1))
    vertices=DUAL_1==3
    gid0=all_volumes[vertices]
    gid1=GID_1[vertices]
    data=np.ones_like(gid1)
    lcd=np.vstack([gid0, gid1, data])
    return lcd

@profile
def get_dual_structure(DUAL_1, adjs, entity):
    dual_adjs=DUAL_1[adjs]
    all_faces=np.arange(len(adjs))
    ee=(dual_adjs==entity).sum(axis=1)==2
    ev=adjs[ee]
    edges=np.unique(ev)
    mapd=np.arange(adjs.max()+1)
    mapd[edges]=np.arange(len(edges))
    adjs0=adjs[:,0][ee]
    adjs1=adjs[:,1][ee]
    lines=np.concatenate([mapd[adjs0],mapd[adjs1]])
    cols=np.concatenate([mapd[adjs1],mapd[adjs0]])
    data=np.ones(len(lines))
    graph=csc_matrix((data,(lines,cols)),shape=(len(edges),len(edges)))
    n_l,labels=csgraph.connected_components(graph,connection='strong')

    asort=np.argsort(labels)
    slabels=labels[asort]
    sedges=edges[asort]
    pos=np.array([0])
    pos=np.append(pos,np.arange(len(labels)-1)[(-slabels[:-1]+slabels[1:])==1]+1)
    pos=np.append(pos,len(labels))
    vs=np.concatenate([range(pos[i]-pos[i-1]) for i in range(1,len(pos))])

    entity_ID=-np.ones(len(DUAL_1))
    entity_ID[edges]=labels
    local_ID=entity_ID.copy()
    local_ID[sedges]=vs

    entities=[]
    for i in range(entity, 4):
        faces=all_faces[(dual_adjs.min(axis=1)==entity) & (dual_adjs.max(axis=1)==i)]
        ents=entity_ID[adjs[faces]].max(axis=1).astype(int)
        asort=np.argsort(ents)
        sents=ents[asort]
        sfaces=faces[asort]
        pos=np.arange(len(faces)-1)[(-sents[:-1]+sents[1:])==1]
        entities.append(np.split(sfaces,pos+1))
    return entities, local_ID.astype(int)
