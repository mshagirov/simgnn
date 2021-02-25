"""
Functions for processing Y. Hara et al. amnioserosa dataset.
"""

import numpy as np

import skimage.io
from skimage import measure, morphology
import matplotlib
import matplotlib.pyplot as plt

def read_tiff_stack(filepath, trim_bound = True):
    '''
    Import tiff image stack using skimage.io.imread() w/ optional removal of image edges (set to zeroes).

    Arg-s:
    - filename  : path to the file (str)
    - trim_bound : Boolean, if True: set image boundary pixels to zeroes.

    Returns:
    imgstack : image stack, a numpy array of shape [frames, Y, X]
    '''
    imgstack = skimage.io.imread(filepath)# [t, Y, X]
    assert len(imgstack.shape)==3 # [t, Y, X]
    print(f'Image stack shape: {imgstack.shape} -- trim boundaries: {trim_bound}')
    if trim_bound:
        imgstack[:,0,:] = 0
        imgstack[:,-1,:]= 0
        imgstack[:,:,0] = 0
        imgstack[:,:,-1]= 0
    return imgstack


def rm_small_holes(imgstack, area_threshold=2,connectivity=1):
    '''Remove small holes in image slices/frames w/ `skimage.morphology.remove_small_holes`.'''
    for t in range(imgstack.shape[0]):
        imgstack[t,:,:] = 255*morphology.remove_small_holes(imgstack[t,:,:]==255,
                                                            area_threshold=area_threshold, connectivity=connectivity)
    return imgstack


def get_cell_colormap(max_label, cmap = plt.cm.nipy_spectral_r, cmap_0 = (.5,.5,.5,1.0) ):
    '''
    Set number of colour (bins) and first colour.

    Arg-s:
    - max_label : maximum value for labels
    - cmap : colormap from `plt.cm` {default : `plt.cm.nipy_spectral_r`}
    - cmap_0 : RBGA values for the first entry defaults to grey colour {default: (.5,.5,.5,1.0)}.
               Set it to `None` to keep the original RBGA.

    Returns:
    - cmap, norm : required for plotting

    Usage w/ imshow: `plt.imshow(labels,cmap=cmap,norm=norm)`
    '''
    # extract all colors from the map
    cmaplist = cmap(range(cmap.N))# or [cmap(i) for i in range(cmap.N)]

    if cmap_0 != None:
        cmaplist[0] = cmap_0 # first entry color, default:grey (.5,.5,.5,1.0)

    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,max_label-1,max_label)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def get_cell_labels(img):
    '''
    Labels non-boundary regions as cells w/ `skimage.measure.label`, omits background (labeled as 0).

    Arg-s:
    - img : 2-d numpy array, a single binary image values \in {0, 255} w/ shape HxW.

    Returns:
    - labels : labeled cells
    - cellxy : centroid (of each label): [x,y] (N,2)-shape
    '''
    # Label connected regions of an integer array
    labels = measure.label(img==0, connectivity=1, background=0) # con-y=1 <--> 4-neighbors ~ meaning "1 hop away"
    np.maximum(labels-1, 0, out=labels) # set cell boundaries to 0

    stats = measure.regionprops(labels)
    cellxy = np.array([xy0.centroid[-1::-1] for xy0 in stats])# idx=l-1

    return labels,cellxy


def t_dist(x, y, w=10):
    '''
    t-Distance kernel, where distance=1-f(r): 0 for Euclidean distance r=0, distance->1 as r->+/-inf.
    f(r) is the Student-t distrib. w/ one degree of freedom, f(r) \in (0,1] .

    Arg-s:
    - x,y: cartesian distance along x and y;
    - w  : width of the distribution at distance=0.5)[sensitivity]

    Returns:
    - distance, numpy array (same shape as x and y)
    '''
    return 1-1/(1+(x/w)**2+(y/w)**2)


def calculate_dist(xycurr, xynext, dstfunc = t_dist):
    '''
    Calculates distance using a given distance function.

    Arg-s:
    - xycurr : numpy array of shape Nx2 with cell locations at frame t
    - xynext : numpy array of shape Mx2 with cell locations at frame t+1
    - dstfunc : distance function: must accept 2 arg-s `dX`, and `dY`  {default: t-distance kernel}.

    Returns:
    - cellDists : label distances between frames (of each label) w/ a shape NxM.
    '''
    #[ frame t , frame t+1 ]
    dX = xycurr[:, 0].reshape( -1, 1).repeat( xynext.shape[0], axis = 1) - \
            xynext[:, 0].reshape( 1, -1).repeat( xycurr.shape[0], axis = 0)
    dY = xycurr[:, 1].reshape( -1, 1).repeat( xynext.shape[0], axis = 1) - \
            xynext[:, 1].reshape( 1, -1).repeat( xycurr.shape[0], axis = 0)
    return dstfunc(dX, dY)


def get_frame_id_pairs(dxyMat):
    '''
    Pair next frame centroids to the current frame centroids.

    Arg-s:
    - dxyMat : distance matrix (axis 1:centroid(t),axis 2:centroid(t+1) )

    Returns:
    - idpairs : 2-columns, col0:id(t) or -111 if there's no match, col1:id(t+1)
    '''
    closest_ids = np.argmin(dxyMat,axis=0) # ids from previous frame
    closest_dist = np.min(dxyMat,axis=0)    # minimum distance (metric)
    idpairs        = np.array([[ closest_ids[k], k] if closest_dist[k]<0.5 else  [ -111, k ] \
                               for k in range(closest_ids.shape[0])])
    #idpairs[:,0]: frame t ids : -111 if not found match in t
    #idpairs[:,1]: frame t+1 ids
    return idpairs


def get_new_cellxy_order(cellxy_tn, idpairs, NumOfPoints_t):
    '''
    Rearrange the centroids according to the previous frame

    Arg-s:
    - cellxy_tn : centroids in the frame t+1
    - idpairs   : index pairs between frames t and t+1 (from getFrameIdPairs() )
    - NumOfPoints_t : numb. of centroids in frame t (cellxy_t.shape[0])

    Returns:
    - newcellxy_tn : new order of cellxy_tn to match frame t.
    '''
    # Total Num of labels=#unpaired + #paired labels
    totalNumOfLabels = idpairs[idpairs[:,0]==-111,:].shape[0] + NumOfPoints_t
    newcellxy_tn     = -111*np.ones((totalNumOfLabels,2))# new labels
    # Update cell centroids to t+1:
    newcellxy_tn[ idpairs[ idpairs[:, 0] != -111, 0], :] = \
         cellxy_tn[ idpairs[ idpairs[ :, 0] != -111, 1] ]
    #check
    if idpairs[ idpairs[ :, 0] == -111, :].shape[0] != 0:
        newcellxy_tn[-idpairs[idpairs[:,0]==-111,:].shape[0] :,:] = \
        cellxy_tn[ idpairs[idpairs[:,0]==-111,1]]
    return newcellxy_tn


def relabel_w_previous(cellxy_t, cellxy_tn, labels_tn):
    '''
    Re-label "frame t+1" cells according to the "frame t" labels.

    Arg-s:
    - cellxy_t : centroids in frame t
    - cellxy_tn : centroids in frame t+1
    - labels_tn : labels in frame t+1

    Returns:
    - newlabels_tn : new labels of labels_tn to match cellxy_t
    - newcellxy_tn : re-arranged cellxy_tn to match cellxy_t
    '''
    dxyMat       = calculate_dist( cellxy_t, cellxy_tn) # calculate t-distance kernel

    # re-order the centroids:
    idpairs      = get_frame_id_pairs(dxyMat) # id: [t , t+1] pairs
    newcellxy_tn = get_new_cellxy_order(cellxy_tn, idpairs, cellxy_t.shape[0])

    newlabels_tn = np.zeros_like(labels_tn) # new labels

    label_counter=cellxy_t.shape[0]+1# new labels start from label_counter
    # cell labels start from 1 to cellxy_t.shape[0]+1 ->(no 0 label cells)
    for k in range(idpairs.shape[0]):
        #  k == idpairs[k,1]
        #label==1 and 0 (boundary) are background pixels
        if idpairs[k,0]==-111:
            newlabels_tn[labels_tn==k+1]=label_counter
            label_counter+=1
        else:
            newlabels_tn[labels_tn==k+1] = idpairs[k,0]+1
    return newlabels_tn, newcellxy_tn


def label_bw_stack(imgstack):
    '''
    Label cells and extract their locations in BW image stack, pixel intensities \in {0, 255} (binary image).

    Arg-s:
    - imgstack : BW stack of cell boundary images (uint8, "255"=boundary), numpy array of shape [frames, Y, X].

    Returns:
    - labelStack : bound ROI labels w/ shape [frames, height, width], with labels "0" (background), "1" to "N" cells
                   (dtype=np.uint64).
    - cellxyList : list of labeled ROI centroids, Nx2 arrays [x,y] of cell locations.
                   Row k corresponds to label "k+1", "-111" entry for [x,y]'s if no centroid is found.

    An example for plotting `labelStack`:
    ```
    cmap, cnorm = get_cell_colormap(int(label_imgs.max()+1))
    plt.imshow(labelStack[0], cmap=cmap, norm=cnorm) # show first frame
    ```
    '''
    labelStack = np.zeros(shape = imgstack.shape, dtype=np.uint64)
    #print('label_bw_stack : labeling '+str(imgstack.shape[0])+' frames')

    cellxyList = []

    # label 1st frame:
    [labelStack[0,:,:], cellxy_t] = get_cell_labels( imgstack[0,:,:] )
    cellxyList.append(cellxy_t.copy())

    for t in range(1, imgstack.shape[0]):
        # label cells, l==0:background
        [labels_tn, cellxy_tn] = get_cell_labels( imgstack[t,:,:] )
        [newlabels_tn, newcellxy_tn] = relabel_w_previous(cellxy_t, cellxy_tn, labels_tn)

        # update current frame's centroids and labels:
        cellxy_t = newcellxy_tn
        labelStack[t,:,:] = newlabels_tn
        cellxyList.append(newcellxy_tn.copy())
    return labelStack, cellxyList


def get_node_locations(labels,xbound,ybound):
    '''
    Find tri-cellular junction nodes in labelled images (labels==cells), for not labeled images see `get_node_mask()`.

    Nodes -- boundary point with more that 2 labels, i.e. more than 2 cells.

    Arg-s:
    - labels : labeled image of cells with integer pixel val-s ("0":background).
    - xbound : x location of boundaries in "labels".
    - ybound : y location of boundaries in "labels" w/ len(ybound)==len(xbound) .

    Returns:
    - nodeLocs : x,y locations, shape:(#nodes,2).
    - nodenames : node names, a list of tuples. "nodenames" -- each node is named by the tuple of cell labels that share the node.
    '''
    nodelocs = []
    nodenames = []
    for x,y in zip(xbound,ybound):
        neighbourLabels = np.unique(labels[y-1:y+2,x-1:x+2]) # unique labels in 3x3 neighbourhood
        if neighbourLabels.shape[0]>3:
            nodelocs.append([x, y])
            nodenames.append(tuple(neighbourLabels[neighbourLabels!=0])) # exclude "0" label
    nodelocs = np.array(nodelocs)
    return nodelocs, nodenames


def extract_nodes(imgstack, labelStack):
    '''
    Extract nodes from labeled cells and BW boundary stacks.

    Arg-s:
    - imgstack : boundary images (uint8, 255: boundary).
    - labelStack : labeled cell images (from labelBWstack() ), labels: cells.

    Returns:
    - node_dict : dictionary of node loc-s w/ tuples of cell labels as keys. Dict key-- vertex
                 represented in terms of cells (labels) sharing the vertex, e.g. for a vertex
                 shared among cells [i,j,k] : {(i, j, k): [x,y] np.array} where [i,j,k] are labels
                 from `labelStack`.
    '''
    # 2D XY grid for images
    Xgrid, Ygrid = np.meshgrid( np.arange( 0, imgstack.shape[2], 1),
                                np.arange( 0, imgstack.shape[1], 1) )

    # nodes in the 1st frame
    xbound = Xgrid[ imgstack[0,:,:] != 0].ravel() # x y loc-s of the boundaries (frame 0)
    ybound = Ygrid[ imgstack[0,:,:] != 0].ravel()
    nodelocs_t, nodenames_t = get_node_locations( labelStack[0,:,:], xbound, ybound)

    # create and load nodes dict; keys:"nodenames"==cell labels that share the node
    node_dict = dict( zip( nodenames_t,
                          np.full( (len(nodenames_t), labelStack.shape[0], 2), np.nan, dtype=np.float64)
                        ) )
    # Process and set node locations in the 1st frame.
    for keyi,val in zip( nodenames_t, nodelocs_t):
        node_dict[keyi][0,:] = val

    # find nodes in other frames (t>0)
    for t in range(1, labelStack.shape[0]):
        # get x y loc-s of the boundaries (frame t):
        xbound = Xgrid[ imgstack[t,:,:] != 0].ravel()
        ybound = Ygrid[ imgstack[t,:,:] != 0].ravel()
        nodelocs_t,nodenames_t = get_node_locations( labelStack[t,:,:], xbound, ybound)
        for keyi,val in zip(nodenames_t, nodelocs_t):
            if keyi in node_dict:
                node_dict[keyi][t,:] = val
            else:
                node_dict[keyi] = np.full( (labelStack.shape[0], 2), np.nan, dtype=np.float64)
                node_dict[keyi][t,:] = val
    return node_dict


def node_dict2graph(node_dict):
    '''
    Selects constant part of the cell monolayer graph (vertices and edges that are present in all frames).

    Arg-s:
    - node_dict : dictionary of node loc-s w/ tuples of cell labels as keys (must have
                  same format as the output of `simgnn.hara.extract_nodes()`).

    Returns:
    - edges_index : edge indices, rows representing "source" and "target" node indices.
    - node2cell_index : node-to-cell "edge indices", rows representing node indices and
                        cell indices (cell labels in `node_dict` keys)
    - node_pos : node positions w/ shape (num_of_frames)x(num_of_nodes)x2
    '''
    # nodes (keys) present in all frames
    v_names = [vn for vn in node_dict if ~np.any(np.isnan( node_dict[vn][:,0]))]

    # find edges -- pairs of node indices in v_names
    edges = np.array( [[ni, nj] for ni in range(len(v_names))
                       for nj in range(ni+1,len(v_names))
                       if len(set(v_names[ni]).intersection(v_names[nj]))>1])

    # reindex nodes and exclude nodes w/o edges
    v_newid = {v_i:l for l, v_i in enumerate([v_c for k, v_c in enumerate(v_names) if k in edges])}
    v_names_new = list(v_newid.keys())
    edges_index = np.array([[v_newid[v_names[e[0]]], v_newid[v_names[e[1]]]] for e in edges]).T

    # node_idx-to-cell_idx
    node2cell_index = np.concatenate([np.stack([np.full((len(v_c),), k, dtype=np.uint64), np.array(v_c)], axis=0)
                                      for k, v_c in enumerate(v_names_new)], axis=1)

    # vert positions array: #frames, #verts, #dims(==2)
    node_pos = np.stack([ node_dict[v_c] for v_c in v_names_new ],axis=1)

    return edges_index, node2cell_index, node_pos


def extract_graph(imgstack, labelStack):
    '''
    Converts BW cell boundary images into graphs. Selects constant part of the
    graph-- vertices and edges that are present in all frames.

    Arg-s:
    - imgstack : boundary images (uint8, 255: boundary).
    - labelStack : labeled cell images (from labelBWstack() ), labels: cells.

    `imgstack` and `labelStack` shapes must be same:(num_of_frames)xHeightxWidth.

    Returns:
    - edges_index : edge indices, rows representing "source" and "target" node indices.
    - node2cell_index : node-to-cell "edge indices", rows representing node indices and
                        cell indices (cell labels in `node_dict` keys)
    - node_pos : node positions w/ shape (num_of_frames)x(num_of_nodes)x2

    Plotting example: plotting 10th edge from 2nd frame
    ```
    plt.plot( node_pos[2, edges_index[0,10], 0], node_pos[2, edges_index[1,10], 1] )
    ```
    '''
    return node_dict2graph( extract_nodes(imgstack, labelStack))
