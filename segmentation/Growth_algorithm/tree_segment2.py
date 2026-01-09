import laspy as lp
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

def read_file(path):
    return lp.read(path)

import numpy as np
def extract_cord_arrays(data, z_scale=3):
    xs, ys, zs, id = np.array(data.x), np.array(data.y), np.array(data.z), np.array(data.treeID)
    
    if 'treeID' in list(data.point_format.dimension_names):
        mask = np.logical_and(data.classification>3,data.treeID!=0)
    else:
        mask = data.classification>3

    xs = xs[mask]
    ys = ys[mask]
    zs = zs[mask]
    id = id[mask]

    xs -= np.min(xs)
    xs /= np.max(xs)

    ys -= np.min(ys)
    ys /= np.max(ys)

    zs -= np.min(zs)
    zs /= np.max(zs)
    zs /= z_scale

    return xs, ys, zs, id

from sklearn.cluster import DBSCAN
def cluster_stems(xs,ys):
    X = np.vstack((xs,ys)).T
    labels = DBSCAN(eps=0.03,min_samples=5).fit_predict(X)
    return labels

def calculate_tree_origins(xs,ys,zs,labels):
    X = np.vstack((xs,ys,zs)).T

    unique_labels = set(labels)

    tree_origins = []
    for label in unique_labels:
        mask = labels==label
        tree_origins.append(np.concat([np.sum(X[mask,:],axis=0)/np.sum(mask),[label]]))

    return np.vstack(tree_origins)

import heapq
def initialize_pq(xs,ys,zs,tree_origins):
    pq = []
    for index in range(len(xs)):
        x, y, z = xs[index], ys[index], zs[index]
        d = ((x-tree_origins[:,0])**2+(y-tree_origins[:,1])**2+(z-tree_origins[:,2])**2)**0.5
        d_i = np.argmin(d)
        heapq.heappush(pq, (d[d_i], d_i, index))
    return pq

def initialize_buffer(xs,ys,zs,pq,tree_origins,start_buffer = 10000):
    points_buffer = []
    for _ in range(start_buffer):
        d, d_i, i = heapq.heappop(pq)
        x, y, z = xs[i], ys[i], zs[i]
        points_buffer.append([(x,y,z),d,tree_origins[d_i,3],i])

    points_buffer.sort(key=lambda x:x[1])
    return points_buffer


def binary_search(sorted, insert, key=None):
    if key is None:
        key = lambda x:x

    low = 0
    high = len(sorted)-1

    while high-low>1:
        mid = (low+high)//2
        value = key(sorted[mid])
        if insert<value:
            high = mid
        else:
            low = mid
    
    if insert>key(sorted[high]):
        return high+1
    elif insert<key(sorted[low]):
        return high-1
    else:
        return high
    
def insert_sorted(sorted,insert,key=None):
    if key is None:
        key = lambda x:x

    i = binary_search(sorted,insert)
    sorted.insert(i,insert)

def average_n_d(points, n):
    s = 0
    c = 0
    for i in range(n):
        try:
            s += points[i][1]*1/(i+1)
        except:
            pass
        c+=1/(i+1)
    return s/c

def add_to_buffer(point_buffer, pq, xs, ys, zs, tree_origins, buffer_size=2000):
    while (average_n_d(point_buffer,100)>0.02 or len(point_buffer)<=300) and pq and len(point_buffer)<buffer_size:
        d, d_i, i = heapq.heappop(pq)
        x0, y0, z0 = xs[i], ys[i], zs[i]
        min_d = d
        min_l = tree_origins[d_i,3]
        for (x1, y1, z1), _, l, _ in point_buffer:
            if abs(x1-x0) > min_d:continue
            if abs(y1-y0) > min_d:continue
            if abs(z1-z0) > min_d:continue
            d  = ((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**0.5
            if d<min_d:
                min_d = d
                min_l = l

        entry = [(x0,y0,z0),min_d,min_l,i]
        insert_sorted(point_buffer,entry,key=lambda x:x[1])
        
def update_buffer(point_buffer,x0,y0,z0,l):
    i = -1
    changed = False
    while i<len(point_buffer)-1:
        i+=1
        (x1, y1, z1), d1, _, _ = point_buffer[i]
        if abs(x1-x0) > d1:continue
        if abs(y1-y0) > d1:continue
        if abs(z1-z0) > d1:continue
        dist = ((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**0.5
        if dist<d1:
            point_buffer[i][1]=dist
            point_buffer[i][2]=l
            changed = True
        
    if changed:
        point_buffer.sort(key=lambda x:x[1])

def list2labels(ls_list,is_list, length):
    unsorted_labels = np.array(ls_list)
    indicies = np.array(is_list)
    
    labels = np.zeros(length, dtype=unsorted_labels.dtype)
    labels[:] = np.nan
    labels[indicies] = unsorted_labels

    return labels

from tqdm import tqdm
def grow_tree(xs, ys, zs, tree_origins, pl):
    # Initialise priority queue.
    pq = initialize_pq(xs,ys,zs,tree_origins)

    # Initialise progressbar.
    length = len(pq)
    pbar = tqdm(total = length, smoothing=0.05)

    # Initialise point buffer.
    point_buffer = initialize_buffer(xs,ys,zs,pq,tree_origins)

    # Initialise finialized lists.
    ls_list = []
    is_list = []

    ari = []
    ami = []

    counter = 0
    while point_buffer:
        # Add new points to buffer, if needed.
        add_to_buffer(point_buffer, pq, xs, ys, zs, tree_origins)

        # Remove first point from buffer.
        (x0, y0, z0), d0, l, i = point_buffer.pop(0)

        # Append to output.
        ls_list.append(l)
        is_list.append(i)

        counter+=1
        if counter % (length//20) == 0:
            labels = list2labels(ls_list,is_list,length)
            ari.append(ARI(labels[~np.isnan(labels)],pl[~np.isnan(labels)]))
            ami.append(AMI(labels[~np.isnan(labels)],pl[~np.isnan(labels)]))

        # update progress.
        pbar.update(1)

        # Update buffer.
        update_buffer(point_buffer, x0, y0, z0, l)

    labels = list2labels(ls_list,is_list,length)

    pbar.close()

    return labels, ari, ami

def segment(path, z_scale=3):
    
    # Read the file
    data = read_file(path)
    
    # Extract and normalize to numpy arrays, z-scale is weight added to z dimmensions.
    xs, ys, zs, true_labels = extract_cord_arrays(data,z_scale=z_scale)

    # Devides z axis into histogram.
    freqs, edges = np.histogram(zs,bins=10)

    # Find the smallest bin that is before the largest bin (ignoring first 4 as we assume that is the ground plane)
    i = np.argmin(freqs[:np.argmax(freqs)])

    # Samples all points in the samllest bins, before largest bin. Largest bins is assumed to be the tree crowns and the smallest bin before that should be only the stems.
    mask = np.logical_and(zs>edges[i],zs<edges[i+1])
    sampled_xs = xs[mask]
    sampled_ys = ys[mask]
    sampled_zs = zs[mask]

    # Clustes these stems
    labels = cluster_stems(sampled_xs,sampled_ys)

    # Calulates middle of tree, using average of each stem cluster.
    tree_origins = calculate_tree_origins(sampled_xs,sampled_ys,sampled_zs,labels)

    # Grow tree
    predicted_labels, ari, ami = grow_tree(xs,ys,zs,tree_origins, true_labels)

    return predicted_labels, true_labels, ari, ami