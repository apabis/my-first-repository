import mdtraj
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

traj_id = "Nprot_docked_amber"
title = "Nprot_docked_amber_test"
data_tag = 'Nprot_docked_amber_test'
no_replicas = 18
start_time = 0.0
step = 200.0 # ps per md frame
max_conf = 1000 # max number of conformations that can be handled to efficiently calculate RMSDs
n_clusters = 6
n_ref = 4 # number of reference models to use for comparison

# Load trajectories to analyse

trajs = []
for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]:
    print(i)
#for i in range(1,no_replicas+1):
    trajs.append(mdtraj.load('../' + str(i) + '/peptide_conf/all_novsite_fit.xtc', top='../frame0_chainA_novsite.pdb'))
topology = mdtraj.load_topology('../frame0_chainA_novsite.pdb')

# Load reference structures

sys1 = mdtraj.load_topology('pept_exp_models/semiclosed23_new1.pdb')
sys2 = mdtraj.load_topology('pept_exp_models/Tamm_1.pdb')
sys3 = mdtraj.load_topology('pept_exp_models/Bax_cut_20aa_1.pdb')
sys4 = mdtraj.load_topology('pept_exp_models/1xop_straight_helix_model1.pdb')

ref1 = mdtraj.load('pept_exp_models/semiclosed23_new1.pdb', atom_indices=sys1.select('resid 0 to 19 and backbone'))
ref2 = mdtraj.load('pept_exp_models/Tamm_1.pdb', atom_indices=sys2.select('backbone'))
ref3 = mdtraj.load('pept_exp_models/Bax_cut_20aa_1.pdb', atom_indices=sys3.select('backbone'))
ref4 = mdtraj.load('pept_exp_models/1xop_straight_helix_model1.pdb', atom_indices=sys4.select('backbone'))

##
#Enable lookups like
#    index = replica[replica_id]['time'].index(target_time)
#    # step_number = replica[replica_id]['step_number'][index]
#    cluster_id = replica[replica_id]['peptides'][peptide_id][index]
#
# E.g.
# replica = {}
# replica[replica_id] = {}
# replica[replica_id]['time'] = []
# replica[replica_id]['step_number'] = []
# replica[replica_id]['peptides'] = {}
# replica[replica_id]['peptides'][peptide_id] = {'cluster_id': []}
## It is okay to keep some redundant information for cross-checking or future changes to access pattern
# replica[replica_id]['peptides'][peptide_id] = {'time': []}
#

# Assuming you are analyzing one simulation trajectory at a time, something like
# analyzed_sim = {}
# analyzed_sim['time'] = []
#analyzed_sim['peptides'] = {'peptide1': {'cluster_id': []},
#...
#
# import json
# datastore = 'datastore.json'
# with open(datastore, 'w') as fh:
#     json.dump(replica, fh)
# with open(datastore, 'r') as fh:
#     replica = json.load(fh)
# # when updating...
# # determine the replica_id of the thing I'm about to do.
# if replica_id in replica:
#     # recurse and check bits and pieces
#     # check if each peptide is there, then check if each time is there, add
#     # extra peptides or extra time as discovered...
# else:
#     replica[replica_id] = # some new stuff
##

# replica lenghts
rep_len = []
for i in range(no_replicas):
    rep_len.append(len(trajs[i]))

traj = trajs[0]
for i in range(1,no_replicas):
    traj = traj.join(trajs[i])

no_pept = 9 * no_replicas
del trajs

md_frames = [rep_len[i]/9 for i in range(no_replicas)] # no. frames for each peptide (=no. frames in each MD replica)

time = [[] for x in range(no_replicas)] # time for each frame in each replica
time_merged = [] # concatenated
for rep in range(no_replicas):
    for f in range(md_frames[rep]):
        time[rep].append(start_time+f*step) # in pico seconds
    time_merged = time_merged + time[rep]

# An array with replica labels for each frame in each replica

rep_id = [[] for x in range(no_replicas)]
rep_id_merged = []
for rep in range(no_replicas):
    for i in range(rep_len[rep]):
        rep_id[rep].append(rep)
    rep_id_merged = rep_id_merged + rep_id[rep]

# An array with peptide labels for each frame

pept_id = [[] for x in range(no_replicas)]
pept_id_merged = []
for rep in range(no_replicas):
    for pept in range(9):
        for i in range(md_frames[rep]*pept,md_frames[rep]*pept+md_frames[rep]):
            pept_id[rep].append(pept)
    pept_id_merged = pept_id_merged + pept_id[rep]


backbone = np.array(traj.topology.select('(resid 0 to 19 and backbone and not element H)'))
all_heavy = np.array(traj.topology.select('(resid 0 to 19 and not element H)'))

# An array that will store cluster assignments for all the frames in their original order (as in trajectory)

all_labels = np.empty((len(traj),1))

# Select a subset of conformations for clustering

max_conf_rep = max_conf/no_replicas
max_conf_pept = max_conf/no_pept # max conformations of each peptide 
max_conf_pept_rep = max_conf_rep/9 # max conformations of each peptide per replica

# Calculate stride for all replicas

stride = []
for rep in range(no_replicas):
    if max_conf_pept_rep == 1:
        stride_per_pept = 0
    else:
        stride_per_pept = md_frames[rep]/(max_conf_pept_rep-1)-1
    stride.append(stride_per_pept)


# Get frame IDs within idividual replics 
# and a "global" frame IDs for each frame in each replica, but corresponding to concatenated trajectory

global_fr_shift = 0
global_frame_id = [[] for x in range(no_replicas)]
replica_frame_id = [[] for x in range(no_replicas)]
for rep in range(no_replicas):
    global_fr_shift = 0
    if rep == 0:
        global_fr_shift = 0
    else:
        for j in range(rep):
            global_fr_shift = global_fr_shift + rep_len[j]
    for i in range(rep_len[rep]):
        replica_frame_id[rep].append(i)
        global_frame_id[rep].append(i+global_fr_shift)


# An array with actual MD frame for each frame of the concatenated trajectory
# In the concatenated trajectory the no. frames is actual MD frames * 9 (no. peptides)

md_frame_id = [[] for x in range(no_replicas)]
md_frame_id_merged = []
for rep in range(no_replicas):
    for pept in range(9):
        for i in range(md_frames[rep]*pept,md_frames[rep]*pept+md_frames[rep]):
            if 0 >= i >= md_frames[rep]:
                md_frame = i
            else:
                md_frame = i - pept*md_frames[rep]
            md_frame_id[rep].append(md_frame)
    md_frame_id_merged = md_frame_id_merged + md_frame_id[rep]

# Select frames from original trajectory that will be used for clustering (from each replica individually)

if len(traj) > max_conf:
    sel_conf_reps = [[] for x in range(no_replicas)]
    for rep in range(no_replicas):
        for i in range(9):
            start = i*md_frames[rep]
            stop = start + md_frames[rep]
            for j in range(max_conf_pept_rep):
                sel_conf_reps[rep].append(start+j*stride[rep])

# Get global frame IDs for selected frames
# (corresponding to concatenated trajectory)

sel_conf = []
for rep in range(no_replicas):
    for i in range(len(sel_conf_reps[rep])):
        sel_conf.append(global_frame_id[rep][sel_conf_reps[rep][i]])
print(sel_conf)

# Get replica ID and peptide ID for the selected frames
rep_id_sel = []
pept_id_sel = []
for i in range(len(sel_conf)):
    rep_id_sel.append(rep_id_merged[sel_conf[i]])
    pept_id_sel.append(pept_id_merged[sel_conf[i]])

# Copy frames for clustering to a new trajectory

sel_traj = traj.slice(np.int_(sel_conf))

# Get frames of remaining conformations

free_conf = []
for i in range(len(traj)):
        if i not in sel_conf[:]:
            free_conf.append(i)

free_conf = np.array(free_conf)

no_conf = len(sel_conf)

# Calculate RMSDs
# pairwise between selected conformations

rmsd_pair = np.empty((no_conf,no_conf))
for i in range(no_conf):
    rmsd_pair[i] = mdtraj.rmsd(sel_traj, sel_traj, i, atom_indices=backbone)
print(np.shape(rmsd_pair))
print('Max pairwise rmsd: %f nm' % np.max(rmsd_pair))

# against a single structure

reference = mdtraj.load('../frame0_chainA_novsite.pdb')
rmsd_single = mdtraj.rmsd(traj, reference, frame=0, atom_indices=backbone)

plt.plot(rmsd_single)

frame_all = np.arange(len(traj))
plt.scatter(frame_all,rmsd_single,c=rep_id_merged,s=1.5)

frame_all = np.arange(len(traj))
plt.scatter(frame_all,rmsd_single,c=pept_id_merged,s=1.5)

# # Average linkage using scikit-learn


clustering = AgglomerativeClustering(affinity='precomputed',linkage='average',
                                     n_clusters=n_clusters).fit(rmsd_pair)

frame_indx = np.arange(len(sel_traj))

# Map the cluster assignments to the original order of frames and add them to all_labels

for i in range(len(sel_conf)):
    all_labels[sel_conf[i],0] = clustering.labels_[i]


# Get all the frames belonging to a given cluster (cluster_frames)
# Get replica id and peptide id for each member of the clusters

members = [[] for x in range(n_clusters)]
members_pept_id = [[] for x in range(n_clusters)]
members_rep_id = [[] for x in range(n_clusters)]
for cl_no in range(n_clusters):
    for i in range(len(sel_traj)):
        if clustering.labels_[i] == cl_no:
            members[cl_no].append(i)
            members_pept_id[cl_no].append(pept_id_merged[sel_conf[i]])
            members_rep_id[cl_no].append(rep_id_merged[sel_conf[i]])

# rmsd pair calculation within each cluster
centroid=[]
rmsd_pair_cl = [[] for x in range(n_clusters)]
for cl in range(n_clusters):
    rmsd_pair_cl[cl]=np.empty((len(members[cl]),len(members[cl])))
    for i in range(len(members[cl])):
        rmsd_pair_cl[cl][i] = mdtraj.rmsd(sel_traj.slice(members[cl]), sel_traj.slice(members[cl]), i, atom_indices=backbone)
    sumsq = (rmsd_pair_cl[cl]**2).sum(axis=1)
    centroid.append(members[cl][sumsq.argmin()]) # atom indices from sel_traj, not the main traj

# Need to map this to the actual frame ID in the original trajectory
for c in range(0,n_clusters):
    centroid[c] = sel_conf[centroid[c]]

print(centroid)

# MDS on individual clusters, only including the pre-selected conforomations

mds_all_clust = [[] for x in range(n_clusters)]
for i in range(n_clusters):
    x = (rmsd_pair_cl[i]+rmsd_pair_cl[i].T)/2
    embedding = MDS(n_components=3,dissimilarity='precomputed')
    mds_all_clust[i] = embedding.fit(x).embedding_

print(np.shape(rmsd_pair))
rmsd_pair.shape

def check_symmetric(a, tol=1e-9):
    return np.allclose(a, a.T, atol=tol)

rmsd_pair_symm = (rmsd_pair+rmsd_pair.T)/2

# MDS on frames selected for clustering (all together, not separated into clusters)

embedding = MDS(n_components=3,dissimilarity='precomputed')
rmsd_mds = embedding.fit(rmsd_pair_symm)

frame_indx = np.arange(len(sel_traj))

# Map the cluster assignments to the original order of frames and add them to all_labels
for i in range(len(sel_conf)):
    all_labels[sel_conf[i]] = clustering.labels_[i]

# Get rmsd to center of clusters for all frames

rmsd_to_center = np.empty((n_clusters,len(traj)))
for c in range(n_clusters):
    reference = traj.slice(centroid[c])
    rmsd_to_center[c] = mdtraj.rmsd(traj, reference, atom_indices=backbone)

# Get rmsd values to centers of clusters for selected frames (grouped by clusters)

rmsd_sel = [[] for x in range(n_clusters)]
for i in range(len(sel_conf)):
        rmsd_sel[clustering.labels_[i]].append(rmsd_to_center[clustering.labels_[i]][sel_conf[i]])

plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')


col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]

for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1)
    new_c = [col_map[i] for i in members_rep_id[c]]
    ax.scatter(np.arange(len(rmsd_sel[c])),rmsd_sel[c],s=6.5,c=new_c,linewidths=0.5,edgecolors="black")
    plt.xlabel("CLuster members (coloured by MD replica)", fontsize=8)
    plt.ylabel("RMSD to center, nm",fontsize=8)
    plt.ylim([0.0,0.5])
    ax.set_title("Cluster "+str(c),fontsize=8,y=0.85)

fig.savefig("Rmsd2center_sel_col_by_MDrep_n"+ str(n_clusters) + ".png", format='png', dpi=600, bbox_inches='tight')


plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')


col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]

for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1)
    new_c = [col_map[i] for i in members_pept_id[c]]
    ax.scatter(np.arange(len(rmsd_sel[c])),rmsd_sel[c],s=6.5,c=new_c,linewidths=0.5,edgecolors="black")
    plt.ylim([0.0,0.5])
    ax.set_title("Cluster "+str(c),fontsize=8,y=0.85)
    plt.xlabel("CLuster members (coloured by peptide)", fontsize=8)
    plt.ylabel("RMSD to center, nm",fontsize=8)

fig.savefig("Rmsd2center_sel_col_by_pept_n"+ str(n_clusters) + ".png", format='png', dpi=600, bbox_inches='tight')

# Count the population of each cluster (the number of times it appear in the assignment array)

unique_cluster_no, counts_cluster = np.unique(clustering.labels_,return_counts=True)
population = np.array(zip(unique_cluster_no,counts_cluster))
print(population)

# Calculate %occupancy

occupancy = []
for i in range(0,len(population)):
    occupancy.append(100*float(population[i][1])/len(sel_traj))
occupancy = np.array(occupancy).reshape(n_clusters,1)
print(occupancy)

# Get rmsd values to centers of clusters for remaining frames (for each frame rmsd to all centers)

rmsd_free = [[] for x in range(len(free_conf))]
for i in range(len(free_conf)):
    for c in range(n_clusters):
        rmsd_free[i].append(rmsd_to_center[c][free_conf[i]])


assignments = np.argmin(rmsd_free,axis=1)

for i in range(len(free_conf)):
    all_labels[free_conf[i]] = assignments[i]


# Count the population of each cluster after assigning the remaining frames

frame_all = np.arange(len(traj))
unique_cluster_no, counts_cluster = np.unique(all_labels,return_counts=True)
population = np.array(zip(unique_cluster_no,counts_cluster))
print(population)
#print(zip(frame_all,all_labels))

# Calculate %occupancy

occupancy = []
for i in range(0,len(population)):
    occupancy.append(100*float(population[i][1])/len(traj))
occupancy = np.array(occupancy).reshape(n_clusters,1)
print(occupancy)

# rmsd to cluster centers for all cluster members (i.e., vs. cluster center of a cluster they belong to)
# (after assigning non-clustered frames) and get MD replica and peptide IDs for all of the members

rmsd_all = [[] for x in range(n_clusters)]
all_members_rep_id = [[] for x in range(n_clusters)]
all_members_pept_id = [[] for x in range(n_clusters)]
for i in range(len(traj)):
    c = np.int_(all_labels[i])
    rmsd_all[c].append(rmsd_to_center[c][i])
    all_members_pept_id[c].append(pept_id_merged[i])
    all_members_rep_id[c].append(rep_id_merged[i])

for i in range(n_clusters):
    print(len(rmsd_all[i]))


plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')


col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]

for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1)
    new_c = [col_map[i] for i in all_members_pept_id[c]]
    ax.scatter(np.arange(len(rmsd_all[c])),rmsd_all[c],s=0.3,c=new_c)
    plt.ylim([0.0,0.5])
    ax.set_title("Cluster "+str(c),fontsize=8,y=0.85)
    plt.xlabel("CLuster members (coloured by peptide)", fontsize=8)
    plt.ylabel("RMSD to center, nm",fontsize=8)

fig.savefig("Rmsd2center_sel_col_by_pept_n"+ str(n_clusters) + "_allframes.png", format='png', dpi=600, bbox_inches='tight')

plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')


col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]

for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1)
    new_c = [col_map[i] for i in all_members_rep_id[c]]
    ax.scatter(np.arange(len(rmsd_all[c])),rmsd_all[c],s=0.3,c=new_c)
    plt.ylim([0.0,0.5])
    ax.set_title("Cluster "+str(c),fontsize=8,y=0.85)
    plt.xlabel("CLuster members (coloured by MD replica)", fontsize=8)
    plt.ylabel("RMSD to center, nm",fontsize=8)

fig.savefig("Rmsd2center_sel_col_by_MDrep_n"+ str(n_clusters) + "_allframes.png", format='png', dpi=600, bbox_inches='tight')

all_labels = all_labels.astype(int)

# Get the mean and biggest outliers from the clusters (only sub-sample)

outliers = np.empty((n_clusters,3))

for i in (range(n_clusters)):
    outliers[i,0] = np.mean(rmsd_sel[i])
    outliers[i,1] = np.std(rmsd_sel[i])
    outliers[i,2] = rmsd_sel[i][np.argmax(rmsd_sel[i])]

print("Mean, sd and max rmsd:")
print(outliers)

# Get the mean and biggest outliers from the clusters (all frames)

print("After assigning the remaining frames")
outliers = np.empty((n_clusters,3))
for i in (range(n_clusters)):
    outliers[i,0] = np.mean(rmsd_all[i])
    outliers[i,1] = np.std(rmsd_all[i])
    outliers[i,2] = rmsd_all[i][np.argmax(rmsd_all[i])]

print()
print("Mean sd max rmsd (all frames):")
print(outliers)

# # Output
# Get all the frames belonging to a given cluster (cluster_frames)

cluster_frames = [[] for x in range(n_clusters)]
cluster_rmsd = [[] for x in range(n_clusters)]
for cl_no in range(n_clusters):
    for i in range(len(traj)):
        if all_labels[i] == cl_no:
            cluster_frames[cl_no].append(i)

# Save trajectories containing only frames belonging to a certain cluster

for i in range(0,n_clusters):
    traj.slice(cluster_frames[i]).save_dcd('Fusion_peptides_AvgLink_n'+str(n_clusters)+"cl"+str(i)+'_'+str(data_tag)+'.dcd')

# Save representative structures for each cluster

for c in range (n_clusters):
    snapshot = traj.slice(centroid[c])
    snapshot.save_pdb('Cluster'+str(c)+"_n" + str(n_clusters) +'_AvgLink_rep.pdb')

# Assign a cluster_id to each peptide in every frame

peptide_class = np.empty((0,9))
for rep in range(no_replicas):
    peptide_class_rep = np.empty((md_frames[rep],9))
    for frame in range(rep_len[rep]):
        glob_id = global_frame_id[rep][frame]
        peptide = pept_id[rep][frame]
        cluster_id = all_labels[glob_id,0]
        peptide_class_rep[md_frame_id[rep][frame],peptide] = cluster_id
    peptide_class = np.vstack([peptide_class,peptide_class_rep])


cluster_output = np.hstack((np.array(time_merged).reshape((len(time_merged),1)),peptide_class)).astype(int)
print(cluster_output)
# First column is time (rounded to integer). Remaining columns are cluster id for
# each fusion peptide
np.savetxt(traj_id + "_peptide_cluster_AvgLink_n"+str(n_clusters)+".out",cluster_output,fmt='%10.0f')

no_plots = int(9*no_replicas)
plot_columns = 9
plot_rows = no_replicas

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows), dpi=300, facecolor='w', edgecolor='k')

md_rep_range = np.empty((no_replicas,3))
for rep in range(no_replicas):
    md_rep_range[rep,0] = rep
    md_fr_shift = 0
    if rep == 0:
        md_fr_shift = 0
    else:
        for j in range(rep):
            md_fr_shift = md_fr_shift + md_frames[j]
    md_rep_range[rep,1] = md_fr_shift # start
    md_rep_range[rep,2] = md_fr_shift + md_frames[rep] -1 # stop

md_rep_range = np.int_(md_rep_range)
for i in range(no_plots):
    rep = i/9
    ax = fig.add_subplot(no_replicas, plot_columns, i+1)
    if i <=8:
        plt.title("pept"+ str(i+1),y=0.75,x=0.2)
    plt.ylim(-0.5,n_clusters+0.5)
    plt.yticks(np.arange(0,n_clusters,1))
    xs = time[rep]
    ys = cluster_output[md_rep_range[rep,1]:md_rep_range[rep,2]+1,i+1-rep*9]
    plt.scatter(xs, ys, s=2,c="black")
fig.savefig("Pept_clusterVStime" + str(n_clusters) + ".png",format='png', dpi=300, bbox_inches='tight')

# Calculate RMSDs between cluster_centers and reference structures

# Read in PDBs with cluster representative and get rmsds

ref_rmsd = np.empty((n_clusters,n_ref+1))
for c in range(0,n_clusters):
    cl = mdtraj.load(('Cluster' + str(c) + "_n" + str(n_clusters) + "_AvgLink_rep.pdb"), atom_indices=backbone)
    ref_rmsd[c,0] = occupancy[c]
    ref_rmsd[c,1] = mdtraj.rmsd(cl, ref1)
    ref_rmsd[c,2] = mdtraj.rmsd(cl, ref2)
    ref_rmsd[c,3] = mdtraj.rmsd(cl, ref3)
    ref_rmsd[c,4] = mdtraj.rmsd(cl, ref4)
print(ref_rmsd)
np.savetxt(traj_id + "_rmsd_center_to_ref_model_AvgLink_n"+str(n_clusters)+".out",ref_rmsd,fmt='%10.4f',header="Occupancy | RMSD of center to:  Weliky -- Tamm --  Bax -- straight helix")

traj_backbone = traj.atom_slice(atom_indices=backbone)

# Get rmsd to exprimental models for all frames

rmsd_to_model = np.empty((4,len(traj)))

rmsd_to_model[0]= mdtraj.rmsd(traj_backbone, ref1) # Weliky
rmsd_to_model[1]= mdtraj.rmsd(traj_backbone, ref2) # Tamm
rmsd_to_model[2]= mdtraj.rmsd(traj_backbone, ref3) # Bax
rmsd_to_model[3]= mdtraj.rmsd(traj_backbone, ref4) # straight helix

rmsd_to_model = rmsd_to_model.transpose()

# Make an array with a reference model label for every single frame (the model with lowest rmsd to this frame)

model_labels = []
for i in range(len(traj)):
    model_labels.append(np.argmin(rmsd_to_model[i]))

# Get reference model labels for all members of the clusters *before assigning all frames* (grouped by clusters)

ref_label=[[] for x in range(n_clusters)]
print(ref_label)
for c in range(n_clusters):
    for i in range(len(members[c])):
        ref_label[c].append(model_labels[sel_conf[members[c][i]]])

# Get reference model labels for all members of the clusters *after assigning all frames* (grouped by clusters)

ref_label_all=[[] for x in range(n_clusters)]
print(ref_label_all)
for c in range(n_clusters):
    for i in range(len(cluster_frames[c])):
        ref_label_all[c].append(model_labels[cluster_frames[c][i]])

# Get reference model labels for all frames selected for clustering (not grouped by clusters)

ref_label_sel=[]
for i in range(len(sel_conf)):
    ref_label_sel.append(model_labels[sel_conf[i]])

# Count what percentage of each cluster is most similar to a given reference model (only selected frames included)

ref_model_population = np.zeros((n_clusters,n_ref))
ref_model_occupancy = np.zeros((n_clusters,n_ref))
for c in range(n_clusters):
    unique_ref_no, counts = np.unique(ref_label[c],return_counts=True)
    population = np.array(zip(unique_ref_no,counts))
    for i in range(len(population)):
        ref_model_population[c][population[i][0]]=population[i][1]
        ref_model_occupancy[c][population[i][0]]=100.*population[i][1]/len(members[c])

# Count what percentage of each cluster is most similar to a given reference model (ALL frames included)

ref_model_population_all = np.zeros((n_clusters,n_ref))
ref_model_occupancy_all = np.zeros((n_clusters,n_ref))
for c in range(n_clusters):
    unique_ref_no, counts = np.unique(ref_label_all[c],return_counts=True)
    population = np.array(zip(unique_ref_no,counts))
    for i in range(len(population)):
        ref_model_population_all[c][population[i][0]]=population[i][1]
        ref_model_occupancy_all[c][population[i][0]]=100.*population[i][1]/len(cluster_frames[c])
np.savetxt(traj_id + "_ref_model_occupancy_AvgLink_n"+str(n_clusters)+".out",ref_model_occupancy_all,fmt='%10.1f',header="% of closest structures to: Weliky -- Tamm --  Bax -- straight helix")

plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')

col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
# teal: Weliky, gray: Tamm, blue: Bax, red: straight helix
for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1, projection='3d')

    x = []
    y = []
    z = []

    for i in range(len(mds_all_clust[c])):
        x.append(mds_all_clust[c][i][0])
        y.append(mds_all_clust[c][i][1])
        z.append(mds_all_clust[c][i][2])
    new_c = [col_map[i] for i in ref_label[c]]
    ax.scatter(x,y,z,c=new_c,s=5.5,linewidth=0.5,edgecolors="black")
    ax.set_title("Cluster "+str(c),fontsize=6)

fig.suptitle('MDS of pairwise RMSD, individual clusters, coloured by reference model',fontsize=7,y=0.9) # or plt.suptitle('Main title')
fig.savefig("MDS_clusters_plot_"+ str(n_clusters) + "_closest_ref_model.png" , format='png', dpi=600, bbox_inches='tight')

fig = plt.figure()
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

ax = fig.add_subplot(111, projection='3d')

x = []
y = []
z = []

col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]
# teal: Weliky, gray: Tamm, blue: Bax, red: straight helix
new_c = [col_map[i] for i in ref_label_sel]

for i in range(len(rmsd_mds.embedding_)):
    x.append(rmsd_mds.embedding_[i][0])
    y.append(rmsd_mds.embedding_[i][1])
    z.append(rmsd_mds.embedding_[i][2])

ax.scatter(x,y,z,c=new_c,s=15.5,linewidth=0.5,edgecolors="black")
ax.set_title("MDS of pairwise rmsd (selected frames), coloured by reference model" + " (n="+str(n_clusters)+")",fontsize=10)
fig.savefig("MDS_colored_by_ref_model_n"+str(n_clusters)+".png" , format='png', dpi=600, bbox_inches='tight')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = []
y = []
z = []


col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"]
#col_map = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000"]
# teal: Weliky, gray: Tamm, blue: Bax, red: straight helix
new_c = [col_map[i] for i in clustering.labels_]

for i in range(len(rmsd_mds.embedding_)):
    x.append(rmsd_mds.embedding_[i][0])
    y.append(rmsd_mds.embedding_[i][1])
    z.append(rmsd_mds.embedding_[i][2])

ax.scatter(x,y,z,c=new_c,s=15.5,linewidth=0.5,edgecolors="black")
ax.set_title("MDS of pairwise rmsd (selected frames), coloured by clusters",fontsize=10)
fig.savefig("Sel_frames_MDS_colored_by_cluster_n"+str(n_clusters)+".png" , format='png', dpi=600, bbox_inches='tight')

# Visualise mds-ed clusters and colour points by closeness to the centre of each cluster

plot_columns = 2
plot_rows = 5
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
fig = plt.figure(figsize=(4*plot_columns, 2*plot_rows),dpi=300, facecolor='w', edgecolor='k')

for c in range(0,n_clusters):
    ax = fig.add_subplot(5,2,c+1, projection='3d')

    x = []
    y = []
    z = []

    for i in range(len(mds_all_clust[c])):
        x.append(mds_all_clust[c][i][0])
        y.append(mds_all_clust[c][i][1])
        z.append(mds_all_clust[c][i][2])

    color = rmsd_sel[c]
    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(color), vmax=max(color))
    colors = [cmap(normalize(value)) for value in color]
    ax.scatter(x,y,z,color=colors,s=5.5,linewidth=0.4,edgecolors="black")
    ax.set_title("Cluster "+str(c),fontsize=6)
    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

fig.suptitle('MDS of pairwise RMSD, individual clusters, coloured by distance to cluster center',fontsize=6,y=0.9) # or plt.suptitle('Main title')
fig.savefig("Clusters_plot_"+ str(n_clusters) + "_dist2centre.png" , format='png', dpi=600, bbox_inches='tight')
