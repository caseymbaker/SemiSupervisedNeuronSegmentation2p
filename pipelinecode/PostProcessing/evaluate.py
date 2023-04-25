import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
import time


def GetPerformance_Jaccard_2(GTMasks, Masks, ThreshJ=0.5):
    '''Calculate the recall, precision, and F1 score of segmented neurons by comparing with ground truth.

    Inputs: 
        GTMasks (sparse.csr_matrix): Ground truth masks.
        Masks (sparse.csr_matrix): Segmented masks.
        ThreshJ (float, default to 0.5): Threshold Jaccard distance for two neurons to match.

    Outputs:
        Recall (float): Percentage of matched neurons over all GT neurons. 
        Precision (float): Percentage of matched neurons over all segmented neurons. 
        F1 (float): Harmonic mean of Recall and Precision. 
    '''
    if 'bool' in str(Masks.dtype): # bool cannot be used to calculate IoU
        Masks = Masks.astype('uint32')
    if 'bool' in str(GTMasks.dtype):
        GTMasks = GTMasks.astype('uint32')
    NGT = GTMasks.shape[0] # Number of GT neurons
    NMask = Masks.shape[0] # Number of segmented neurons
    a1 = np.repeat(GTMasks.sum(axis=1).A, NMask, axis=1)
    a2 = np.repeat(Masks.sum(axis=1).A.T, NGT, axis=0)
    intersectMat = GTMasks.dot(Masks.transpose()).A
    unionMat = a1 + a2 - intersectMat
    JaccardInd = intersectMat/unionMat # IoU between each pair of neurons
    Dmat = 1-JaccardInd # Jaccard distance is 1 - IoU
    # Dmat[intersectMat == a1] = 0
    # Dmat[intersectMat == a2] = 0
    D = Dmat
    # When Jaccard distance is larger than ThreshJ, it is set to 2, meaning infinity
    D[D > ThreshJ] = 2 
    # Use Hungarian algorithm to match two sets of neurons
    row_ind2, col_ind2 = linear_sum_assignment(D) 
    num_match = (D[row_ind2, col_ind2]<1).sum() # Number of matched neurons
    if num_match == 0:
        Recall = Precision = F1 = 0
    else:
        Recall = num_match/NGT
        Precision = num_match/NMask
        F1 = 2*Recall*Precision/(Recall+Precision)
    return Recall, Precision, F1

def GetOverlappingNeurons(GTMasks, Masks, ThreshJ=0.5):
    '''Calculate the recall, precision, and F1 score of segmented neurons by comparing with ground truth.

    Inputs: 
        GTMasks (sparse.csr_matrix): Ground truth masks.
        Masks (sparse.csr_matrix): Segmented masks.
        ThreshJ (float, default to 0.5): Threshold Jaccard distance for two neurons to match.

    Outputs:
        Recall (float): Percentage of matched neurons over all GT neurons. 
        Precision (float): Percentage of matched neurons over all segmented neurons. 
        F1 (float): Harmonic mean of Recall and Precision. 
    '''
    if 'bool' in str(Masks.dtype): # bool cannot be used to calculate IoU
        Masks = Masks.astype('uint32')
    if 'bool' in str(GTMasks.dtype):
        GTMasks = GTMasks.astype('uint32')
    NGT = GTMasks.shape[0] # Number of GT neurons
    NMask = Masks.shape[0] # Number of segmented neurons
    a1 = np.repeat(GTMasks.sum(axis=1).A, NMask, axis=1)
    a2 = np.repeat(Masks.sum(axis=1).A.T, NGT, axis=0)
    intersectMat = GTMasks.dot(Masks.transpose()).A
    unionMat = a1 + a2 - intersectMat
    JaccardInd = intersectMat/unionMat # IoU between each pair of neurons
    Dmat = 1-JaccardInd # Jaccard distance is 1 - IoU
    # Dmat[intersectMat == a1] = 0
    # Dmat[intersectMat == a2] = 0
    D = Dmat
    
    # When Jaccard distance is larger than ThreshJ, it is set to 2, meaning infinity
    D[D > ThreshJ] = 2 
    # Use Hungarian algorithm to match two sets of neurons
    row_ind2, col_ind2 = linear_sum_assignment(D) 
    D2 = np.where(D[row_ind2, col_ind2]<1) # indicies of matched neurons
    D3 = np.where(D[row_ind2, col_ind2]>=1) # indicies of unmatched neurons

    return D2, D3, np.sum(JaccardInd)




import h5py
import fissa
import os
from scipy.io import loadmat


def get_all_traces(network_input:np.array, rois, dir_save:str, Exp_ID:str):
    '''get traces from predicted masks to input into 1D CNN for final classification". 

    Inputs: 
        network_input (3D numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing.
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
        In addition, the raw and unmixed traces before and after FISSA are saved in the same folder
            but a different sub-folder in another "(Exp_ID).h5" file. The ".h5" file has two datasets, 
            "raw_traces" and "unmixed_traces" saving the traces before and after FISSA, respectively. 
    '''

    (nframesf, rowspad, colspad) = network_input.shape
    (ncells, rows, cols) = rois.shape
    # The lateral shape of "network_input" can be larger than that of "rois" due to padding in pre-processing
    # This step crop "network_input" to match the shape of "rois"
    network_input = network_input[:, :rows, :cols]

    # Use FISSA to calculate the decontaminated traces of neural activities. 
    folder_FISSA = os.path.join(dir_save, 'FISSA')    
    start = time.time()
    experiment = fissa.Experiment([network_input], [rois.tolist()], folder_FISSA, ncores_preparation=1)
    experiment.separation_prep(redo=True)
    prep = time.time()
    
    experiment.separate(redo_prep=False, redo_sep=True)
    finish = time.time()
    experiment.save_to_matlab()
    del network_input
    print('FISSA time: {} s'.format(finish-start))
    print('    Preparation time: {} s'.format(prep-start))
    print('    Separation time: {} s'.format(finish-prep))

    # %% Extract raw and unmixed traces from the output of FISSA
    raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
    unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
    del experiment

    # Save the raw and unmixed traces into a ".h5" file under folder "dir_trace".
    dir_trace = os.path.join(dir_save, "traces")
    if not os.path.exists(dir_trace):
        os.makedirs(dir_trace)        
    f = h5py.File(os.path.join(dir_trace, Exp_ID+"_predicted.h5"), "w")
    f.create_dataset("raw_traces", data = raw_traces)
    f.create_dataset("unmixed_traces", data = unmixed_traces)
    f.close()