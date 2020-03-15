import os 
import glob
import numpy as np
from data import transforms as T
import torch
from scipy.io import loadmat
from common.subsample import MaskFunc
from matplotlib import pyplot as plt 
import random 
import h5py 
from tqdm import tqdm 
from common.utils import tensor_to_complex_np

def to_numpy(x):

    real = x[:,:,0].numpy()
    img  = x[:,:,1].numpy()

    cplx = real + 1j * img

    return cplx


def cobmine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils""" 
    combined = T.complex_multiply(sensitivity[...,0], 
                                  -sensitivity[...,1], 
                                  image[...,0],  
                                  image[...,1])
    
    return combined.sum(dim = 0)


def load_traindata_path(dataset_dir, name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    validation = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
    which_view = os.path.join(dataset_dir, name)
    
    data_list =  {}
    data_list['train'] = []
    data_list['validation'] = []
    
    for k in train:
            
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
            
        for i in range(11, 30): #(1, n_slice+1):
    
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['train'] += [[raw, sen, k, i]] # rawdata, sensitivity, foldername, filename
        
    
    for k in validation:
            
        subject_id = os.path.join(which_view, str(k))
        n_slice = len(glob.glob('{0}/rawdata*.mat'.format(subject_id)))
            
        for i in range(11, 30): #(1, n_slice+1):
    
            raw = '{0}/rawdata{1}.mat'.format(subject_id, i)
            sen = '{0}/espirit{1}.mat'.format(subject_id, i)
            data_list['validation'] += [[raw, sen, k, i]]   
            
    return data_list


def data_for_training(rawdata, sensitivity, mask, norm=True):

    ''' normalize each slice using complex absolute max value'''
    
    coils, Ny, Nx, ps = rawdata.shape
   
    # shift data
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = T.ifftshift(shift_kspace, dim=(-3,-2)) * torch.from_numpy(adjust).view(1, Ny, Nx, 1).float()

    #masked_kspace = torch.where(mask == 0, torch.Tensor([0]), shift_kspace)
    mask = T.ifftshift(mask)
    mask  = mask.unsqueeze(0).unsqueeze(-1).float()
    mask  = mask.repeat(coils,1,1,ps)

    masked_kspace  = shift_kspace * mask
 
    img_gt, img_und = T.ifft2(shift_kspace), T.ifft2(masked_kspace)

    if norm:
        # perform k space raw data normalization
        # during inference there is no ground truth image so use the zero-filled recon to normalize
        norm = T.complex_abs(img_und).max()
        if norm < 1e-6: norm = 1e-6
        # normalized recon
    else: 
        norm = 1
    
    # normalize data to learn more effectively    
    img_gt, img_und = img_gt/norm, img_und/norm

    rawdata_und = masked_kspace/norm  # faster

    sense_gt = cobmine_all_coils(img_gt, sensitivity)
    
    sense_und = cobmine_all_coils(img_und, sensitivity) 

    sense_und_kspace = T.fft2(sense_und) 

    return sense_und, sense_gt, sense_und_kspace, rawdata_und, mask, sensitivity
            
name = 'coronal_pd'
dataset_dir = '/media/htic/NewVolume3/Balamurali/knee_mri_vsnet'
mode = 'validation'
data_list = load_traindata_path(dataset_dir, name)
save_dir = '/media/htic/NewVolume3/Balamurali/knee_mri_vsnet/coronal_dp_h5/{}'.format(mode)

#center_fract = 0.08
#acc   = 4
#shape = [1,640,368,2] 

#mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
#mask = mask_func(shape) ## 
#mask = T.ifftshift(mask) 

mask_path = '/media/htic/NewVolume3/Balamurali/knee_mri_vsnet/coronal_pd/masks/random4_masks_640_368.mat'
mask = loadmat(mask_path)['mask']
maskT = T.to_tensor(mask)


for rawdata_name,coil_name,folder_name,file_name in tqdm(data_list[mode]):

    rawdata = np.complex64(loadmat(rawdata_name)['rawdata']).transpose(2,0,1)
    sensitivity = np.complex64(loadmat(coil_name)['sensitivities'])

    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity.transpose(2,0,1))

    img_und,img_gt,img_und_kspace,rawdata_und,masks,sensitivity  = data_for_training(rawdata2, sensitivity2, maskT)

    #print (img_und.shape,img_gt.shape,img_und_kspace.shape,rawdata_und.shape,masks.shape,sensitivity.shape)

    img_und_np        = img_und.numpy()
    img_gt_np         = img_gt.numpy()
    img_und_kspace_np = img_und_kspace.numpy()
    rawdata_und_np    = rawdata_und.numpy()
    masks_np          = masks.numpy()
    sensitivity_np    = sensitivity.numpy()

    #print (img_und_np.shape,img_gt_np.shape,img_und_kspace_np.shape,rawdata_und_np.shape,masks_np.shape,sensitivity_np.shape)
  
    #img_und_np = tensor_to_complex_np(img_und)
    #img_gt_np  = tensor_to_complex_np(img_gt)    

    #img_und_np_abs = np.abs(img_und_np)
    #img_gt_np_abs  = np.abs(img_gt_np)

    #print (np.max(img_und_np_abs),np.min(img_und_np_abs))
    #print (np.max(img_gt_np_abs),np.min(img_gt_np_abs))

    save_path = os.path.join(save_dir,'{}_{}.h5'.format(folder_name,file_name))    

    with h5py.File(save_path,'w') as hf:
        hf['img_gt'] = img_gt_np
        hf['img_und'] = img_und_np
        hf['img_und_kspace'] = img_und_kspace_np
        hf['rawdata_und'] = rawdata_und_np
        hf['masks'] = masks_np
        hf['sensitivity'] = sensitivity_np 

    #break
