
from os.path import *
import numpy as np
import random
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import butter, sosfilt
from .unet import UNet1d
from monai.inferers import SlidingWindowInferer

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # sos = butter(order, [low, high], analog=False,
    #              btype='band', output='sos')
    sos = butter(order, low, analog=False,
                 btype='high', output='sos')
    y = sosfilt(sos, data)
    return y


def norm(ecg):
    min1, max1 = np.percentile(ecg, [1, 99])
    ecg[ecg > max1] = max1
    ecg[ecg < min1] = min1
    ecg = (ecg - min1)/(max1-min1)
    return ecg

def mad(data):
    median = np.median(data)
    deviation = np.abs(data - median)
    return np.median(deviation)


def run(input_eeg,
        input_ecg=None,
        sfreq=500,
        iter_num=5000,
        winsize_sec=2,
        lr=5e-4,
        lr_sche = 'step',
        overlapped = None,
        pretrain = False, 
        pretrain_path = None, 
        weight_path = None,
        kernels = [3],
        mad_param = 5,
        up_weight = None,
        d = 'cuda',
        show = False,
        section = [1000,2000],
        ens_time = 1
        ):
    
    window = winsize_sec * sfreq
    eeg_raw = input_eeg
    eeg_channel = eeg_raw.shape[0]
    
    eeg_filtered = eeg_raw * 0
    t = time.time()
    for ii in range(eeg_channel):
        eeg_filtered[ii, ...] = butter_bandpass_filter(
            eeg_raw[ii, :], 0.25, sfreq*0.4, sfreq)
        # eeg_filtered[ii, ...] = butter_bandpass_filter(
        #     eeg_raw[ii, :], 0.5, 40, sfreq)

    baseline = eeg_raw - eeg_filtered
       

    if input_ecg is None:
        from sklearn.decomposition import PCA

        ecg = PCA(n_components=1)
            # ecg = norm(pca.fit_transform(eeg_filtered.T)[:, 0].flatten())
    else:
        ecg = input_ecg.flatten()
            # ecg = norm(input_ecg.flatten())


    torch.cuda.empty_cache()
    if d == 'cpu':
        device = torch.device('cpu')
    elif d == 'cuda' and torch.cuda.is_available():
        device = ('cuda')
    else:
        print('cuda is not available')
        device = torch.device('cpu')


    BCG_pred_sum = None
    for ens in range(ens_time):
        
        NET = UNet1d(n_channels=1, n_classes=eeg_channel, nfilter=8, kernel_list = kernels, up_weight = up_weight).to(device)
        
        if pretrain:
            print('used pretrained model')
            NET.load_state_dict(torch.load(pretrain_path, map_location=device))
            
        optimizer = torch.optim.Adam(NET.parameters(), lr=lr)
        optimizer.zero_grad()
        maxlen = ecg.size
        
        if lr_sche == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps = iter_num)
        elif lr_sche == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 2000, eta_min = lr * 0.001, last_epoch=-1)
        elif lr_sche == 'step':
            #constant learning rate
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        elif lr_sche == 'constant':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    
        loss_list = []
    
        index_all = (np.arange(0, maxlen-window, (maxlen-window)/iter_num )).astype(int)
        np.random.shuffle(index_all)
        pbar = tqdm.tqdm(index_all[:iter_num])
        
        
        if show == True:
            EEG = eeg_filtered[:, :]
            ECG = ecg[:]
            
            plt.figure(figsize=(14, 2), dpi=1200)
            plt.plot(ECG[section[0]:section[1]], 'darkslateblue', linewidth=2)
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right')
            plt.yticks(fontsize=15)
            plt.xticks([])
            plt.savefig(f"jpg/BRNet_paper/processing_ECG.jpg", bbox_inches='tight', dpi = 300)
            plt.show()
            plt.close()
            
            plt.figure(figsize=(14, 2), dpi=1200)
            plt.plot(EEG[30, section[0]:section[1]], 'darkslateblue', linewidth=2)
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right')
            plt.yticks(fontsize=15)
            plt.xticks([])
            plt.savefig(f"jpg/BRNet_paper/processing_EEG.jpg", bbox_inches='tight', dpi = 300)
            plt.show()
            plt.close()

        
        
        count = 0
        for index in pbar:      ##pbar
            count += 1
            ECG = ecg[index:(index + window)]
            EEG = eeg_filtered[:, index:(index + window)]
            if np.any(ECG > mad(ECG) * mad_param):
                continue
            ECG = norm(ECG)
            ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
            EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()
    
            # step 3: forward path of UNET
            logits = NET(ECG_d)
            loss = nn.MSELoss()(logits, EEG_d)
            loss_list.append(loss.item())
            
    
    
            # Step 5: Perform back-propagation
            loss.backward(retain_graph=True) #accumulate the gradients
            optimizer.step() #Update network weights according to the optimizer
            optimizer.zero_grad() #empty the gradients
            scheduler.step()
            
            if count % 50 == 0:
                pbar.set_description(f"Loss {np.mean(loss_list):.3f}, lr: {optimizer.param_groups[0]['lr']:.5f}")
                loss_list = [] 
            if show == True:
                if count == iter_num / 2:
                    loss_list = []
                    ECG = ecg[:]
                    EEG = eeg_filtered[:, :]
                    ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
                    EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()
                    logits = NET(ECG_d)
                    EEG_pred = logits.cpu().detach().numpy()
    
                    plt.figure(figsize=(14, 2), dpi=1200)
    
                    plt.plot(EEG[30, section[0]:section[1]], 'darkslateblue', linewidth=2)
                    plt.plot(EEG[30, section[0]:section[1]] - EEG_pred[0, 30, section[0]:section[1]], 'red', linewidth=2)
                    plt.gca().yaxis.label.set(rotation='horizontal', ha='right')
                    plt.yticks(fontsize=15)
                    plt.xticks([])
                    plt.savefig(f"jpg/BRNet_paper/processing_{count}.jpg", bbox_inches='tight', dpi = 300)
                    plt.show()
                    plt.close()
                    
                
                # plt.title(f' {time1} seconds')
        
        if show == True:
            loss_list = []
            ECG = ecg[:]
            EEG = eeg_filtered[:, :]
            ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
            EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()
            logits = NET(ECG_d)        
            EEG_pred = logits.cpu().detach().numpy()
            plt.figure(figsize=(14, 2), dpi=300)
    
            plt.plot(EEG[30, section[0]:section[1]], 'darkslateblue', linewidth=2)
            plt.plot(EEG[30, section[0]:section[1]] - EEG_pred[0, 30, section[0]:section[1]], 'red', linewidth=2)
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right');
            plt.yticks(fontsize=15)
            plt.xlabel('Time(s)', fontsize = 15)
            plt.xticks(np.linspace(0, section[1]-section[0], 5), labels=np.linspace(0, (section[1]-section[0])/500, 5, dtype = int), fontsize = 15)
            plt.savefig(f"jpg/BRNet_paper/processing_{count}.jpg", bbox_inches='tight', dpi = 300)
            plt.show()
            plt.close()
        
        if overlapped is not None:
            inferer = SlidingWindowInferer(roi_size=(window), sw_batch_size=1, overlap=overlapped, mode="gaussian")
            
        EEG = eeg_filtered
        #ECG = norm(butter_bandpass_filter(data['ECG'], 0.5, 20, sfreq))
        ECG = ecg
        ECG_d = torch.from_numpy(ECG[None, ...][None, ...]).to(device).float()
        EEG_d = torch.from_numpy(EEG[None, ...]).to(device).float()
        
        
        with torch.no_grad():
            if overlapped is not None:
                logits = inferer(ECG_d, NET)
            else:
                logits = NET(ECG_d)
    
        BCG_pred = logits.cpu().detach().numpy()[0, ...]
        if BCG_pred_sum is None:
            BCG_pred_sum = np.copy(BCG_pred)
        else:
            BCG_pred_sum += BCG_pred
            
    neweeg = EEG - BCG_pred_sum / ens_time #+ baseline

    return neweeg
