import numpy as np
import torch

from torch import nn

import NoInterNet_model as NIN
import NoInterNet_fraction_model as NIN_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Preprocessing

def load_Cls(file_name, channel=20, l_max=5000, l_min=0, norm_Z=True, norm_correction=True, n_load=None, Z_names=['Z_scale'], Z_min=np.array([0.1]), Z_max=np.array([1.5]), int_names='Cl_int', true_names='Cl_true'):
    import pandas as pd

    Cl_paths = pd.read_csv(file_name)

    Cls_contaminated = [] #input
    labels = [] #labels

    N = len(Cl_paths.index) if n_load is None else n_load

    NZ = len(Z_names)
    Zs = np.zeros(NZ)

    for i, row in Cl_paths.head(N).iterrows():
        Cl_int  = np.load(row[int_names])
        Cl_true = np.load(row[true_names])

        for j, Zn in enumerate(Z_names): Zs[j] = row[Zn]

        assert np.allclose(Cl_int[channel,0,:], Cl_true[channel,0,:]), f"{i:d}: True and contaminated Pks have different binning!"

        sell = (Cl_int[channel,0,:] > l_min) & (Cl_int[channel,0,:] < l_max)

        Cls_contaminated.append(Cl_int[channel,1,:][sell])
        
        #labels orderd as [correction(l), f]
        label = Cl_true[channel,1,:][sell] / Cl_int[channel,1,:][sell]
        
        Z_scale = minmax(Zs, Z_min, Z_max) if norm_Z else Zs
        label = np.append(label, Z_scale)
        labels.append(label)

    labels = np.array(labels)

    max_corr, min_corr = -999, -999

    if norm_correction:
        max_corr = labels[:,:-NZ].max(axis=0)
        min_corr = labels[:,:-NZ].min(axis=0)
        labels[:,:-NZ] = NIN_model.maxmin_corr(labels[:,:-NZ], max_corr, min_corr)

    return np.array(Cls_contaminated), labels, sum(sell), max_corr, min_corr

def load_Cls_multiples(file_name, channel=20, l_max=5000, l_min=0, norm_Z=True, norm_correction=True, n_load=None, Z_names=['Z_scale'], Z_min=np.array([0.1]), Z_max=np.array([1.5]), int_names=['Cl_int'], true_names=['Cl_true'], contam_name='Cl_int', input_channels=np.array([20])):
    import pandas as pd

    Cl_paths = pd.read_csv(file_name)

    Cls_contaminated = [] #input
    labels = [] #labels

    N = len(Cl_paths.index) if n_load is None else n_load

    NZ = len(Z_names)
    Zs = np.zeros(NZ)

    #l selection
    l = np.load(Cl_paths[int_names[0]].iloc[0])[channel,0,:]
    sell = (l > l_min) & (l < l_max)

    for i, row in Cl_paths.head(N).iterrows():
        Cl_int = []
        Cl_true = []

        contam = np.load(row[contam_name])[channel,1,:][sell]

        for int_name in int_names: Cl_int.append(np.load(row[int_name])[input_channels,1,:][:,sell].reshape((np.sum(sell)*len(input_channels),))) #[:,sell]
        for true_name in true_names: Cl_true.append(np.load(row[true_name])[channel,1,:][sell] / contam)
        for j, Zn in enumerate(Z_names): Zs[j] = row[Zn]

        Cls_contaminated.append(np.array([Cl_int]).flatten())
        
        #labels orderd as [correction(l), f]
        label = Cl_true
        
        Z_scale = minmax(Zs, Z_min, Z_max) if norm_Z else Zs
        label = np.append(label, Z_scale)
        labels.append(label)

    labels = np.array(labels)

    max_corr, min_corr = -999, -999

    if norm_correction:
        max_corr = labels[:,:-NZ].max(axis=0)
        min_corr = labels[:,:-NZ].min(axis=0)
        labels[:,:-NZ] = NIN_model.maxmin_corr(labels[:,:-NZ], max_corr, min_corr)

    return np.array(Cls_contaminated), labels, sum(sell), max_corr, min_corr

def load_Cls_auto_and_cross(file_name, channel=20, l_max=5000, l_min=0, norm_Z=True, norm_correction=True, n_load=None, Z_names=['Z_scale'], Z_min=np.array([0.1]), Z_max=np.array([1.5]), auto_names=['Cl_int'], cross_names=['OII_all_cross_1', 'OIII_all_cross_1'], true_names=['Cl_true'], contam_name='Cl_int', input_channels=np.array([20]), add_cross=None, add_channel=None):
    import pandas as pd

    Cl_paths = pd.read_csv(file_name)

    Cls_contaminated = [] #input
    labels = [] #labels

    N = len(Cl_paths.index) if n_load is None else n_load

    NZ = len(Z_names)
    Zs = np.zeros(NZ)

    #l selection
    l = np.load(Cl_paths[auto_names[0]].iloc[0])[channel,0,:]
    sell = (l > l_min) & (l < l_max)

    for i, row in Cl_paths.head(N).iterrows():
        Cl_int = []
        Cl_true = []

        contam = np.load(row[contam_name])[channel,1,:][sell]

        for int_name in auto_names: Cl_int.append(np.load(row[int_name])[input_channels,1,:][:,sell].reshape((np.sum(sell)*len(input_channels),))) #[:,sell]
        Cl_int = np.array([Cl_int]).flatten()
        
        for cross_name in cross_names: Cl_int = np.append(Cl_int, np.load(row[cross_name])[channel,1,:][sell])
        
        if add_channel is not None:
            for i in range(len(add_channel)): Cl_int = np.append(Cl_int, np.load(row[add_cross[i]])[add_channel[i],1,:][sell])

        for true_name in true_names: Cl_true.append(np.load(row[true_name])[channel,1,:][sell] / contam)
        for j, Zn in enumerate(Z_names): Zs[j] = row[Zn]

        Cls_contaminated.append(Cl_int.flatten())
        
        #labels orderd as [correction(l), f]
        label = Cl_true
        
        Z_scale = minmax(Zs, Z_min, Z_max) if norm_Z else Zs
        label = np.append(label, Z_scale)
        labels.append(label)

    labels = np.array(labels)

    max_corr, min_corr = -999, -999

    if norm_correction:
        max_corr = labels[:,:-NZ].max(axis=0)
        min_corr = labels[:,:-NZ].min(axis=0)
        labels[:,:-NZ] = NIN_model.maxmin_corr(labels[:,:-NZ], max_corr, min_corr)

    return np.array(Cls_contaminated), labels, sum(sell), max_corr, min_corr

#normalizations

def minmax(Z, Z_min=0.1, Z_max=1.5):
    Z -= Z_min
    Z /= (Z_max - Z_min)
    return Z

def inv_minmax(Z, Z_min=0.1, Z_max=1.5):
    Z *= (Z_max - Z_min)
    Z += Z_min
    return Z
    
#network

class NoLIMItNet_fraction_compress_inference(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8, N_Z=1):
        '''Find first and second moment of correction and metallicity for N components
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size
        self.output_size = input_size * 2 if output_size is None else output_size * 2
        self.output_size += N_Z * 2
        modules = [nn.Linear(self.input_size, n_in),
                   nn.LeakyReLU()] #input
         
        self.n_hidden = 0 #number of hidden layers
        n_check = n_in
        n_out = n_in if n_out is None else n_out
        while n_check > n_min:
            self.n_hidden += 1
            n_out_hid = int(n_check / 2)
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L compress
            n_check = n_out_hid
        while n_check < n_out:
            self.n_hidden += 1
            n_out_hid  = n_check * 2
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L decompress
            n_check = n_out_hid
            
        modules.append(nn.Linear(n_check,self.output_size)) #output
        
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        #x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out

class NoLIMItNet_patchwork(nn.Module):
    def __init__(self, pretrained_model, output_2, freeze_trunk=True, warm_head_1=True, head_2_hidden=None):
        '''I attach to the pretrained model nome new stuffs'''
        super().__init__()

        # The trunk is all layers EXCEPT the last one (original output layer for pretrained model)
        self.trunk = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False
        
        # Determine trunk_output_size (out_features of the second to last layer)
        trunk_output_size = list(pretrained_model.children())[-2].out_features # this works if -2 is a Linear layer

        # Head for 1 is the original last layer
        self.head_a = list(pretrained_model.children())[-1]
        if freeze_trunk:
             for param in self.head_a.parameters(): 
                 param.requires_grad = warm_head_1 

        # Create new Head for 2
        head_b_layers = []
        prev_b_size = trunk_output_size
        if head_2_hidden:
            for hidden_size in head_2_hidden:
                head_b_layers.append(nn.Linear(prev_b_size, hidden_size))
                head_b_layers.append(nn.LeakyReLU())
                prev_b_size = hidden_size
        head_b_layers.append(nn.Linear(prev_b_size, output_2*2)) #inference 
        self.head_b = nn.Sequential(*head_b_layers)

    def forward(self, x):
        shared_features = self.trunk(x)
        out_a = self.head_a(shared_features)
        out_b = self.head_b(shared_features)
        return out_a, out_b

#Training
def train_loop_inference(dataloader, model, optimizer, lenk, N_Z, N_out):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0
    lenNk = lenk * N_out

    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        corr_pred = pred[:,:lenNk]
        corr_labl = y[:,:lenNk]
        f_pred = pred[:,lenNk:lenNk+N_Z]
        f_labl = y[:,-N_Z:]
        corr_std = pred[:,lenNk+N_Z:lenNk*2+N_Z]
        f_std = pred[:,-N_Z:]

        loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
        loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)

        loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
        loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)

        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches
            
def test_loop_inference(dataloader, model, lenk, N_Z, N_out):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    lenNk = lenk * N_out
    
    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            corr_pred = pred[:,:lenNk]
            corr_labl = y[:,:lenNk]
            f_pred = pred[:,lenNk:lenNk+N_Z]
            f_labl = y[:,-N_Z:]
            corr_std = pred[:,lenNk+N_Z:lenNk*2+N_Z]
            f_std = pred[:,-N_Z:]
    
            loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
            loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)
    
            loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
            loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)
    
            loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary)

            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training_inference(train_dataloader, val_dataloader, model, optimizer, epochs, patience, output, lenk, neurons=60, N_Z=1, N_out=1):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop_inference(train_dataloader, model, optimizer, lenk, N_Z, N_out)
        val_loss = test_loop_inference(val_dataloader, model, lenk, N_Z, N_out)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            NIN.checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history

def train_loop_patchwork(dataloader, model, optimizer, lenk, N_Z, warm_head_1, N_out_1, N_out_2):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0

    lenN1k = lenk * N_out_1
    lenN2k = lenk * N_out_2

    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        out_a, out_b = model(X)

        corr_pred = out_b[:,:lenN2k]
        corr_labl = y[:,:lenN2k]
        corr_std = out_b[:,lenN2k:lenN2k*2]
    
        loss_correction_primary_b = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
        loss_correction_secondary_b = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)

        loss = torch.log(loss_correction_primary_b) + torch.log(loss_correction_secondary_b) 

        if warm_head_1:
            corr_pred = out_a[:,:lenN1k]
            corr_labl = y[:,:lenN1k]
            f_pred = out_a[:,lenN1k:lenN1k+N_Z]
            f_labl = y[:,-N_Z:]
            corr_std = out_a[:,lenN1k+N_Z:lenN1k*2+N_Z]
            f_std = out_a[:,-N_Z:]
    
            loss_correction_primary_a = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
            loss_f_primary_a = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)
    
            loss_correction_secondary_a = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
            loss_f_secondary_a = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)

            loss += torch.log(loss_correction_primary_a) + torch.log(loss_correction_secondary_a) + torch.log(loss_f_primary_a) + torch.log(loss_f_secondary_a)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches
            
def test_loop_patchwork(dataloader, model, lenk, N_Z, warm_head_1, N_out_1, N_out_2):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    lenN1k = lenk * N_out_1
    lenN2k = lenk * N_out_2

    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            out_a, out_b = model(X)

            corr_pred = out_b[:,:lenN2k]
            corr_labl = y[:,:lenN2k]
            corr_std = out_b[:,lenN2k:lenN2k*2]
        
            loss_correction_primary_b = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
            loss_correction_secondary_b = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
    
            loss = torch.log(loss_correction_primary_b) + torch.log(loss_correction_secondary_b) 
    
            if warm_head_1:
                corr_pred = out_a[:,:lenN1k]
                corr_labl = y[:,:lenN1k]
                f_pred = out_a[:,lenN1k:lenN1k+N_Z]
                f_labl = y[:,-N_Z:]
                corr_std = out_a[:,lenN1k+N_Z:lenN1k*2+N_Z]
                f_std = out_a[:,-N_Z:]
        
                loss_correction_primary_a = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
                loss_f_primary_a = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)
        
                loss_correction_secondary_a = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
                loss_f_secondary_a = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)
    
                loss += torch.log(loss_correction_primary_a) + torch.log(loss_correction_secondary_a) + torch.log(loss_f_primary_a) + torch.log(loss_f_secondary_a)
        
            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training_patchwork(train_dataloader, val_dataloader, model, optimizer, epochs, patience, output, lenk, neurons=60, N_Z=1, warm_head_1=True, N_out_1=1, N_out_2=1):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop_patchwork(train_dataloader, model, optimizer, lenk, N_Z, warm_head_1, N_out_1, N_out_2)
        val_loss = test_loop_patchwork(val_dataloader, model, lenk, N_Z, warm_head_1, N_out_1, N_out_2)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            NIN.checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history

def inv_maxmin_corr(corr, max, min):
    correct = corr * (max - min)
    correct = correct + min
    return correct

def train_loop_constrained(dataloader, model, optimizer, lenk, N_Z, N_out, lambda_):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0
    lenNk = lenk * N_out

    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        n = n.to(device)
        pred = model(X)

        corr_pred = pred[:,:lenNk]
        corr_labl = y[:,:lenNk]
        f_pred = pred[:,lenNk:lenNk+N_Z]
        f_labl = y[:,-N_Z:]
        corr_std = pred[:,lenNk+N_Z:lenNk*2+N_Z]
        f_std = pred[:,-N_Z:]

        loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
        loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)

        loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
        loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)

        corr_pred_norm = inv_maxmin_corr(corr_pred, n[0,1,:], n[0,0,:])
        corr_pred_norm = corr_pred_norm.reshape((corr_pred_norm.shape[0], N_out, lenk))

        sum_pred = corr_pred_norm.sum(axis=1)
        residual = sum_pred - 1.

        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary) + lambda_ * torch.log(torch.mean(residual**2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches
            
def test_loop_constrained(dataloader, model, lenk, N_Z, N_out, lambda_):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    lenNk = lenk * N_out
    
    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            n = n.to(device)
            pred = model(X)

            corr_pred = pred[:,:lenNk]
            corr_labl = y[:,:lenNk]
            f_pred = pred[:,lenNk:lenNk+N_Z]
            f_labl = y[:,-N_Z:]
            corr_std = pred[:,lenNk+N_Z:lenNk*2+N_Z]
            f_std = pred[:,-N_Z:]
    
            loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
            loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum((f_pred - f_labl)**2, axis=1), axis=0)
    
            loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
            loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0) if N_Z == 1 else torch.mean(torch.sum(((f_pred - f_labl)**2 - f_std**2)**2, axis=1), axis=0)
    
            corr_pred_norm = inv_maxmin_corr(corr_pred, n[0,1,:], n[0,0,:])
            corr_pred_norm = corr_pred_norm.reshape((corr_pred_norm.shape[0], N_out, lenk))
    
            sum_pred = corr_pred_norm.sum(axis=1)
            residual = sum_pred - 1.

            loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary) + lambda_ * torch.log(torch.mean(residual**2))
    

            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training_constrained(train_dataloader, val_dataloader, model, optimizer, epochs, patience, output, lenk, neurons=60, N_Z=1, N_out=1, lambda_=0.1):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop_constrained(train_dataloader, model, optimizer, lenk, N_Z, N_out, lambda_)
        val_loss = test_loop_constrained(val_dataloader, model, lenk, N_Z, N_out, lambda_)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            NIN.checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history