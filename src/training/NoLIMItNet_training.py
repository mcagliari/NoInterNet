import numpy as np
from matplotlib import pyplot as plt
import os

import NoInterNet_model as NIN
import NoInterNet_fraction_model as NINf
import NoLIMItNet_utils as NLIM
import argparse

import torch

from torch.utils.data import DataLoader
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(ns):
    #preparing data
    print(ns)
    print('Running on', device)

    if not os.path.exists(ns.output): 
        os.makedirs(ns.output)

    assert (len(ns.Z_names) == len(ns.Z_min)) & (len(ns.Z_names) == len(ns.Z_min)), f"I need the same number of Z_names, Z_min, and Z_max, but they are {len(ns.Z_names)}, {len(ns.Z_min)}, and {len(ns.Z_max)}"
    
    if (len(ns.Cl_int) == 1) & (len(ns.Cl_true) == 1):
        Cls, labels, lenl, max_corr, min_corr = NLIM.load_Cls(ns.input, ns.channel, ns.l_max, ns.l_min, ns.norm_Z, ns.norm_c, Z_names=ns.Z_names, Z_min=np.array(ns.Z_min), Z_max=np.array(ns.Z_max), int_names=ns.Cl_int[0], true_names=ns.Cl_true[0])
        
        #input and output dimensions
        input_size = lenl
        output_size = lenl
        n_out = ns.neurons
    else:
        if not ns.autocross:
            Cls, labels, lenl, max_corr, min_corr = NLIM.load_Cls_multiples(ns.input, ns.channel, ns.l_max, ns.l_min, ns.norm_Z, ns.norm_c, Z_names=ns.Z_names, Z_min=np.array(ns.Z_min), Z_max=np.array(ns.Z_max), int_names=ns.Cl_int, true_names=ns.Cl_true, contam_name=ns.contam_name, input_channels=np.array(ns.input_channels))
    
            #input and output dimensions
            # NOTE: as it is implemented now it works for the auto Cls from different channels, I have to see the cross Cl data to modify the routine accordingly
            input_size = lenl * len(ns.Cl_int) * len(ns.input_channels) 
            output_size = lenl * len(ns.Cl_true)
            n_out = ns.neurons * len(ns.Cl_true)
        
        else:
            add_cross, add_channel = None, None
            if ns.morecross:
                add_cross, add_channel = ns.add_cross, ns.add_channels
            Cls, labels, lenl, max_corr, min_corr = NLIM.load_Cls_auto_and_cross(ns.input, ns.channel, ns.l_max, ns.l_min, ns.norm_Z, ns.norm_c, Z_names=ns.Z_names, Z_min=np.array(ns.Z_min), Z_max=np.array(ns.Z_max), auto_names=ns.Cl_int, cross_names=ns.Cl_cross, true_names=ns.Cl_true, contam_name=ns.contam_name, input_channels=np.array(ns.input_channels), add_cross=add_cross, add_channel=add_channel)
    
            #input and output dimensions
            # NOTE: as it is implemented now it works for the auto Cls from different channels, I have to see the cross Cl data to modify the routine accordingly
            input_size = lenl * (len(ns.Cl_int) * len(ns.input_channels) + len(ns.Cl_cross)) 
            if ns.morecross: input_size += lenl * len(add_channel)
            output_size = lenl * len(ns.Cl_true)
            n_out = ns.neurons * len(ns.Cl_true)

    NZ = len(ns.Z_names)

    max_Cls = Cls.max(axis=0)
    min_Cls = Cls.min(axis=0)
    Cls = NINf.maxmin_corr(Cls, max_Cls, min_Cls)
    mm = np.vstack((min_corr, max_corr))

    norm = np.repeat(mm[np.newaxis,:,:], len(Cls), axis=0)

    #training, validation, and test
    
    n_train = int(len(Cls)*ns.train_fraction)
    n_val = int(len(Cls)*ns.val_fraction)
    
    Cls_train = Cls[:n_train]
    Cls_val   = Cls[n_train:n_val]
    
    labels_train = labels[:n_train]
    labels_val   = labels[n_train:n_val]

    norm_train = np.array(norm[:n_train, :, :])
    norm_val   = np.array(norm[n_train:n_val, :, :])

    train = NIN.PkDataset(Cls_train, labels_train, norm_train)
    val   = NIN.PkDataset(Cls_val, labels_val, norm_val)
    
    train_dataloader = DataLoader(train, batch_size=ns.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=ns.batch_size, shuffle=True)
    
    #preparing model
    if ns.moments:
        print("Moments model")
        model = NLIM.NoLIMItNet_fraction_compress_inference(input_size, ns.neurons, output_size, n_out, ns.n_min, NZ).to(device)
    else:
        raise "Better run your inference!"
        #model = NINf.NoInterNet_fraction_compress(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)
    
    if ns.logLoss:
        loss_fn = NINf.LogFractionLoss() 
    else:
        loss_fn = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=ns.learning_rate)
    
    #training
    if ns.moments:
        if ns.constrain:
            train_history, val_history = NLIM.training_constrained(train_dataloader, val_dataloader, model, optimizer, ns.epochs, ns.patience, ns.output, lenl, ns.neurons, NZ, len(ns.Cl_true), ns.lamb)
        else:
            train_history, val_history = NLIM.training_inference(train_dataloader, val_dataloader, model, optimizer, ns.epochs, ns.patience, ns.output, lenl, ns.neurons, NZ, len(ns.Cl_true))
    else:
        train_history, val_history = NIN.training(train_dataloader, val_dataloader, model, loss_fn, optimizer, ns.epochs, ns.patience, ns.output, ns.neurons)
    
    #results
    plt.plot(train_history, label="training set")
    plt.plot(val_history, label="test set")
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.savefig(ns.output+"loss_PT.pdf")
    plt.close()
    
    out_name = "params.dat"
    out_path = ns.output + out_name
    
    with open(out_path, 'w') as f:
        for k in vars(ns):
            print(k, getattr(ns, k), file=f)
        print('N_interlopers', NZ, file=f)
        print('Input_size', input_size, file=f)
        print('Output_size', output_size, file=f)
        print('N_out', n_out, file=f)
        print('Total_trainng_epochs', len(train_history), file=f)
        
    history_name = "history.dat"
    history_path = ns.output + history_name
    
    np.savetxt(history_path, np.vstack([train_history, val_history]).transpose(), header="Train Val")

    np.save(ns.output+'min_correction.npy', min_corr)
    np.save(ns.output+'max_correction.npy', max_corr)

    np.save(ns.output+'min_Cls.npy', min_Cls)
    np.save(ns.output+'max_Cls.npy', max_Cls)
    
    if ns. opt:
        return val_history
    
if __name__ == '__main__':
    desc = 'Train a dense NN that takes as inputs the interloper-contaminated Cl and outputs the Cl correction and the metallicity rescaling'
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'input to csv file containg the path to the Cl (first column contaminated, second column not contaminated, third column Z scaling)'
    group.add_argument('--input', type=str, help=h, required=True)

    h = 'output folder path'
    group.add_argument('--output', type=str, help=h, required=True)    

    #optional arguments

    h = 'SphereX channel'
    parser.add_argument('--channel', type=int, default=20, help=h)

    h = 'Z scale column names in csv'
    parser.add_argument('--Z-names', type=str, nargs='+', default='Z_scale', help=h)

    h = 'Z maximum'
    parser.add_argument('--Z-max', type=float, nargs='+', default=1.5, help=h)

    h = 'Z minimum'
    parser.add_argument('--Z-min', type=float, nargs='+', default=0.1, help=h)

    h = 'Cl contaminated names'
    parser.add_argument('--Cl-int', type=str, nargs='+', default='Cl_int', help=h)

    h = 'Cl cross names (to be used if autocross is on)'
    parser.add_argument('--Cl-cross', type=str, nargs='+', default='OII_all_cross_1', help=h)

    h = 'Cl true names'
    parser.add_argument('--Cl-true', type=str, nargs='+', default='Cl_true', help=h)

    h = 'Cl auto contam main frequency'
    parser.add_argument('--contam-name', type=str, default='Cl_int', help=h)

    h = 'input channels to use'
    parser.add_argument('--input-channels', type=int, nargs='+', default=20, help=h)

    h = 'Cl add crosses'
    parser.add_argument('--add-cross', type=str, nargs='+', default='OIII_all_cross_1', help=h)

    h = 'add channels to use in cross'  
    parser.add_argument('--add-channels', type=int, nargs='+', default=38, help=h)

    h = 'maximum l in the Cl'
    parser.add_argument('--l-max', type=float, default=5000, help=h)

    h = 'minimum l in the Cl'
    parser.add_argument('--l-min', type=float, default=0, help=h)

    h = 'fraction of Cls used for training'
    parser.add_argument('--train-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.75, help=h)
    
    h = 'fraction of Cls used for validation (must be greater than tranin-fraction) '
    parser.add_argument('--val-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.90, help=h)

    h = 'number of neurons of hidden layers, or of the first hidden layers'
    parser.add_argument('--neurons', type=int, default=60, help=h)
    
    h = 'minimum number of neurons'
    parser.add_argument('--n-min', type=int, default=8, help=h)
    
    h = 'learning rate'
    parser.add_argument('--learning-rate', type=float, default=1e-3, help=h)
    
    h = 'batch size'
    parser.add_argument('--batch-size', type=int, default=64, help=h)

    h = 'maximum number of epochs for training'
    parser.add_argument('--epochs', type=int, default=400, help=h)
    
    h = 'patience for earlystopping'
    parser.add_argument('--patience', type=int, default=20, help=h)

    h = 'contrain scale'
    parser.add_argument('--lamb', type=float, default=0.1, help=h)

    #flags

    h = 'use LogMSELoss?'
    parser.add_argument('--logLoss', action='store_true', help=h)

    h = 'hyper-parameter optimization flag'
    parser.add_argument('--opt', action='store_true', help=h)

    h = 'rescale Z in [0,1]'
    parser.add_argument('--norm-Z', action='store_true', help=h)

    h = 'rescale correction in [0,1]'
    parser.add_argument('--norm-c', action='store_true', help=h)

    h = 'moment network'
    parser.add_argument('--moments', action='store_true', help=h)

    h = ' use auto and cross'
    parser.add_argument('--autocross', action='store_true', help=h)

    h = 'add cross with other channels'
    parser.add_argument('--morecross', action='store_true', help=h)

    h = 'components constrain'
    parser.add_argument('--constrain', action='store_true', help=h)
    
    # and go!
    main(parser.parse_args())