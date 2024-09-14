import time
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import GCNNet

def train(model, dataloader, optimizer, criterion, args, **kwargs):
    
    epoch_train_loss = 0
    list_train_loss = list()
    cnt_iter = 0
    final_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        X, A, y = batch[0].long(), batch[1].long(), batch[2].float()
        X, A, y = X.to(args.device), A.to(args.device), y.to(args.device)
    
        model.train()
        optimizer.zero_grad()

        pred_y = model(X, A)
        
        train_loss = criterion(pred_y, y)
        epoch_train_loss += train_loss.item()
        list_train_loss.append({'epoch':batch_idx/len(dataloader)+kwargs['epoch'], 'train_loss':train_loss.item()})
        final_loss = train_loss.item()
        train_loss.backward()
        optimizer.step()
        
        cnt_iter += 1
    return model, list_train_loss, final_loss


def validate(model, dataloader, criterion, args):
    
    epoch_val_loss = 0
    cnt_iter = 0
    for batch_idx, batch in enumerate(dataloader):
        X, A, y = batch[0].long(), batch[1].long(), batch[2].float()
        X, A, y = X.to(args.device), A.to(args.device), y.to(args.device)
    
        model.eval()
        pred_y = model(X, A)
        val_loss = criterion(pred_y, y)
        epoch_val_loss += val_loss.item()
        cnt_iter += 1

    return epoch_val_loss/cnt_iter

def test(model, dataloader, args, **kwargs):

    list_y, list_pred_y = list(), list()
    for batch_idx, batch in enumerate(dataloader):
        X, A, y = batch[0].long(), batch[1].long(), batch[2].float()
        X, A, y = X.to(args.device), A.to(args.device), y.to(args.device)
    
        model.eval()
        pred_y = model(X, A)
        list_y += y.cpu().detach().numpy().tolist()
        list_pred_y += pred_y.cpu().detach().numpy().tolist()

    # mae = mean_absolute_error(list_y, list_pred_y)
    rmse = np.sqrt(mean_squared_error(list_y, list_pred_y))
    std = np.std(np.array(list_y)-np.array(list_pred_y))
    return rmse, std, list_y, list_pred_y


def experiment(partition, args):
    ts = time.time()
    
    model = GCNNet(args)    
    model.to(args.device)
    criterion = nn.MSELoss()
    
    # Initialize Optimizer
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optim == 'ADAM':
        optimizer = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.l2_coef)
    elif args.optim == 'RMSProp':
        optimizer = optim.RMSprop(trainable_parameters, lr=args.lr, weight_decay=args.l2_coef)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(trainable_parameters, lr=args.lr, weight_decay=args.l2_coef)
    else:
        assert False, "Undefined Optimizer Type"
        
    # Train, Validate, Evaluate
    list_train_loss = list()
    list_val_loss = list()
    list_mse = list()
    list_std = list()
    
    args.best_rmse = 10000
    for epoch in range(args.epoch):
        model, train_losses, final_loss = train(model, partition['train'], optimizer, criterion, args, **{'epoch':epoch})
        args.final_loss = final_loss
        val_loss = validate(model, partition['val'], criterion, args)
        # rmse, std, true_y, pred_y = test(model, partition['test'], args, **{'epoch':epoch})
        rmse, std, true_y, pred_y = test(model, partition['val'], args, **{'epoch':epoch})
        
        list_train_loss += train_losses
        list_val_loss.append({'epoch':epoch, 'val_loss':val_loss})
        list_mse.append({'epoch':epoch, 'rmse':rmse})
        list_std.append({'epoch':epoch, 'std':std})
        
        if args.best_rmse > rmse or epoch==0:
            args.best_epoch = epoch
            args.best_rmse = rmse
            args.best_std = std
            args.best_true_y = true_y
            args.best_pred_y = pred_y
            

    # End of experiments
    te = time.time()
    args.elapsed = te-ts
    args.train_losses = list_train_loss
    args.val_losses = list_val_loss
    args.rmses = list_mse
    args.stds = list_std

    return model, args