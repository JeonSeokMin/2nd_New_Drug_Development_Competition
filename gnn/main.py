import argparse
import time 
# from sklearn.metrics import mean_absolute_error
# from utils import *
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from dataset import gcnDataset, get_splitted_lipo_dataset
from solver import experiment

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

import numpy as np

def calculate_score(y_true, y_pred):
    # IC50(nM) to pIC50 변환
    def to_pIC50(IC50):
        return -np.log10(IC50 * 1e-9)
    
    y_true_pIC50 = to_pIC50(y_true)
    y_pred_pIC50 = to_pIC50(y_pred)
    
    # Normalized RMSE 계산
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    normalized_rmse = rmse / (np.max(y_true) - np.min(y_true))
    
    # Correct Ratio 계산
    absolute_errors_pIC50 = np.abs(y_true_pIC50 - y_pred_pIC50)
    correct_ratio = np.mean(absolute_errors_pIC50 <= 0.5)
    
    # 최종 점수 계산
    A = normalized_rmse
    B = correct_ratio
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    
    return score, normalized_rmse, correct_ratio

if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # ==== Embedding Config ==== #
    args.max_len = 70
    args.vocab_size = 40
    args.degree_size = 6
    args.numH_size = 5
    args.valence_size = 6
    args.isarom_size = 2
    args.emb_train = True


    # ==== Model Architecture Config ==== #
    args.in_dim = 64
    args.out_dim = 256
    args.molvec_dim = 512
    args.n_layer = 1
    args.use_bn = True
    args.act = 'relu'
    args.dp_rate = 0.3


    # ==== Optimizer Config ==== #
    args.lr = 0.00005
    args.l2_coef = 0.0001
    args.optim = 'ADAM'


    # ==== Training Config ==== #
    args.epoch = 300
    args.batch_size = 256
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.exp_name = 'exp1_lr_stage'


    # writer = Writer(prior_keyword=['n_layer', 'use_bn', 'lr', 'dp_rate', 'emb_train', 'epoch', 'batch_size'])
    # writer.clear()

    # Define Hyperparameter Search Space
    #list_n_layer = [1]
    list_lr = [0.001, 0.005]
    list_n_layer = [2,3,4,5]

    # Load Dataset
    datasets = get_splitted_lipo_dataset(ratios=[0.7, 0.3, 0], seed=seed)

    train_dataloader = DataLoader(gcnDataset(datasets[0], args.max_len), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(gcnDataset(datasets[1], args.max_len), batch_size=args.batch_size, shuffle=False)
    val_dataloader.dataset.smiles = datasets[1]["Smiles"]
    val_dataloader.dataset.exp = datasets[1]["pIC50"].values
    # test_dataloader = DataLoader(gcnDataset(datasets[2], args.max_len), batch_size=args.batch_size, shuffle=False)
    partition = {'train': train_dataloader, 'val': val_dataloader}# , 'test': test_dataloader}

    cnt_exp = 0
    for lr in list_lr:
        for n_layer in list_n_layer:
            args.lr = lr
            args.n_layer = n_layer

            model, result = experiment(partition, args)
            # print(result)
            # writer.write(result)
            
            cnt_exp += 1
            print('[Exp {:2}] got train_loss: {}, rmse: {:2.3f}, std: {:2.3f} at epoch {:2} took {:3.1f} sec'.format(cnt_exp, result.final_loss, result.best_rmse, result.best_std, result.best_epoch, result.elapsed))
            print('Hyperparameters: lr: {}, n_layer: {}'.format(lr, n_layer))

            # save model
            torch.save(model.state_dict(), 'model/exp1_lr_stage_{}.pt'.format(cnt_exp))
            val_dataset = gcnDataset(datasets[1])
            model.eval()
            X = torch.tensor(val_dataset.X).long().to(args.device).long()
            A = torch.tensor(val_dataset.A).long().to(args.device).long()
            val_y_pred = model.forward(X, A).cpu().detach().numpy().reshape(-1)
            mse = mean_squared_error(pIC50_to_IC50(val_dataset.exp), pIC50_to_IC50(val_y_pred))
            rmse = np.sqrt(mse)
            print(f'Validation RMSE: {rmse:.4f}, MSE: {mse:.4f}')
            
            final_score, normalized_rmse, correct_ratio = calculate_score(val_dataset.exp, val_y_pred)
            print(f'Final Score: {final_score:.4f}, Normalized RMSE: {normalized_rmse:.4f}, Correct Ratio: {correct_ratio:.4f}')
