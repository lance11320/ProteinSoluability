import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import r2_score,f1_score,roc_auc_score
import sklearn.metrics as metrics
import os,sys
import os.path as osp
import pandas as pd
from transformers import AutoTokenizer
import argparse
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import ESMclassficationModel
# from models_norm import ESMclassficationModel
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
if not osp.exists('./Ckpts'):
    os.makedirs('./Ckpts')
parser = argparse.ArgumentParser(description='Protein_Solution')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size')
parser.add_argument('--max_sql', type=int, default=512, 
                    help='sequence length')
parser.add_argument('--seed', type=int, default=42,
                    help='set random seed')
parser.add_argument('--lr', type=int, default= 1e-3, help='Learning Rate')
parser.add_argument('--ckpt',type=str ,default='./Ckpts/best_model.pt')
args = parser.parse_args()
torch.manual_seed(args.seed)
datapath = './Datasets_Protein/eSol'
train_path = os.path.join(datapath,'eSol_train.csv')
test_path = os.path.join(datapath,'eSol_test.csv')
train_lab_path = os.path.join(datapath,'eSol_train.csv')
test_lab_path = os.path.join(datapath,'eSol_test.csv')

def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    r2 = metrics.r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auc = metrics.roc_auc_score(binary_true, y_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'r2': r2,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result

def encoder(max_len,text_list):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    Tokens = tokenizer(
    text_list,
    padding = True,
    truncation = True,
    max_length = max_len,
    return_tensors='pt' 
    )
    input_ids = Tokens['input_ids']
    attention_mask = Tokens['attention_mask']
    return input_ids, attention_mask
def load_data(path):
    csvFileObj = open(path,encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    for row in readerObj:
        if readerObj.line_num == 1:
            continue
        text = row[2]
        lab = float(row[1])
        labels.append(lab)
        text_list.append(text)

    input_ids, attention_mask = encoder(max_len = args.max_sql, text_list=text_list)
    labels = torch.tensor(labels)
    data = TensorDataset(input_ids,attention_mask,labels)
    return data

print('---Start Loading Data, It Takes a Lot of Time, Please Wait---')
train_data = load_data(train_path)
test_data = load_data(test_path)
train_set,valid_set = random_split(dataset=train_data,lengths=[0.9,0.1])
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2*args.batch_size, shuffle=False)

# Use gpu or cpu to train
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

if use_gpu:
    torch.cuda.set_device(0)
    device = torch.device(0)
    print('You will use a GPU')
else:
    device = torch.device("cpu")

print('---Seting the Model Now---')
model = ESMclassficationModel()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)

def evaluate():
    model.to(device)
    model.eval()
    with torch.no_grad():
        r2s = []
        lab_all = []
        output_all = []
        for step, (input_ids,attention_mask,labels) in enumerate(valid_loader):              
            input_ids,attention_mask,labels=input_ids.to(device),attention_mask.to(device),labels.to(device)
            out_put = model(input_ids,attention_mask)
            rsquare = r2_score(labels.cpu().numpy(),out_put.cpu().numpy())
            r2s.append(rsquare)
            lab_all.extend(labels.cpu().numpy())
            output_all.extend(out_put.cpu().numpy())

        r2s = r2_score(lab_all,output_all)
        lossall = criterion(torch.Tensor(output_all),torch.Tensor(lab_all))
        return lossall,r2s

def train() :
    model.to(device)
    t_total = len(train_loader)
    total_epochs = args.epochs
    bestr2s = 0
    print('--- All Set! Happy Training! ---')
    for epoch in range(total_epochs): 
        r2s = []
        for step, (input_ids,attention_mask,labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            out_put = model(input_ids,attention_mask)
            out = out_put.detach().cpu().numpy()
            rsquare = r2_score(labels.cpu().numpy(),out)
            r2s.append(rsquare)
            loss = criterion(out_put, labels)
            loss.backward()
            optimizer.step()
            if (step + 1) % 50 == 0:
                train_acc = np.mean(r2s)
                print("Train Epoch[{}/{}],step[{}/{}],r2s{:.3f},loss:{:.3f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc,loss.item()))
            if (step + 1) % 100 == 0:
                train_acc = np.mean(r2s)
                losseval, r2eval = evaluate()
                if bestr2s < r2eval:
                    bestr2s = r2eval
                    path = args.ckpt
                    torch.save(model.state_dict(), path)
                print("Valid Epoch[{}/{}],step[{}/{}],valid_r2s{:.3f},loss:{:.3f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),r2eval,losseval.item()))
    
        scheduler.step(r2eval)  
def test():
    best_model = ESMclassficationModel()
    best_model.load_state_dict(torch.load(args.ckpt))
    best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        r2s = []
        labels_plt = []
        output_plt = []
        for step, (input_ids,attention_mask,labels) in enumerate(test_loader):              
            input_ids,attention_mask,labels=input_ids.to(device),attention_mask.to(device),labels.to(device)
            out_put = best_model(input_ids.to(device),attention_mask.to(device))
            loss = criterion(out_put, labels)
            rsquare = r2_score(labels.cpu().numpy(),out_put.cpu().numpy())
            r2s.append(rsquare)
            labels_plt.extend(labels.cpu().numpy()) 
            output_plt.extend(out_put.cpu().numpy()) 
        plt.scatter(labels_plt,output_plt)
        plt.show()
        print('Overall R2 is',r2_score(labels_plt,output_plt))
        results = analysis(labels_plt,output_plt)
        print(results)

        return r2_score(labels_plt,output_plt), metrics.mean_squared_error(labels_plt,output_plt)

if __name__ == '__main__':
    testres,loss = test()
    print('Your Test Result is',testres)
    print('RMSE is',np.sqrt(loss))