import os,sys
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
os.chdir(sys.path[0])
# path
Dataset_Path = './Data/'
Model_Path = './ESMModel/'
Result_Path = './ESMResult/'

amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 5
LEARNING_RATE = 5E-5
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 1
NUM_CLASSES = 1

# GCN parameters
GCN_FEATURE_DIM = 1371
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})


def load_features(sequence_name, sequence, mean, std, blosum):

    feature_matrix = np.load(Dataset_Path + 'node_feature/' + sequence_name + '.npy')

    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load(Dataset_Path + 'edge_features/' + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


def load_values():
    # (23,)
    blosum_mean = np.load(Dataset_Path + 'eSol_blosum_mean.npy')
    blosum_std = np.load(Dataset_Path + 'eSol_blosum_std.npy')

    # (71,)
    oneD_mean = np.load(Dataset_Path + 'eSol_oneD_mean.npy')
    oneD_std = np.load(Dataset_Path + 'eSol_oneD_std.npy')

    mean = np.concatenate([blosum_mean, oneD_mean])
    std = np.concatenate([blosum_std, oneD_std])

    return mean, std


def load_blosum():
    with open(Dataset_Path + 'BLOSUM62_dim23.txt', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result


class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['gene'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['solubility'].values
        self.mean, self.std = load_values()
        self.blosum = load_blosum()

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        # L * 91
        sequence_feature = load_features(sequence_name, sequence, self.mean, self.std, self.blosum)
        # L * L
        sequence_graph = load_graph(sequence_name)
        return sequence_name, sequence, label, sequence_feature, sequence_graph

    def __len__(self):
        return len(self.labels)





class Attention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-1, -2))

        attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, 1e-9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        return context, attention


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        hid_dim = 1371
        self.rnn = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, num_layers=1, batch_first=True)
        self.attn = Attention()
        self.dense = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self,hidden_state):
        hidden_state = torch.unsqueeze(hidden_state,0)
        out, gru_hidden = self.rnn(hidden_state)
        hidden_state, atten1 = self.attn(hidden_state,out,hidden_state,scale=1)
        self.attn_weight = atten1
        linear_output = self.dense(hidden_state)
        linear_output = torch.squeeze(torch.mean(linear_output,dim=1),1)
        return  linear_output


def train_one_epoch(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        _, _, labels, sequence_features, sequence_graphs = data

        sequence_features = torch.squeeze(sequence_features)
        sequence_graphs = torch.squeeze(sequence_graphs)

        if torch.cuda.is_available():
            features = Variable(sequence_features.cuda())
            graphs = Variable(sequence_graphs.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_features)
            graphs = Variable(sequence_graphs)
            y_true = Variable(labels)

        features = features.to(torch.float32)
        y_pred = model(features)
        y_true = y_true.float()

        # calculate loss
        loss = model.criterion(y_pred, y_true)

        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, sequence_features, sequence_graphs = data

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)
                y_true = Variable(labels)

            features = features.to(torch.float32)
            y_pred = model(features)
            y_true = y_true.float()

            loss = model.criterion(y_pred, y_true)
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, valid_true, valid_pred, valid_name


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    train_losses = []
    train_pearson = []
    train_r2 = []
    train_binary_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_auc = []
    train_mcc = []
    train_sensitivity = []
    train_specificity = []

    valid_losses = []
    valid_pearson = []
    valid_r2 = []
    valid_binary_acc = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []
    valid_auc = []
    valid_mcc = []
    valid_sensitivity = []
    valid_specificity = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss: ", np.sqrt(epoch_loss_train_avg))
        print("Train pearson:", result_train['pearson'])
        print("Train r2:", result_train['r2'])
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train precision: ", result_train['precision'])
        print("Train recall: ", result_train['recall'])
        print("Train F1: ", result_train['f1'])
        print("Train auc: ", result_train['auc'])
        print("Train mcc: ", result_train['mcc'])
        print("Train sensitivity: ", result_train['sensitivity'])
        print("Train specificity: ", result_train['specificity'])

        train_losses.append(np.sqrt(epoch_loss_train_avg))
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        train_binary_acc.append(result_train['binary_acc'])
        train_precision.append(result_train['precision'])
        train_recall.append(result_train['recall'])
        train_f1.append(result_train['f1'])
        train_auc.append(result_train['auc'])
        train_mcc.append(result_train['mcc'])
        train_sensitivity.append(result_train['sensitivity'])
        train_specificity.append(result_train['specificity'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid auc: ", result_valid['auc'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Valid sensitivity: ", result_valid['sensitivity'])
        print("Valid specificity: ", result_valid['specificity'])

        valid_losses.append(np.sqrt(epoch_loss_valid_avg))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])
        valid_binary_acc.append(result_valid['binary_acc'])
        valid_precision.append(result_valid['precision'])
        valid_recall.append(result_valid['recall'])
        valid_f1.append(result_valid['f1'])
        valid_auc.append(result_valid['auc'])
        valid_mcc.append(result_valid['mcc'])
        valid_sensitivity.append(result_valid['sensitivity'])
        valid_specificity.append(result_valid['specificity'])

        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_detail_dataframe = pd.DataFrame({'gene': valid_name, 'solubility': valid_true, 'prediction': valid_pred})
            valid_detail_dataframe.sort_values(by=['gene'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')

    # save calculation information
    result_all = {
        'Train_loss': train_losses,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        'Train_binary_acc': train_binary_acc,
        'Train_precision': train_precision,
        'Train_recall': train_recall,
        'Train_f1': train_f1,
        'Train_auc': train_auc,
        'Train_mcc': train_mcc,
        'Train_sensitivity': train_sensitivity,
        'Train_specificity': train_specificity,
        'Valid_loss': valid_losses,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        'Valid_binary_acc': valid_binary_acc,
        'Valid_precision': valid_precision,
        'Valid_recall': valid_recall,
        'Valid_f1': valid_f1,
        'Valid_auc': valid_auc,
        'Valid_mcc': valid_mcc,
        'Valid_sensitivity': valid_sensitivity,
        'Valid_specificity': valid_specificity,
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')

def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name,map_location='cpu'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", np.sqrt(epoch_loss_test_avg))
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test auc: ", result_test['auc'])
        print("Test mcc: ", result_test['mcc'])
        print("Test sensitivity: ", result_test['sensitivity'])
        print("Test specificity: ", result_test['specificity'])

        test_result[model_name] = [
            np.sqrt(epoch_loss_test_avg),
            result_test['pearson'],
            result_test['r2'],
            result_test['binary_acc'],
            result_test['precision'],
            result_test['recall'],
            result_test['f1'],
            result_test['auc'],
            result_test['mcc'],
            result_test['sensitivity'],
            result_test['specificity'],
        ]

        test_detail_dataframe = pd.DataFrame({'gene': test_name, 'solubility': test_true, 'prediction': test_pred})
        test_detail_dataframe.sort_values(by=['gene'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'pearson', 'r2', 'binary_acc', 'precision',
                                                            'recall', 'f1', 'auc', 'mcc', 'sensitivity', 'specificity'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')

def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    pearson = pearsonr(y_true, y_pred)
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
        'pearson': pearson,
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


def cross_validation(all_dataframe,fold_number=5):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['gene'].values
    sequence_labels = all_dataframe['solubility'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model()
        if torch.cuda.is_available():
            model.cuda()

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1

def single_train(all_dataframe):
    sequence_names = all_dataframe['gene'].values
    sequence_labels = all_dataframe['solubility'].values
    kfold = KFold(n_splits=10, shuffle=True)
    model = Model()
    if torch.cuda.is_available():
        model.cuda()
    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        train(model, train_dataframe, valid_dataframe)
        break

if __name__ == "__main__":
    train_dataframe = pd.read_csv(Dataset_Path + 'eSol_train.csv',sep=',')
    single_train(train_dataframe)
    test_dataframe = pd.read_csv(Dataset_Path + 'eSol_test.csv', sep=',')
    test(test_dataframe) 
