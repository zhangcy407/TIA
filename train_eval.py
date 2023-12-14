import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model.TIAM import TIAM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import wandb
import random


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the random seed
set_seed()


def load_data(data_path):
    with open(data_path, 'rb') as file:
        dataset = pickle.load(file)
    features = torch.tensor(np.array(dataset['features']))
    labels = torch.tensor(np.array(dataset['labels']))
    return TensorDataset(features, labels)


def load_data_for_final(data_path_wav2vec2, data_path_librosa):
    with open(data_path_wav2vec2, 'rb') as file:
        dataset_wav2vec2 = pickle.load(file)
    with open(data_path_librosa, 'rb') as file:
        dataset_librosa = pickle.load(file)
    features_wav2vec2 = torch.tensor(np.array(dataset_wav2vec2['features']))
    labels = torch.tensor(np.array(dataset_wav2vec2['labels']))
    features_librosa = torch.tensor(np.array(dataset_librosa['features']))
    return TensorDataset(features_wav2vec2, features_librosa, labels)


def train_model(model, train_loader, val_loader, test_loader, epochs, dataset_path, model_name, device='cuda'):
    wandb.init(project='TIA')
    model.train()
    save_model_path = dataset_path + '/saved_dict/' + model_name + '.ckpt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_batch = 0
    val_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()
    for epoch in range(epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs))
        for batch in train_loader:
            wav2vec2_features, librosa_features, labels = batch
            wav2vec2_features, librosa_features, labels = wav2vec2_features.to(device), librosa_features.to(device), labels.to(device)
            outputs = model(wav2vec2_features, librosa_features)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = accuracy_score(true, predic)
                val_acc, val_loss = evaluate_model(model, val_loader, model_name)
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    torch.save(model.state_dict(), save_model_path)
                    # torch.save(model, config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%} {5}'
                print(msg.format(total_batch, loss.item(), train_acc, val_loss, val_acc, improve))
                wandb.log(
                    {'Val_Acc': val_acc, 'Train_Loss': loss.item(), 'Val_Loss': val_loss})
                model.train()
            total_batch += 1
            if total_batch - last_improve > 1500:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(model, test_loader, save_model_path, model_name)


def test(model, test_iter, save_model_path, model_name):
    # test
    model.load_state_dict(torch.load(save_model_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate_model(model, test_iter, model_name, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate_model(model, data_loader, model_name, test=False, device='cuda'):
    model.eval()
    loss_total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            wav2vec2_features, librosa_features, labels = batch
            wav2vec2_features, librosa_features, labels = wav2vec2_features.to(device), librosa_features.to(device), labels.to(device)
            outputs = model(wav2vec2_features, librosa_features)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    if test:
        report = classification_report(all_labels, all_predictions, target_names=['no', 'yes'], digits=4)
        confusion = confusion_matrix(all_labels, all_predictions)
        return accuracy, loss_total / len(data_loader), report, confusion
    return accuracy, loss_total / len(data_loader)


def train_and_evaluate(model_name, dataset_path, batch_size, epochs):
    if model_name == 'TIAM':
        model = TIAM().to('cuda')
        wav2vec2_dataset_path = dataset_path + '/wav2vec2/'
        librosa_dataset_path = dataset_path + '/librosa/'
        wav2vec2_train_path = os.path.join(wav2vec2_dataset_path, 'train_data.pkl')
        wav2vec2_val_path = os.path.join(wav2vec2_dataset_path, 'val_data.pkl')
        wav2vec2_test_path = os.path.join(wav2vec2_dataset_path, 'test_data.pkl')
        librosa_train_path = os.path.join(librosa_dataset_path, 'train_data.pkl')
        librosa_val_path = os.path.join(librosa_dataset_path, 'val_data.pkl')
        librosa_test_path = os.path.join(librosa_dataset_path, 'test_data.pkl')
        print('loading data...')
        train_data = load_data_for_final(wav2vec2_train_path, librosa_train_path)
        val_data = load_data_for_final(wav2vec2_val_path, librosa_val_path)
        test_data = load_data_for_final(wav2vec2_test_path, librosa_test_path)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_model(model, train_loader, val_loader, test_loader, epochs, dataset_path, model_name)
