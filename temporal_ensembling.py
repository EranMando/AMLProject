import numpy as np
from timeit import default_timer as timer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import calc_metrics, prepare_mnist, weight_schedule,prepare_kmnist,prepare_fashion_mnist,prepare_emnist
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_curve
from sklearn import metrics
import time

    

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# cross validation function
def crossValid(model, seed,dataset):
    n_classes = 10
    batch_size = 100
    max_epochs = 80
    max_val = 30
    ramp_up_mult = -5.
    k = 100
    n_samples = 60000

    # Changeable(you can add or delete values from the arrays)
    lrVals = [1e-4,1e-5,1e-6]
    momentumVals = [0.5,0.7,0.9]
    # Configuration options(changeable)
    k_folds = 10
    num_epochs = 50

    results = [[' ' for x in range(len(lrVals) * len(momentumVals))] for y in range(k_folds)]
    torch.manual_seed(42)

    dataset_train_part, dataset_test_part = dataset()
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    ntrain = len(dataset_train_part)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')
    t1=time.perf_counter()
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        t1=time.perf_counter()
        counter = 0
        for learnningrate in lrVals:
            for moment in momentumVals:
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                print(f'Learning Rate = {learnningrate} \t momentum = {moment}\n')
                # Define data loaders for training and testing data in this fold
                trainloader, testloader, indices = sample_train(dataset_train_part, dataset_test_part, 100,
                                                                100, 10, seed, shuffle_train=False)

                model.apply(reset_weights)

                # Initialize optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=learnningrate, momentum=moment)

                current_loss = 0.0
                z = torch.zeros(ntrain, n_classes).float().cuda()  # temporal outputs
                outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

                for epoch in range(num_epochs):
                    # evaluate unsupervised cost weight
                    w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
                    if (epoch + 1) % 10 == 0:
                        print('unsupervised loss weight : {}'.format(w))

                    # turn it into a usable pytorch object
                    w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
                    l = []
                    supl = []
                    unsupl = []
                    for i, (images, labels) in enumerate(trainloader):
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda(), requires_grad=False)

                        optimizer.zero_grad()
                        out = model(images)
                        zcomp = Variable(z[i * batch_size: (i + 1) * batch_size], requires_grad=False)
                        loss, suploss, unsuploss, nbsup = temporal_loss(out, zcomp, w, labels)

                        # save outputs and losses
                        outputs[i * batch_size: (i + 1) * batch_size] = out.data.clone()
                        l.append(loss.item())
                        supl.append(nbsup * suploss.item())
                        unsupl.append(unsuploss.item())

                        # backprop
                        loss.backward()
                        optimizer.step()

                        # Print statistics
                        current_loss += loss.item()
                        if i % 500 == 499:
                            print('Loss after mini-batch %5d: %.3f' %
                                  (i + 1, current_loss / 500))
                            current_loss = 0.0
                # Process is complete.
                print('Training process has finished. Saving trained model.')
                # Print about testing
                print('Starting testing')
                # Saving the model
                save_path = f'./model-fold-{fold}.pth'
                torch.save(model.state_dict(), save_path)
                # Evaluationfor this fold
                correct, total = 0, 0
                with torch.no_grad():
                    # Iterate over the test data and generate predictions
                    for i, (images, labels) in enumerate(testloader, 0):
                        # Get inputs
                        inputs = Variable(images.cuda())
                        targets = Variable(labels.cuda(), requires_grad=False)

                        # Generate outputs
                        outputs = model(inputs)
                        # Set total and correct
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                    t2=time.perf_counter()
                    # Print accuracy
                    print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                    print('--------------------------------')
                    results[fold][
                        counter] = f'{100.0 * (correct / total)} - Learning Rate = {learnningrate} - momentum = {moment} - time to train ={t2-t1} '
                    counter += 1

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for foldNum in range(k_folds):
        print(f'\nFold {foldNum} Results:')
        curSum = 0.0
        for value in results[foldNum]:
            acc = value.split()
            curSum += float(acc[0])
            sum += float(acc[0])
            print(f'\t Accuracy : {value}')
        print(f'\n\t\tFold {foldNum} Average = {curSum / 9}\n')
    print(f'\nTotal Average: {sum / (k_folds * len(lrVals) * len(momentumVals))} %')


def sample_train(train_dataset, test_dataset, batch_size, k, n_classes,
                 seed, shuffle_train=True, return_idxs=True):
    
    n = len(train_dataset)
    rrng = np.random.RandomState(seed)
    
    cpt = 0
    indices = torch.zeros(k)
    other = torch.zeros(n - k)
    card = k // n_classes
    
    for i in range(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()[:, 0]#class_items = (train_dataset.train_labels == i).nonzero()
        n_class = len(class_items)
        rd = np.random.permutation(np.arange(n_class))
        indices[i * card: (i + 1) * card] = class_items[rd[:card]]
        other[cpt: cpt + n_class - card] = class_items[rd[card:]]
        cpt += n_class - card

    other = other.long()
    train_dataset.train_labels[other] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size,
                                              num_workers=0,
                                              shuffle=False)
    
    if return_idxs:
        return train_loader, test_loader, indices 
    return train_loader, test_loader


def temporal_loss(out1, out2, w, labels):
    
    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()
    
    def masked_crossentropy(out, labels):
        cond = (labels >= 0)
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs, masked_labels)
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0
    
    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


def train(model, seed,dataset, k=100, alpha=0.6, lr=0.002, beta2=0.99, num_epochs=15,
          batch_size=100, drop=0.5, std=0.15, fm1=16, fm2=32,
          divide_by_bs=False, w_norm=False, data_norm='pixelwise',
          early_stop=None, c=300, n_classes=10, max_epochs=80,
          max_val=30., ramp_up_mult=-5., n_samples=60000,
          print_res=True, **kwargs):
    
    
    
    # retrieve data
    train_dataset, test_dataset = dataset()
    ntrain = len(train_dataset)
    print(train_dataset)

    #for train_dataset,test_dataset in rkf.split(train_dataset1,test_dataset1)
    # build model
    model.cuda()

    #crossValid(model, seed,dataset)  # put this line as a note to reduce runtime if crossValid func runs fine (tested well)
    print('\n\n')
    
    # make data loaders
    train_loader, test_loader, indices = sample_train(train_dataset, test_dataset, batch_size,
                                                      k, n_classes, seed, shuffle_train=False)
    print(train_loader)
    # setup param optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # train
    t1=time.perf_counter()
    model.train()
    losses = []
    sup_losses = []
    unsup_losses = []
    best_loss = 20.

    Z = torch.zeros(ntrain, n_classes).float().cuda()        # intermediate values
    z = torch.zeros(ntrain, n_classes).float().cuda()        # temporal outputs
    outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

    for epoch in range(num_epochs):
        t = timer()
        
        # evaluate unsupervised cost weight
        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
     
        if (epoch + 1) % 10 == 0:
            print ('unsupervised loss weight : {}'.format(w))
        
        # turn it into a usable pytorch object
        w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
        
        l = []
        supl = []
        unsupl = []
        for i, (images, labels) in enumerate(train_loader):  
            images = Variable(images.cuda())
            
            labels = Variable(labels.cuda(), requires_grad=False)
            
            # get output and calculate loss
            optimizer.zero_grad()
            out = model(images)
            zcomp = Variable(z[i * batch_size: (i + 1) * batch_size], requires_grad=False)
            loss, suploss, unsuploss, nbsup = temporal_loss(out, zcomp, w, labels)

            # save outputs and losses
            outputs[i * batch_size: (i + 1) * batch_size] = out.data.clone()
            l.append(loss.item())
            supl.append(nbsup * suploss.item())
            unsupl.append(unsuploss.item())

            # backprop
            loss.backward()
            optimizer.step()

            # print loss
            if (epoch + 1) % 10 == 0:
                if i + 1 == 2 * c:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f, Time (this epoch): %.2f s' 
                           %(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, np.mean(l), timer() - t))
                elif (i + 1) % c == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f' 
                           %(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, np.mean(l)))

        # update temporal ensemble
        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))

        # handle metrics, losses, etc.
        # print(l,type(l))
        eloss = np.mean(l)
        # eloss=torch.mean(l).detach().cpu().numpy()
        losses.append(eloss)
        sup_losses.append((1. / k) * np.sum(supl))  # division by 1/k to obtain the mean supervised loss
        unsup_losses.append(np.mean(unsupl))
        
        # saving model 
        if eloss < best_loss:
            best_loss = eloss
            torch.save({'state_dict': model.state_dict()}, 'model_best.pth.tar')

    # test
    model.eval()
    acc,y_pred,y_true = calc_metrics(model, test_loader)
    t2=time.perf_counter()
    trues=[]
    preds=[]
    i=0
    while(i<len(y_true)):
        trues.extend(y_true[0].cpu().numpy())
        i=i+1
    i=0
    while(i<len(y_pred)):
        preds.extend(y_pred[0].cpu().numpy())
        i=i+1
    #print('list for ypred --------------------------------')
    #print(preds)
    #print('list for ytrue --------------------------------')
    #print(trues)
    if print_res:
        print ('Accuracy of the network on the 10000 test images: %.2f %%' % (acc))
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(trues)): 
        if trues[i]==preds[i]:
           TP += 1
        if preds[i]>=1 and trues[i]!=preds[i]:
           FP += 1
        if trues[i]==preds[i]==0:
           TN += 1
        if preds[i]==0 and trues[i]!=preds[i]:
           FN += 1
    TPR=float(TP/(TP+FN))
    FPR=float(FP/(FP+TN))
    precision=float(TP/(TP+FP))
    recall=float(TP/(TP+FN))
    fpr, tpr, thresholds = metrics.roc_curve(trues, preds, pos_label=2)
    #pos_probs = preds[:, 1]
    #precision, recall2, thresholds = precision_recall_curve(trues, preds)
    auc=metrics.auc(fpr, tpr)
    print(recall)
    print(precision)
    #prerecall=metrics.auc(recall,precision)
    
    print('TPR:')
    print(TPR)
    print(tpr)
    print('FPR:')
    print(FPR)
    print('precision:')
    print(precision)
    print('auc')
    print(auc)
    print('Area under the Precision-Recall')
    #print(prerecall)
    print('time to train:')
    print(t2-t1)
    
    
    # test best model
    checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    acc_best = acc
    if print_res:
        print ('Accuracy of the network (best model) on the 10000 test images: %.2f %%' % (acc_best))
     
    
    return acc, acc_best, losses, sup_losses, unsup_losses, indices







