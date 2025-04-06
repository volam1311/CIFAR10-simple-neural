from cifa import MyDataset
from model import Simple_Neural_Networks
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torch
if __name__ == '__main__':
    num_epochs = 1
    train_dataset = MyDataset(root = './cifar/cifar-10-batches-py',train = True, transform = ToTensor())
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=16,
        shuffle= True,
        num_workers=4,
        drop_last= True)
    test_dataset = MyDataset(root = './cifar/cifar-10-batches-py',train = False,transform = ToTensor())
    test_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=16,
        num_workers=4,
        drop_last= False
    )
    model = Simple_Neural_Networks(num_classess= 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum= 0.9)
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            outputs = model(images) #forward
            loss_value = criterion(outputs,labels)
            num_iter = len(train_dataloader)
            print("Epoch: {}/{}. Iteration: {}/{}. Loss: {}".format(epoch+1, num_epochs, iter+1,num_iter,loss_value))
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
    all_predictions = []
    all_labels = []
    model.eval()
    for iter, (images, labels) in enumerate(test_dataloader):
        all_labels.extend(labels)
        with torch.no_grad():
            predictions  = model(images)
            indicies = torch.argmax(predictions, dim = 1)
            all_predictions.extend(indicies)    
            loss_value = criterion(outputs, labels)
    all_labels = [label.item() for label in all_labels]
    all_predictions = [prediction.item() for prediction in all_predictions]
    print(all_predictions)
    print(all_labels)
    total_correct = 0
    for i in range(len(all_predictions)):
        if all_predictions[i] == all_labels[i]:
            total_correct += 1
    accuracy  = total_correct/len(all_labels)
    print(accuracy)
    exit(0)


