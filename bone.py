import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class LabelledData(Dataset):

    def __init__(self, predictors, labels):
        self.predictors = predictors
        self.labels = labels

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, idx):
        return self.predictors[idx], self.labels[idx]


class UnlabelledData(Dataset):

    def __init__(self, predictors):
        self.predictors = predictors

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, idx):
        return self.predictors[idx]


class BoneClassification(nn.Module):
    def __init__(self):
        super(BoneClassification, self).__init__()
        self.lin_1 = nn.Linear(in_features=12, out_features=100)
        self.lin_2 = nn.Linear(in_features=100, out_features=200)
        self.lin_3 = nn.Linear(in_features=200, out_features=100)
        self.lin_4 = nn.Linear(in_features=100, out_features=1)

        self.batchnorm_1 = nn.BatchNorm1d(num_features=100)
        self.batchnorm_2 = nn.BatchNorm1d(num_features=200)
        self.batchnorm_3 = nn.BatchNorm1d(num_features=100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.batchnorm_1(self.relu(self.lin_1(x)))
        x = self.batchnorm_2(self.relu(self.lin_2(x)))
        x = self.batchnorm_3(self.relu(self.lin_3(x)))
        x = self.dropout(x)
        x = self.lin_4(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_set = np.genfromtxt(r"train-io.txt", delimiter=" ")
data = training_set[:, 0:12]
labels = training_set[:, 12]

training_predictors, validation_predictors, training_labels, validation_labels = train_test_split(
    data, labels, test_size=0.20, random_state=11)

test_set = np.genfromtxt(r"test-in.txt", delimiter=" ")
test_predictors = test_set[:, 0:12]

scaler = StandardScaler()
training_predictors = scaler.fit_transform(training_predictors)
validation_predictors = scaler.fit_transform(validation_predictors)
test_predictors = scaler.fit_transform(test_predictors)

training_dataset = LabelledData(torch.FloatTensor(
    training_predictors), torch.FloatTensor(training_labels))
validation_dataset = UnlabelledData(torch.FloatTensor(validation_predictors))
testing_dataset = UnlabelledData(torch.FloatTensor(test_predictors))

training_loader = DataLoader(
    dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)
testing_loader = DataLoader(dataset=testing_dataset, batch_size=1)

model = BoneClassification()
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)
model.train()

if os.path.isfile('bone.pth'):
    print("Model found, starting evaluation...")

    model.load_state_dict(torch.load('bone.pth'))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    THRESHOLD_MONKEYING = 0.24

    def evaluator(loader):
        prediction_list = []
        with torch.no_grad():
            for predictors in loader:
                predictors = predictors.to(device)
                temp_prediction = model(predictors)
                temp_prediction = torch.sigmoid(temp_prediction)
                predicted_label = torch.round(
                    temp_prediction+THRESHOLD_MONKEYING)
                prediction_list.append(predicted_label.cpu().numpy())

        prediction_list = [l.squeeze().tolist() for l in prediction_list]
        return prediction_list

    validation_predictions = evaluator(validation_loader)
    test_predictions = evaluator(testing_loader)

    print(confusion_matrix(validation_labels, validation_predictions))
    print(classification_report(validation_labels, validation_predictions))

    test_predictions = [round(x) for x in test_predictions]
    with open('test-output.txt', 'w') as f:
        for label in test_predictions:
            f.write("%s\n" % label)
else:
    print("No model found, starting training...")

    for epoch in range(300):
        for predictors, labels in training_loader:
            predictors, labels = predictors.to(device), labels.to(device)
            optimiser.zero_grad()
            predictions = model(predictors)
            loss = loss_function(predictions, labels.unsqueeze(1))
            loss.backward()
            optimiser.step()
        print("Epoch:"+str(epoch+1))

    torch.save(model.state_dict(), ".\\bone.pth")
