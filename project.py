import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.utils import prune

# Wybór urządzenia (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ustawienia trenowania
num_classes = 26           # liczba klas w EMNIST Letters
num_epochs = 12            # ile epok trenować
batch_size = 64            # rozmiar batcha
learning_rate = 0.001      # współczynnik uczenia

# Augmentacja i normalizacja danych wejściowych
transform = transforms.Compose([
    transforms.RandomRotation(10),                         # losowy obrót o max 10°
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # losowe przesunięcia obrazu
    transforms.ToTensor(),                                 # konwersja do tensora
    transforms.Normalize((0.5,), (0.5,))                   # normalizacja do [-1, 1]
])

# Ładowanie zbiorów danych EMNIST Letters
train_dataset = datasets.EMNIST(
    root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(
    root='./data', split='letters', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # loader treningowy
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)       # loader testowy

# Definicja sieci LeNet-5 z BatchNorm i Dropout
class ImprovedLeNet5(nn.Module):
    def __init__(self, num_classes=26):
        super(ImprovedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 4 * 4)     # spłaszczenie wejścia do fc
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)            # regularizacja
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)            # regularizacja
        x = self.fc3(x)
        return x

# Inicjalizacja modelu i przeniesienie na urządzenie
model = ImprovedLeNet5(num_classes=num_classes).to(device)

# PRUNING: globalnie 20% wag z wybranych warstw, przed treningiem
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight')
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2                              # procent wszystkich wag
)

# Optymalizator i funkcja kosztu
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Early stopping & najlepszy wynik
best_acc = 0.0
patience = 3
patience_counter = 0

# TRENING z zapisem modelu z pruningiem (z maskami wag)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device) - 1      # etykiety od 0 do 25
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # EWALUACJA po każdej epoce
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device) - 1
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Test accuracy: {100*acc:.2f}%')

    # Early stopping oraz zapis najlepszego modelu
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_improved_lenet5_pruned.pt") # z maskami
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# USUWANIE REPARAMETRYZACJI PRUNINGOWEJ (po treningu, tylko raz!)
print("\nUsuwanie reparametryzacji pruningowej...")
for module, name in parameters_to_prune:
    prune.remove(module, name)                             # bez masek

# Zapis finalnego modelu (tylko zwykłe wagi)
torch.save(model.state_dict(), "best_improved_lenet5_final.pt")

# ŁADOWANIE FINALNEGO MODELU i EWALUACJA
model.load_state_dict(torch.load("best_improved_lenet5_final.pt"))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device) - 1
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Dokładność i raport klasyfikacji
correct = np.sum(np.array(y_true) == np.array(y_pred))
total = len(y_true)
print(f'\nTest Accuracy (Best Model): {100 * correct / total:.2f}%')
print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

# MACIERZ POMYŁEK (wizualizacja)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - EMNIST Letters (Improved)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.colorbar()
plt.xticks(np.arange(num_classes), [chr(i+65) for i in range(num_classes)])
plt.yticks(np.arange(num_classes), [chr(i+65) for i in range(num_classes)])
plt.show()
