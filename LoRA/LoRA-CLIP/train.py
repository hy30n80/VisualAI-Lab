import torch
import torch.nn as nn
import pdb as pdb

def train(model, device, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs):
    epochs = num_epochs
    model = model.to(device)
    optimizer = optimizer
    criterion = criterion
    scheduler = scheduler

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            val_losses.append(val_running_loss / len(val_loader))
            val_accuracies.append(val_accuracy)

            #Save model if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_wts = model.state_dict()
        
            print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Train Accuracy {train_accuracy}%, Val Loss: {val_running_loss / len(val_loader)}, Val Accuracy: {val_accuracy}%')


    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model to disk
    torch.save(model.state_dict(), 'best_model.pth')

    print("Training completed")