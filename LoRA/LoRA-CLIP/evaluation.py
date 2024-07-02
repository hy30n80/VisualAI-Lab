import torch


def evaluate(model, device, test_loader, PATH="best_model.pth"):
    model.load_state_dict(torch.load(PATH))
    model.eval()

    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(test_loader, start = 1):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy}%")
    