from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from utils import selection_dataset, partition_dataset, CustomDataset
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from lora import LoRALayer, print_trainable_parameters, LoRA_Config, mark_only_lora_as_trainable, apply_lora_to_model
from train import train
from evaluation import evaluate
import pdb as pdb
import yaml
import time
from transformers import CLIPModel, AutoProcessor, AutoTokenizer

dataset = load_dataset("food101")
shuffled_dataset, selected_labels = selection_dataset(dataset, 5)
tra, val, test = partition_dataset(shuffled_dataset, selected_labels)

transform = transforms.Compose([
    transforms.Resize((224, 224)), #for ResNet
    transforms.ToTensor()
])

train_dataset = CustomDataset(tra, transform)
val_dataset = CustomDataset(val, transform)
test_dataset = CustomDataset(test, transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False)


model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
preprocessor = AutoProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(
    model_name,
    #load_in_8bit = True,
    #device_map = 'auto',
)

print("** Befor Adapting Lora **")
print_trainable_parameters(model)

pdb.set_trace()

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
    lora_config = LoRA_Config(
        r=cfg['lora']['r'], 
        lora_alpha=cfg['lora']['alpha'], 
        lora_dropout=cfg['lora']['dropout'], 
        merge_weights=cfg['lora']['merge_weights'], 
        target_modules=cfg['lora']['target_modules'],
    )
    model = apply_lora_to_model(model, lora_config)
    mark_only_lora_as_trainable(model)

    print("** After Adapting Lora **")
    print_trainable_parameters(model)


pdb.set_trace()

num_epochs = 20 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

# start_time = time.time()
train(model, device, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs = 10)
# print("Training Time : ", time.time() - start_time)

evaluate(model, device, test_loader)
