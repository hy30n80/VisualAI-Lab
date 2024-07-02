from datasets import load_dataset
from torch.utils.data import Dataset



def selection_dataset(dataset, n=5):
    num_labels = len(dataset['train'].features['label'].names)
    num_train = len(dataset['train'])
    num_test = len(dataset['validation'])
    print(f"Number of train set: {num_train},  Number of test set: {num_test}")
    print(f"Number of labels in the dataset: {num_labels}")

    #101개의 label 이기에 데이터가 너무 많아서, 5개만 추려서 사용
    selected_labels = range(n)
    filtered_datset = dataset['train'].filter(lambda example: example['label'] in selected_labels)
    shuffled_dataset = filtered_datset.shuffle(seed=42)
    return shuffled_dataset, selected_labels


def partition_dataset(dataset, selected_labels, train_size = 300, val_size = 20, test_size = 50):
    train, val, test = [], [], []
    counts = {label: {'train': 0, 'val': 0, 'test': 0} for label in selected_labels}

    for example in dataset:
        label = example['label']
        if counts[label]['train'] < train_size:
            train.append(example)
            counts[label]['train'] += 1
        
        elif counts[label]['val'] < val_size:
            val.append(example)
            counts[label]['val'] += 1
        
        elif counts[label]['test'] < test_size:
            test.append(example)
            counts[label]['test'] += 1
    
    print(f"Total training examples: {len(train)}")
    print(f"Total validation examples: {len(val)}")
    print(f"Total test examples: {len(test)}")
    return train, val, test


class CustomDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        image = self.transform(image)
        label = example['label']
        return image, label