import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse

def build_model(num_labels, hidden_units=4096, arch='vgg19'):
    # Load a pre-trained model
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Usupported network architecture', arch)
        
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Extract features only by removing the last layer
    features = list(model.classifier.children())[:-1]
    
    # Number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features
    
    # Extend the existing architecture with new classification layers  
    features.extend([
        nn.Dropout(),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
    ])
    
    model.classifier = nn.Sequential(*features)
    
    return model
    
def train_model(data_dir, hidden_units=2048, arch='vgg19', epochs=20, learning_rate=0.001, gpu=True, checkpoint=''):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'   
    
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),        
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),        
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),        
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])    
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=2),
        'valid': data.DataLoader(image_datasets['valid'], batch_size=4, shuffle=True, num_workers=2),
        'test': data.DataLoader(image_datasets['test'], batch_size=4, shuffle=True, num_workers=2)
    } 
    
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
    
    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = build_model(num_labels, hidden_units, arch)
    
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using: GPU')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using: CPU')
        device = torch.device("cpu")     
    
    # Defining criterion, optimizer and scheduler.
    # Only parameters that require gradients are optimized.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_model_state = copy.deepcopy(model.state_dict())
    best_model_accuracy = 0.0
    time_start = time.time()

    for epoch in range(epochs):
        print()
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        for phase in ['train', 'valid']:
            is_train = phase == 'train'
            is_valid = phase == 'valid'

            if is_train:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Clear the gradients, do this because gradients are accumulated
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train): # track history if only in train
                    # forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass and optimize only if in training phase
                    if is_train:
                        loss.backward()
                        optimizer.step()

                # get statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_accuracy = running_corrects.double()/len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            # deep copy the model
            if is_valid and epoch_accuracy > best_model_accuracy:
                best_model_accuracy = epoch_accuracy
                best_model_state = copy.deepcopy(model.state_dict())

    time_end = time.time() - time_start
    print()
    print('Training completed in {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60))
    print('Best Accuracy: {:4f}'.format(best_model_accuracy))

    # Load best model weights
    model.load_state_dict(best_model_state)

    # Store class_to_idx into a model property
    model.class_to_idx = image_datasets['train'].class_to_idx    
    
    # Save checkpoint if required
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units
        }
        torch.save(checkpoint_dict, checkpoint)
    
    # Return the model
    return model
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
    
    args = parser.parse_args()
    
    params = {}
    if args.hidden_units:
        params['hidden_units'] = args.hidden_units
    if args.arch:
        params['arch'] = args.arch
    if args.epochs:
        params['epochs'] = args.epochs
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.gpu:
        params['gpu'] = args.gpu
    if args.checkpoint:
        params['checkpoint'] = args.checkpoint
        
    train_model(args.data_dir, **params)
    