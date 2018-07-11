import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from train import build_model

import json

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image 

def predict(image_path, checkpoint, top_k=5, gpu=True):
    checkpoint_dict = torch.load(checkpoint)

    arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    model = build_model(num_labels, hidden_units, arch)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']

    if gpu and torch.cuda.is_available():
        model.cuda()
        
    model.eval()
    
    # Predict the class from an image file
    image = process_image(image_path)
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) # this is for VGG   
        
    if gpu and torch.cuda.is_available():     
        image = image.cuda()   
    
    result = model(image).topk(top_k)
    
    if gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]
        
    classes = [list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(x)] for x in classes]
    
    return probs, classes  
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Image to predict')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint to use when predicting')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='JSON file containing category names')
    parser.add_argument('--top_k', type=int, help='Return top K predictions')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')    
    
    args = parser.parse_args()
    
    params = {}
    if args.top_k:
        params['top_k'] = args.top_k
    if args.gpu:
        params['gpu'] = args.gpu
        
    probs, classes = predict(args.image_path, args.checkpoint, **params) 

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    labels = [cat_to_name[x] for x in classes]
    
    print(list(zip(labels, probs)))    