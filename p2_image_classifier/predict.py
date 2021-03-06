""" predict.py
Predict flower name from an image with predict.py along with the probability of
that name. That is, you'll pass in a single image /path/to/image and return the
flower name and class probability.

Basic usage:
    python predict.py /path/to/image checkpoint

Args:
    --img
    --checkpoint
    --top_k
    --category_names
    --gpu

Returns:
    Most likely flower names and class probabilities

Examples:
    Return top K most likely classes:
        python predict.py input checkpoint --top_k 3

    Use a mapping of categories to real names:
        python predict.py input checkpoint --category_names cat_to_name.json

    Use GPU for inference:
        python predict.py input checkpoint --gpu
"""
__author__ = "Ken Norton <ken@kennethnorton.com>"

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms, models


def main():
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image with along with the \
        probability of that name. ')

    parser.add_argument('--img',
                        type=str,
                        dest='image_path',
                        default='flowers/train/1/image_06770.jpg',
                        action='store',
                        help='File path to an image input')
    parser.add_argument('--checkpoint',
                        default='model_checkpoint.pth',
                        type=str,
                        dest='checkpoint',
                        action='store',
                        help='Model checkpoint')
    parser.add_argument('--top_k',
                        dest='top_k',
                        type=int,
                        default=5,
                        action='store',
                        help='Number of top classes to return')
    parser.add_argument('--category_names',
                        type=str,
                        dest='category_names',
                        default='cat_to_name.json',
                        action='store',
                        help='JSON file containing category-name mapping')
    parser.add_argument('--gpu',
                        type=bool,
                        dest='gpu',
                        default=True,
                        action='store',
                        help='Number of epochs')

    pa = parser.parse_args()
    path_to_img = pa.image_path
    checkpoint = pa.checkpoint
    top_k = pa.top_k
    category_names = pa.category_names
    gpu = pa.gpu

    if (torch.cuda.is_available() and gpu):
        gpu = True
        device = torch.device('cuda:0')
    else:
        gpu = False
        device = torch.device('cpu')

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    def load_checkpoint(filepath):
        cp = torch.load(filepath)

        if cp['architecture'] == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.name = 'vgg16'
        elif cp['architecture'] == 'alexnet':
            model = models.alexnet(pretrained=True)
            model.name = 'alexnet'

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.class_to_idx = cp['class_to_idx']
        model.classifier = cp['classifier']
        model.load_state_dict(cp['state_dict'])

        return model

    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a NumPy array
        '''
        img_handler = Image.open(image)
        process_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return process_img(img_handler)

    def predict(image_path, model):
        model.to(device)
        img_torch = process_image(image_path)
        img_torch = img_torch.unsqueeze_(0)
        img_torch = img_torch.float()

        with torch.no_grad():
            output = model.forward(img_torch.cuda())

        probability = F.softmax(output.data, dim=1)

        pb, cl = probability.topk(top_k)
        if gpu:
            return pb.cpu().numpy(), cl.cpu().numpy()
        else:
            return pb.numpy(), cl.numpy()

    def print_predict(image_path, model):
        probs, classes = predict(image_path, model)
        probabilities = probs[0]
        class_names = {val: key for key, val in model.class_to_idx.items()}
        c_names = [cat_to_name[class_names[x]] for x in classes[0]]
        index = np.arange(len(c_names))
        print('Predictions for ', image_path, ':')
        for i in index:
            prob = "{0:.2f}".format(probabilities[i] * 100)
            print(prob, '% -- ', c_names[i])

    model = load_checkpoint(checkpoint)

    print_predict(path_to_img, model)


if __name__ == "__main__":
    main()
