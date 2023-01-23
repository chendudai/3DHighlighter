import argparse
import clip
import copy
import json
import kaolin as kal
import kaolin.ops.mesh
import numpy
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision
import imageio
from itertools import permutations, product
from neural_highlighter import NeuralHighlighter
from Normalization import MeshNormalizer
# from mesh import Mesh
from pathlib import Path
from render import Renderer
from tqdm import tqdm
from torch.autograd import grad
from torchvision import transforms
from utils import device, color_mesh
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def optimize(agrs):
    # Constrain most sources of randomness
    # (some torch backwards functions within CLIP are non-determinstic)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load CLIP model 
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    res = 224
    Path(os.path.join(args.output_dir, 'renders')).mkdir(parents=True, exist_ok=True)

    # door_clipseg = torch.load('./data/door.pkl')
    # door_clipseg[door_clipseg<0.5] = 0
    # door_clipseg[door_clipseg>=0.5] = 1
    # plt.imshow(door_clipseg.unsqueeze(dim=2))
    # plt.show()

    # Initialize variables
    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    n_augs = args.n_augs
    dir = args.output_dir

    # Record command line arguments
    with open(os.path.join(dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])
    
    # MLP Settings
    mlp = NeuralHighlighter(args.depth, args.width, out_dim=args.n_classes, positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)
    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate)

    # list of possible colors
    rgb_to_color = {(204/255, 1., 0.): "highlighter", (180/255, 180/255, 180/255): "gray"}
    color_to_rgb = {"highlighter": [204/255, 1., 0.], "gray": [180/255, 180/255, 180/255]}
    # full_colors = [[204/255, 1., 0.], [180/255, 180/255, 180/255]]
    # full_colors = [[0, 1., 0.], [180 / 255, 180 / 255, 180 / 255]]
    # full_colors = [[204 / 255, 1., 0.], [0, 0, 0]]
    full_colors = [[0, 1., 0.], [0, 0, 0]]
    colors = torch.tensor(full_colors).to(device)

    
    # --- Prompt ---

    # encode prompt with CLIP
    # prompt = "A photo of a green statue with a black background".format(args.object[0], args.classes[0])
    # prompt = "A photo of a statue over a black background"
    # prompt = "A photo of a statue on a green screen"
    prompt = "cathedral with a green portal"
    save_dir = './results/0053/cathedral_with_a_green_portal'

    with torch.no_grad():
        prompt_token = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(prompt_token)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

    # image = preprocess(Image.open('./data/statue.jpg'))
    image = preprocess(Image.open('./data/0053.jpg'))
    image /= image.max()
    image = image.to(device)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1]).astype(np.float32)
    y = np.linspace(0, image.shape[2] - 1, image.shape[2]).astype(np.float32)
    img_coord = []
    for i in x:
        for j in y:
            coord1 = (i / (image.shape[1] - 1)).astype(np.float32)
            coord2 = (j / (image.shape[2] - 1)).astype(np.float32)
            img_coord.append([coord1, coord2])

    img_coord = torch.tensor(img_coord).cuda()
    # loss_cosine = nn.CosineEmbeddingLoss()
    # loss_cosine = nn.CrossEntropyLoss()
    # vertices = copy.deepcopy(mesh.vertices)
    losses = []

    # Optimization loop
    for iter in tqdm(range(args.n_iter)):

        optim.zero_grad()

        pred_class = mlp(img_coord)

        pred_image = pred_class.reshape((image.shape[1], image.shape[2], 2))
        pred_image = pred_image.cuda()
        # rendered_image_channel = ((image[0,:,:]* 0.5 + pred_image*0.5))
        # R_channel = rendered_image_channel.unsqueeze(dim=0)

        # R_channel =  image[0, :, :].unsqueeze(dim=0)
        # G_channel =  image[1, :, :].unsqueeze(dim=0)
        # B_channel = image[2, :, :].unsqueeze(dim=0)
        #
        # rendered_image = torch.concat((R_channel, G_channel, B_channel), dim=0)
        # pred_image[pred_image < 0.15] = 0
        # pred_image[pred_image >= 0.5] = 1

        # rendered_image = torch.multiply(pred_image[:,:,0], image)

        # rendered_image = torch.zeros(3,224,224).to(device)
        # for class_idx, color in enumerate(colors):
        #     class_pred_image = pred_image[:, :, class_idx]
        #     for c in range(3):
        #         rendered_image[c, :, :] = (0.05 * image[c, :, :] + 0.95 * class_pred_image) * color[c]


        alpha = 0.2
        rendered_image = alpha * image + (1-alpha) * (pred_image @ colors).permute(2,0,1)

        # rendered_image = (rendered_image.unsqueeze(dim=0) * 255).int()
        rendered_image_unsqueezed = rendered_image.unsqueeze(dim=0)

        # Calculate CLIP Loss

        rendered_image_unsqueezed = clip_transform(rendered_image_unsqueezed)
        encoded_renders = clip_model.encode_image(rendered_image_unsqueezed)
        encoded_renders = encoded_renders / encoded_renders.norm(dim=1, keepdim=True)
        loss = 1-torch.nn.functional.cosine_similarity(encoded_renders, encoded_text)
        loss.backward(retain_graph=True)

        optim.step()

        # update variables + record loss
        with torch.no_grad():
            losses.append(loss.item())
            print(loss.item())

            if (iter) % 1000 == 0:
                save_results(image, rendered_image, pred_image, iter, save_dir, colors)


    save_results(image, rendered_image, pred_image, iter, save_dir, colors)






# ================== HELPER FUNCTIONS =============================
def save_results(image, rendered_image, pred_image, i, save_dir, colors):
    os.makedirs(save_dir, exist_ok=True)

    if i == 0:
        original_image = image.permute(1, 2, 0).cpu().detach().numpy()
        imageio.imwrite(save_dir + '/original' + '.png', original_image)

    output_image = rendered_image.permute(1, 2, 0).cpu().detach().numpy()
    imageio.imwrite(save_dir + '/output_' + str(i) + '.png', output_image)

    pred_image_firstClass = pred_image[:,:,0].cpu().detach().numpy()
    imageio.imwrite(save_dir + '/pred_firstClass_' + str(i) + '.png', pred_image_firstClass)

    pred_image_firstClass_threshold = pred_image[:, :, 0]
    pred_image_firstClass_threshold[pred_image_firstClass_threshold<0.5] = 0
    pred_image_firstClass_threshold[pred_image_firstClass_threshold>0.5] = 1
    pred_image_firstClass_threshold = pred_image_firstClass_threshold.cpu().detach()

    image = image.cpu().detach()
    alpha = 0.1
    image_firstClass_threshold = alpha * image[0,:,:] + (1 - alpha) * pred_image_firstClass_threshold
    pred_firstClass_Threshold = torch.concat((image_firstClass_threshold.unsqueeze(dim=2), image[1,:,:].unsqueeze(dim=2), image[2,:,:].unsqueeze(dim=2)),dim=2)
    imageio.imwrite(save_dir + '/pred_firstClass_Threshold' + str(i) + '.png', pred_firstClass_Threshold)

def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'renders/iter_{}.jpg'.format(i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # directory structure
    parser.add_argument('--obj_path', type=str, default='data/scenes/scene.obj')
    parser.add_argument('--output_dir', type=str, default='results/segment/1')

    # mesh+prompt info
    parser.add_argument('--prompt', nargs="+", default='a pig with pants')
    parser.add_argument('--object', nargs=1, default='cow')
    parser.add_argument('--classes', nargs="+", default='sphere cube')

    # render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--n_augs', type=int, default=1)
    parser.add_argument('--clipavg', type=str, default='view')
    parser.add_argument('--render_res', type=int, default=224)

    # CLIP
    parser.add_argument('--clipmodel', type=str, default='ViT-L/14')
    parser.add_argument('--jit', action="store_true")

    # network
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--n_iter', type=int, default=5000)

    args = parser.parse_args()

    optimize(args)
