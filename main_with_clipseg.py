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
from neural_highlighter import NeuralHighlighter, SimpleNetwork
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
import torchvision.transforms as T

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

    res = 224
    Path(os.path.join(args.output_dir, 'renders')).mkdir(parents=True, exist_ok=True)

    door_clipseg = torch.load('/home/cc/students/csguests/chendudai/Thesis/repos/3DHighlighter/results/0732/statue.pkl')
    transform = T.Resize(size=(res, res))
    door_clipseg = transform(door_clipseg.unsqueeze(dim=0))
    # door_clipseg = torch.concat([door_clipseg, 1-door_clipseg],dim=0)
    door_clipseg[door_clipseg<0.5] = 0
    door_clipseg[door_clipseg>=0.5] = 1
    plt.imshow(door_clipseg.permute(1,2,0))


    input_image = Image.open('/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/dense/images/0732.jpg')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224)),
    ])
    img = transform(input_image).unsqueeze(0)
    y = 0.8*door_clipseg + 0.2*img
    plt.imshow(y.squeeze(dim=0).permute(1,2,0))


    plt.show()
    door_clipseg = door_clipseg.to(device)
    # door_clipseg = door_clipseg.unsqueeze(dim=0)

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


    # MLP Settings
    mlp = NeuralHighlighter(args.depth, args.width, out_dim=args.n_classes, positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)
    # mlp = SimpleNetwork().to(device)
    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate)


    with torch.no_grad():

        x = np.linspace(0, res - 1, res).astype(np.float32)
        y = np.linspace(0, res - 1, res).astype(np.float32)
        img_coord = []
        for i in x:
            for j in y:
                coord1 = (i / (res - 1)).astype(np.float32)
                coord2 = (j / (res - 1)).astype(np.float32)
                img_coord.append([coord1, coord2])

        img_coord = torch.tensor(img_coord).cuda()

    # loss_CrossEntropyLoss = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    losses = []

    # Optimization loop
    for iter in tqdm(range(args.n_iter)):

        optim.zero_grad()
        pred_class = mlp(img_coord)

        # Create Prediction Image
        # pred_image = torch.zeros(1,res, res)
        # for i in range(res):
        #     for j in range(res):
        #         pred_image[0,i,j] = pred_class[i*res + j,0] # + image[i,j,0]
        pred_image = pred_class.reshape((224,224,2))

        pred_image = pred_image[:,:,0].unsqueeze(dim=0)
        pred_image = pred_image.cuda()

        loss = loss_func(door_clipseg, pred_image)
        loss.backward(retain_graph=True)
        optim.step()

        # update variables + record loss
        with torch.no_grad():
            losses.append(loss.item())
            print(loss.item())

            if (iter) % 100 == 0:
                save_results(pred_image[0,:,:], iter, mlp)


    save_results(pred_image[0,:,:], iter, mlp)






# ================== HELPER FUNCTIONS =============================
def save_results(rendered_image, i, mlp):
    dir = './results/statue_0732/clipseg_train_statue_threshold=0.5'
    os.makedirs(dir, exist_ok=True)
    torch.save(rendered_image, dir + 'pred' + str(i) + '.pkl')

    plt.imshow(rendered_image.unsqueeze(dim=2).cpu().detach().numpy())

    # input_image = Image.open('/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/images/0053.jpg')
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize((224, 224)),
    # ])
    # img = transform(input_image).unsqueeze(0)
    # y = 0.8*rendered_image.unsqueeze(dim=0).cpu()+ 0.2*img
    # plt.imshow(y.squeeze(dim=0).permute(1,2,0))
    #




    plt.show()
    torch.save(mlp.state_dict(), dir + '/mlp_' + str(i) + '.pkl')


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
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_iter', type=int, default=1000)

    args = parser.parse_args()

    optimize(args)
