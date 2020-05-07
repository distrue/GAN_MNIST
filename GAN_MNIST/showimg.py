import math
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import torch
from torch.autograd import Variable 
from IPython import display

from .model import GAN_MNIST_GENERATOR

def printImg():
    num_test_samples = 9

    test_noise = Variable(torch.randn(num_test_samples, 100).cuda())

    size_figure_grid = int(math.sqrt(num_test_samples))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)

    gpu = True

    generator = GAN_MNIST_GENERATOR().cuda()

    checkpoint = Path(__file__).absolute().parent.parent / 'checkpoint_generator.pt'
    if checkpoint.exists():
        print("loading generator checkpoint from {}".format(checkpoint))
        generator.load_state_dict(torch.load(checkpoint))
    
    test_images = generator(test_noise)

    for k in range(num_test_samples):
        i = k//3
        j = k%3
        ax[i,j].cla()
        ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.savefig('mnist-gan.png')
