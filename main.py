import numpy as np
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import itertools
from tqdm import trange
from scipy.spatial.distance import cdist

from dataset import LogoDataset

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logo_data = torchvision.datasets.ImageFolder('/media/hwt/Elements SE/logos',
                                            transform=transforms.Compose([
                                                transforms.ToTensor()])
                                            )

# logo_data = LogoDataset('/media/hwt/Elements SE/logos')
data_size = len(logo_data)

data_loader = torch.utils.data.DataLoader(logo_data, shuffle=True)


train_data, test_data, remaining = torch.utils.data.random_split(logo_data, [500,100,data_size-500-100])
train_loader = torch.utils.data.DataLoader(train_data, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, shuffle = True)


## Model ##
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Sequential()
resnet18.to(device)


def feature_extractor(loader):
    gallery = []
    with torch.no_grad():
        for _ in trange(len(loader)):
            x, y = next(iter(loader))
            x = x.to(device)
            feature = resnet18(x)
            gallery.append(feature.detach().cpu().numpy())
    return gallery

train_gallery = feature_extractor(train_loader)
test_gallery = feature_extractor(test_loader)

eu_dists = []
# for i in trange(len(test_gallery)):
for j in range(len(train_gallery)):
    eu_dist = cdist(test_gallery[0],train_gallery[j],'euclidean')
    eu_dists.append(eu_dist.item())

eu_dists_sorted = eu_dists.copy()
eu_dists_sorted.sort()
sim_idx = []
for i in eu_dists_sorted[0:10]:
    idx_in_subset = eu_dists.index(i)
    idx_in_dataset = train_data.indices[idx_in_subset]
    sim_idx.append(idx_in_dataset)
    

# ##**** Visualization *****##
# def imshow(image):
#     out = torchvision.utils.make_grid(image)
#     plt.imshow(out.numpy().transpose((1,2,0)))
#     plt.show()

# o_img,_ = next(iter(test_loader))
# imshow(o_img)
# for i in sim_idx:
#     compare_img,_ = logo_data[sim_idx[i]]
#     imshow(compare_img)
