import torch
from torch.utils import data
from torchvision.transforms import Compose
from torchvision.models import resnet152
import os
import fdc
import transforms_nyu
from dataset import NyuV2

data_path = './data/nyu_v2/'

seed = 2
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

depth_size = (25, 32)
model_input = 224
test_crop = (427, 561)
crop_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

transform = Compose([
    transforms_nyu.Scale(),
    transforms_nyu.FDCPreprocess(crop_ratios)
])

nyu = NyuV2(os.path.join(data_path, 'train'), transform=transform)

dataloader = data.DataLoader(nyu, batch_size=1, shuffle=True, num_workers=6)

fdc_model = fdc.FDC()

model = resnet152(pretrained=True)
model.fc = torch.nn.Linear(2048, depth_size[0] * depth_size[1])
wts = './models/resnet_crop_test/114_model.pt'
model.load_state_dict(torch.load(wts))
model = model.to(device)
model.eval()

f_m_hat, f = fdc_model.forward(model, dataloader)

fdc_model.fit(f_m_hat, f)
fdc_model.save_weights('./models/FDC/')