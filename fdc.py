import math
import torch
import numpy as np
from skimage import transform


depth_size = (25, 32)
ncoeff = depth_size[0] * (math.floor(depth_size[1] / 2) + 1) * 2 # Conjugate symmetry
crop_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_fdc(model):
    f_m_hat = torch.empty([len(train_loader_fdc.dataset), len(crop_ratios), ncoeff])
    f = torch.empty([len(train_loader_fdc.dataset), ncoeff])

    model.eval()
    with torch.no_grad():
        for t, (data, label) in enumerate(train_loader_fdc):
            data = data.to(device)
            bsz, ncrops, c, h, w = data.size()
            result = model(data.view(-1, c, h, w))
            candidates = merge_crops(result)
            f_m_hat[t] = img2fourier(candidates)
            f[t] = img2fourier(label.view(1, depth_size[0], depth_size[1]))

    return f_m_hat, f


def merge_crops(crops):
    h, w = depth_size
    merged_crops = torch.empty([len(crop_ratios), h, w])
    for r in range(len(crop_ratios)):
        ratio = crop_ratios[r]
        merged = torch.zeros([round(h / ratio), round(w / ratio)])
        weights = torch.zeros(merged.shape)

        for i in range(4):
            crop = crops[r * 4 + i].view(depth_size) / ratio
            if i == 0:  # Top-left
                merged[:h, :w] = crop
                weights[:h, :w] += 1
            elif i == 1:  # Top-right
                merged[:h, -w:] = crop
                weights[:h, -w:] += 1
            elif i == 2:  # Bottom-left
                merged[-h:, :w] = crop
                weights[-h:, :w] += 1
            elif i == 3:  # Bottom-right
                merged[-h:, -w:] = crop
                weights[-h:, -w:] += 1

        merged = np.array(merged / weights)
        merged = transform.resize(merged, depth_size, preserve_range=True).astype('float32')
        merged_crops[r] = torch.from_numpy(merged)

    return merged_crops


def img2fourier(images):
    """
    images: (batch, N1, N2)
    """
    images_fd = torch.empty([images.shape[0], ncoeff])
    for i in range(images.shape[0]):
        images_fd[i] = torch.rfft(images[i], 2).view(-1)
    return images_fd


def fourier2img(images_fd, shape):
    """
    images_fd: (batch, ncoeff)
    shape: shape of the original image
    """
    images = torch.empty([images_fd.shape[0], shape[0], shape[1]])
    for i in range(images_fd.shape[0]):
        img_fd = images_fd[i].view(shape[0], math.floor(shape[1] / 2) + 1, 2)
        images[i] = torch.irfft(img_fd, 2, signal_sizes=shape)
    return images


class FDC:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, f_m_hat, f):
        """
        f_m_hat: (T, M, K)
        f: (T, K)
        """
        T, M, K = f_m_hat.shape
        self.weights = torch.zeros(M, K)
        self.bias = torch.zeros(M, K)

        for k in range(K):
            t_k = f[:, k].unsqueeze_(1)
            b_k = torch.mean(f_m_hat[:, :, k] - t_k, 0, True)
            T_k = f_m_hat[:, :, k] - b_k
            w_k = torch.mm(torch.pinverse(T_k), t_k)
            self.weights[:, k] = w_k.squeeze_()
            self.bias[:, k] = b_k.squeeze_()

    def predict(self, f_m_hat):
        """
        f_m_hat: (M, K)
        """
        return torch.sum(self.weights * (f_m_hat - self.bias), 0)