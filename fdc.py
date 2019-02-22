import math
import torch
import pickle
import numpy as np
import os
from skimage import transform


depth_size = (25, 32)
ncoeff = depth_size[0] * (math.floor(depth_size[1] / 2) + 1) * 2 # Conjugate symmetry
crop_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FDC:

    def __init__(self, model):
        self.weights = None
        self.bias = None
        self.model = model
        
        
    def __call__(self, batch):
        predictions = []
        for i in range(batch.shape[0]):
            with torch.no_grad():
                result = self.model(batch[i])
            candidates = self.merge_crops(result)
            f_m_hat = self.img2fourier(candidates)
            f_hat = self.predict(f_m_hat)
            d_hat = self.fourier2img(f_hat.view(1, -1), depth_size)
            predictions.append(d_hat[0])

        return predictions
        


    def fit(self, f_m_hat, f):
        """
        f_m_hat: (T, M, K)
        f: (T, K)
        """
        print('Fitting FDC wieghts...')
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


    def save_weights(self, path_to_dir):
        print('Saving FDC wieghts...')
        with open(os.path.join(path_to_dir, 'fdc_weights.p') ,'wb') as w:
            pickle.dump(self.weights, w)
        with open(os.path.join(path_to_dir, 'fdc_bias.p') ,'wb') as b:
            pickle.dump(self.bias, b)


    def load_weights(self, path_to_dir):
        with open(os.path.join(path_to_dir, 'fdc_weights.p') ,'rb') as w:
            self.weights = pickle.load(w)
        with open(os.path.join(path_to_dir, 'fdc_bias.p') ,'rb') as b:
            self.bias = pickle.load(b)


    def forward(self, dataloader):
        print('Forward phase')
        f_m_hat = torch.empty([len(dataloader.dataset), len(crop_ratios), ncoeff])
        f = torch.empty([len(dataloader.dataset), ncoeff])

        self.model.eval()
        with torch.no_grad():
            for t, data in enumerate(dataloader):
                inputs = data['stacked_images'].to(device).float()
                labels = data['depth'].to(device).float()

                bsz, ncrops, c, h, w = inputs.size()
                result = self.model(inputs.view(-1, c, h, w))
                candidates = self.merge_crops(result)
                f_m_hat[t] = self.img2fourier(candidates)
                f[t] = self.img2fourier(labels.view(1, depth_size[0], depth_size[1]))

        return f_m_hat, f


    def merge_crops(self, crops):
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
            merged = transform.resize(merged, depth_size, mode='reflect',
                                      anti_aliasing=True, preserve_range=True).astype('float32')
            merged_crops[r] = torch.from_numpy(merged)

        return merged_crops


    def img2fourier(self, images):
        """
        images: (batch, N1, N2)
        """
        images_fd = torch.empty([images.shape[0], ncoeff])
        for i in range(images.shape[0]):
            images_fd[i] = torch.rfft(images[i], 2).view(-1)
        return images_fd


    def fourier2img(self, images_fd, shape):
        """
        images_fd: (batch, ncoeff)
        shape: shape of the original image
        """
        images = torch.empty([images_fd.shape[0], shape[0], shape[1]])
        for i in range(images_fd.shape[0]):
            img_fd = images_fd[i].view(shape[0], math.floor(shape[1] / 2) + 1, 2)
            images[i] = torch.irfft(img_fd, 2, signal_sizes=shape)
        return images