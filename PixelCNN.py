import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import json
import cv2
from VQ_VAE import *


class MaskedConv2d(nn.Conv2d):
    """
    Implements a conv2d with mask applied on its weights.

    Args:
        mask (torch.Tensor): the mask tensor.
        in_channels (int) – Number of channels in the input image.
        out_channels (int) – Number of channels produced by the convolution.
        kernel_size (int or tuple) – Size of the convolving kernel
    """

    def __init__(self, mask, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        # 把mask变成二维并且注册一个常变量
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        self.weight.data *= self.mask  # mask weights
        return super().forward(x)


class VerticalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height (k//2, k), but for simplicity, we stick with masking here.
        # 用来判断是 A mask 还是 B mask
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0] // 2, :] = 1.0
        # B类型，当前预测的位置需要设置为已知
        if self.mask_type == "B":
            mask[kernel_size[0] // 2, :] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)


class HorizontalStackConv(MaskedConv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size)
        assert kernel_size[0] == 1
        if "padding" in kwargs:
            if isinstance(kwargs["padding"], int):
                kwargs["padding"] = (0, kwargs["padding"])

        mask = torch.zeros(kernel_size)
        mask[:, :kernel_size[1] // 2] = 1.0
        if self.mask_type == "B":
            mask[:, kernel_size[1] // 2] = 1.0

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)


class GatedMaskedConv(nn.Module):

    def __init__(self, in_channels, kernel_size=3, dilation=1):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2
        self.conv_vert = VerticalStackConv("B", in_channels, 2 * in_channels, kernel_size, padding=padding,
                                           dilation=dilation)
        self.conv_horiz = HorizontalStackConv("B", in_channels, 2 * in_channels, kernel_size, padding=padding,
                                              dilation=dilation)
        # 1*1的卷积核映射
        self.conv_vert_to_horiz = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        # 分成两路
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        # 通过门
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        # 接收水平的信息
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        # 通过门
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        # 输出
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        # shortcut
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class GatedPixelCNN(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super().__init__()

        # Initial first conv with mask_type A
        self.conv_vstack = VerticalStackConv("A", in_channels, channels, 3, padding=1)
        self.conv_hstack = HorizontalStackConv("A", in_channels, channels, 3, padding=1)
        # Convolution block of PixelCNN. use dilation instead of
        # downscaling used in the encoder-decoder architecture in PixelCNN++
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(channels),
            GatedMaskedConv(channels, dilation=2),
            GatedMaskedConv(channels)
        ])

        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        # first convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        # 输出
        out = self.conv_out(F.elu(h_stack))
        return out


class PixelCNN(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(PixelCNN, self).__init__()
        self.model = GatedPixelCNN(in_channels, channels, out_channels)
        self.num_embeddings = in_channels

    def Train(self,
              train_indices,
              train_loader,
              epochs,
              print_freq,
              pixcelCNN_name,
              log_name,
              lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        log = []
        for epoch in range(epochs):
            print("Start training epoch {}".format(epoch, ))
            for i, (indices) in enumerate(train_indices):
                indices = indices.cuda()
                one_hot_indices = F.one_hot(indices, self.num_embeddings).float().permute(0, 3, 1, 2).contiguous()
                outputs = self.model(one_hot_indices)
                loss = F.cross_entropy(outputs, indices)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                    print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))
                    log.append({"loss": loss.item()})
        f = open(log_name, 'w', encoding='utf-8')
        f.write(json.dumps(log, ensure_ascii=False, indent=4))
        f.close()
        torch.save(self.model, pixcelCNN_name)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    # dataset1 = datasets.MNIST('data/', train=True, download=True,
    #                           transform=transform)
    # dataset2 = datasets.MNIST('data/', train=False,
    #                           transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1, batch_size=128, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset2, batch_size=128)
    # train_indices = []
    # encoder = torch.load("encoder.pth")
    # vq = torch.load("vq.pth")
    # for images, labels in train_loader:
    #     images = images - 0.5
    #     images = images.cuda()
    #     with torch.inference_mode():
    #         z = encoder(images)
    #         b, c, h, w = z.size()
    #         z = z.permute(0, 2, 3, 1).contiguous()
    #         flat_z = z.reshape(-1, c)
    #         encoding_indices = vq.get_code_indices(flat_z)
    #         encoding_indices = encoding_indices.reshape(b, h, w)
    #         train_indices.append(encoding_indices.cpu())
    # model = PixelCNN(128, 128, 128).cuda()
    # model.Train(train_indices, train_loader, 20, 500, "pixcel_CNN.pth", "pixcel_log.json", 1e-3)
    pixelcnn = torch.load("pixcel_CNN.pth")
    n_samples = 10
    prior_size = (7, 7)  # h, w
    priors = torch.zeros((n_samples,) + prior_size, dtype=torch.long).cuda()

    # use pixelcnn to generate priors
    pixelcnn.eval()

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(prior_size[0]):
        for col in range(prior_size[1]):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            with torch.inference_mode():
                one_hot_priors = F.one_hot(priors, 128).float().permute(0, 3, 1, 2).contiguous()
                logits = pixelcnn(one_hot_priors)
                probs = F.softmax(logits[:, :, row, col], dim=-1)
                # Use the probabilities to pick pixel values and append the values to the priors.
                priors[:, row, col] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

    vq_layer = torch.load("vq.pth")
    decoder = torch.load("decoder.pth")
    with torch.inference_mode():
        z = vq_layer.quantize(priors)
        z = z.permute(0, 3, 1, 2).contiguous()
        pred = decoder(z)

    generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
    generated_samples = generated_samples.reshape(n_samples, 28, 28)
    for i in range(n_samples):
        filename = str(i+1) + ".png"
        cv2.imwrite(filename, generated_samples[i])
