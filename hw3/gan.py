import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import hw3.autoencoder as autoencoder

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        out_channels = 1024
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = autoencoder.EncoderCNN(in_channels=in_size[0], out_channels=out_channels).to(device=self.device)
        encoder_in_size = (1,in_size[0],in_size[1],in_size[2])
        flatter_in_size = (self.encoder(torch.zeros(encoder_in_size).to(device=self.device))).reshape(-1).shape[0]
        self.flatter =  nn.Linear(flatter_in_size, 1).to(device=self.device)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        # y = self.cnn(x)
        features = self.encoder(x)
        hidden = features.view(features.shape[0], -1)
        y = self.flatter(hidden)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.hidden_dims = 1024
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.featuremap_size = featuremap_size
        self.decoder = autoencoder.DecoderCNN(self.hidden_dims, out_channels=out_channels).to(self.device)
        self.linear = nn.Linear(z_dim, self.hidden_dims * (featuremap_size **2)).to(self.device)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn((n, self.z_dim), device=device)
        if with_grad:
            samples = self(z)
        else:
            with torch.no_grad():
                samples = self(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ====== 
        h = self.linear(z).reshape(z.shape[0], self.hidden_dims, self.featuremap_size, self.featuremap_size)
        x = self.decoder(h)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    loss_fn = torch.nn.BCEWithLogitsLoss()
    noisy_data = data_label + label_noise * (torch.rand(size=y_data.shape, device=device) - 0.5)
    generated_labels = (1 - data_label) + label_noise * (torch.rand(size=y_data.shape, device=device) - 0.5)
    loss_data = torch.mean(loss_fn(y_data, noisy_data.to(device)).to(device))
    loss_generated = torch.mean(loss_fn(y_generated, generated_labels.to(device)).to(device))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generated_labels = torch.ones_like(y_generated) * data_label
    loss = loss_fn(y_generated, generated_labels.to(device)).to(device)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    device = x_data.device
    dsc_optimizer.zero_grad()
    non_gen_y = dsc_model(x_data).to(device)
    gen_y = dsc_model(gen_model.sample(x_data.shape[0], with_grad=False)).to(device)
    dsc_loss = dsc_loss_fn(non_gen_y, gen_y)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    gen_y = dsc_model(gen_model.sample(x_data.shape[0], with_grad=True))
    gen_loss = gen_loss_fn(gen_y)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    last_losses = gen_losses[-4:]
    best_loss = min(last_losses)
    last_loss = gen_losses[-1]
    if last_loss <= best_loss:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
