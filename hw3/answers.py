r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=10,
        seq_len=2,
        h_dim=60,
        n_layers=2,
        dropout=0,
        learn_rate=0.009,
        lr_sched_factor=0.01,
        lr_sched_patience=0.01,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size']        = 256
    hypers['seq_len']           = 64
    hypers['h_dim']             = 256
    hypers['n_layers']          = 5
    hypers['dropout']           = 0.1
    hypers['learn_rate']        = 0.0015
    hypers['lr_sched_factor']   = 0.1
    hypers['lr_sched_patience'] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = "This deep assiement is the real masterpiece "
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.0008
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


We had several reasons to split our corpus into sequences:

First of all, if we wouldn't split the corpus it would require to fit all the data in the memory at once, which can be impossible sometimes.
In addition, training over all the corpus would make the training extrimly long, since every batch will need to go over all the text before making a gradient step.
Moreover, it might cause vanishing or exploding gradients, when using very long sequence, as we saw in the lectures.

"""

part1_q2 = r"""
**Your answer:**

In our training process the model learned all the corpus, and 'storing' (using its parameters, and hidden state) information that was gained
from the entire corpus, and related to all of it. Therefore, it can generate text using this learning, 'based' also on the courpus and not just the sequence it was givven.
This practice allows the system to generate text with longer memory.
"""

part1_q3 = r"""
**Your answer:**


Each batch is organized such that each sequence continues the corresponding sequence from the previous batch.
Tht way we enable the model to learn longer sequences from the corpus. By passing the hidden state from one batch,
to the other, we can learn sequences that span over all the batches in the ds, without including them in a single foward pass.
If we were shuffling hte batches, we would break this property and mess up the training of the hidden state,
and by that, the network wont be able to produce coherent long sequences.
"""

part1_q4 = r"""
**Your answer:**


1. We lower the temperature for sampling since when we do that, it enhance the most probable characters in hope to get a more coherent sequence and to ensure that the generated text will make sense.

2. When the temperature is very high the generated text will be less likely to make sense.
   The distribution converges to a uniform distribution and there is an equal probability to get every character.
   Its like samp;ing charecthers from (almost) uniform distribution, and then the model has almost no effect on the choices for the next character.

3. When the temp is very low, the generated text will be more likely to have proper structure, and with 'correct' words,
   but much more repetitve. The variance of the generated twxt will be very low, and we will see littel different between different parts in the output text.




1. We lower the temperature for sampling because we want to enhance the most probable characters in hope to get a more coherent sequence.

2. When the temperature is very high, the distribution converges to a uniform distribution and there is an equal probability to get every character.

3. When the temp is very low, the generated text will be more likely to have proper structure, and with 'correct' words,
but much more repetitve. The variance of the generated twxt will be very low, and we will see 

 the generated text will more likely have a proper structure with words that have no typos but it will also be more repetitive 
 since most of the generated characters other than the very few that the model has deemed the most likely to appear will have a 
 probability that approaches zero and therefore the variance of the generated text will also be very low and we will see little 
 difference between the different parts in our “play”. For example, in our case we saw the phrase “SCENE II” and “exit” appear 
 more and more as we lowered the temprature.

1. We lower the temperature for sampling because when we do that we strengthen the probability of characters that the model has deemed more likely to appear next and weaken the probability of less likely characters and in that way we are able to ensure that the generated text will make more sense (with real words, proper structure etc...)

2. We can also understand that by noting the fact that if we sample from a uniform or almost uniform distribution then the model has almost no effect on the choice of what the next character will be.

3. When the temprature is very low the generated text will more likely have a proper structure with words that have no typos but it will also be more repetitive since most of the generated characters other than the very few that the model has deemed the most likely to appear will have a probability that approaches zero and therefore the variance of the generated text will also be very low and we will see little difference between the different parts in our “play”. For example, in our case we saw the phrase “SCENE II” and “exit” appear more and more as we lowered the temprature.



"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 16
    hypers['h_dim'] = 1024  
    hypers['z_dim'] = 200  
    hypers['x_sigma2'] = 0.0008
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.9, 0.99)  
    # ========================
    return hypers


part2_q1 = r"""

The $\sigma^2$ hyperparameter is used to determine the importance of the KL loss. <br />
meaning:the larger x_sigma2 will give larger weight to the KL loss and smaller values will give larger weight to the reconstruction loss. <br />
This hyperparamater also controls the variance of the data generated with noise, <br />
since the more importance given to the KL loss by a higher sigma, the more randomness is allowed in the decoder mapping. 


"""

part2_q2 = r"""

1. the reconstruction loss term is the exception over all sampels of z sampled from passing our  <br />
x through the encoder. which means how good we reconstructed from our latent space. <br />
the KL divergence loss applies regularization on the output of the encoder in the latent space and also showing
how close or far our approximated latent space posterior distribution from our latent space prior distribution <br />

2. The KL term is used to make the latent space distribution closer to gaussian distribution.

3. What we gain from this effect on the latent space distribution that we can sample 
from it without prior knowlage of the network that was used to make the distribution space. And thats why
we have more freedom with our choise of model that generates an image.
"""

part2_q3 = r"""

Maximizing the evidence distribution will increase our chances for the model 
to generate a similare image to the given ones in our dataset. Since we want to find a structure of the data we have.  
"""

part2_q4 = r"""

When we look at the log function on positive numbers that are smaller than one, it generates results with a much wider spread,
so when we have numerical errors they have less effect. Furthermore, when using log calculating determinant
in the loss function it produces faster results because the logs are added instead of multiplied. 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 16
    hypers['z_dim'] = 50
    hypers['data_label'] = 0
    hypers['label_noise'] = 0.07
    hypers['discriminator_optimizer'] = dict(type='Adam',
                                             lr=20e-5,
                                             betas=(0.5, 0.999)
                                             )
    hypers['generator_optimizer'] = dict(type='Adam',
                                         lr=33e-5,
                                         betas=(0.5, 0.999)
                                         )
    # ========================
    return hypers


part3_q1 = r"""

The reason for that sometimes when we sample from the GAN we need to maintain gradiants and sometimes we dont is that we are training two different networks at the same time. <br />
So, for the generator we do not disable the gradient because we need it to do back propogation and update the
generator weights.

When we train the discriminator, the samples from the generator are used as training data, and we do not want the loss of the discirminator to propogate to the generator 
since we want the discriminator to be able to distinguish between real and fake images without changing the fake images (that would be cheating) so we do not keep the gradiant <br />

"""

part3_q2 = r"""
1. We should not stop the training when the generator loss reaches a certain threshhold since our model's success depends on the performance of the discriminator and the generator's relationship. <br />
If both the Generator and the Discriminator are doing bad jobs and one when hand noisy images are generated and on the other hand
they are still able to fool the descriminator, 
the generator might achive a better loss then a case where the Generator generates highly realistic images while the discriminator is
still able to distinguish the generated ones better and the generator loss will be much higher. <br />
we can not conclude how the generator is performing using the Generator loss alone. Since the Generator loss is dependent on the performance of the Discriminator.  <br />

2. It means that the descriminator is missclasifying more fake images as real and the generators loss decreases.
but its also classifying more real images as real so the descriminators loss is stagnant.   
It can mean that both the discriminator and the generator are improving at the same time but at different rates, and if we wouldnt train them at
the same time we would have gotten a decrease in the descriminator loss.

"""

part3_q3 = r"""

From our results there are some differences in the generated images:
First of all the generated GAN images are alot more detailed and sharp in the background compared to VAE 
which result as blurrirer and less detailed outside the center of the face . the background of the VAE images is blurriness compared to the GAN can be 
explained by the fact that less weight is given to it and more to features that reappear in all images such as the facial features, on the other hand for the GAN's descriminator, a blurry background could have been
an indication of a fake image therefore the generator was encouraged to make it more detailed and realistic. the same case can be made for how deatiled a suite and tie are in the GAN 
compared to a blurrier result in VAE. This is due to the different goals of VAE and GAN when producing
examples. VAE's goal is reconstruction of an example while GAN's goal is generating an image that is indistinguishable 
from images labeled in the same way. <br />
In addition, the VAE produces better facial features than the GAN.
Furthermore, VAE's results are more similer to one another and more consistent with each other. 

"""

# ==============
