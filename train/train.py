import os,argparse,torch,itertools
import torchvision.datasets as dset
import numpy as np
from torch.autograd import Variable
import train.models as mod

modle = ""
os.makedirs("../images", exist_ok=True)
os.makedirs("../models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--cuda", type=int, default=False, help="enables cuda")
if modle == "acgan":
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
opt = parser.parse_args()
print(opt)

opt.cuda = opt.cuda if torch.cuda.is_available() else False
# aimGen:黑盒攻击目标生成器
# dataloader:训练集；在黑盒攻击场景下作为辅助train
# aux_testloader:黑盒攻击辅助train
def aae_train(opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
    aae = mod.aae()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    n_row = 8
    fake_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))

    # Initialize generator and discriminator
    encoder = aae.Encoder(opt,Tensor)
    decoder = aae.Decoder(opt)
    discriminator = aae.Discriminator(opt)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            aux_test_imgs = None
            for j, (fake_imgs, _) in enumerate(aux_testloader):
                aux_test_imgs = fake_imgs
                break
            log = aae.train(encoder, decoder, optimizer_G, optimizer_D,
                            discriminator, adversarial_loss, pixelwise_loss,
                            opt, imgs_shape, Tensor,
                            imgs=imgs, aimGen=aimGen, aux_test_imgs=aux_test_imgs)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), log[0], log[1])
            )
        aae.epoch_save(epoch, modPath, imgPath, encoder, decoder, discriminator, imgs, fake_noise)
    aae.final_save(modPath, encoder, decoder, discriminator)

def began_train(opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
    began = mod.began()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    n_row = 8
    for i, (imgs, _)  in enumerate(dataloader, 0):
        fake_noise = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        break

    # Initialize generator and discriminator
    generator = began.Generator(opt)
    discriminator = began.Discriminator(opt)

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(mod.weights_init_normal)
    discriminator.apply(mod.weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    gamma = 0.75
    lambda_k = 0.001
    k = 0.0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            aux_test_imgs = None
            for j, (fake_imgs, _) in enumerate(aux_testloader):
                aux_test_imgs = fake_imgs
                break
            log = began.train(generator, discriminator, optimizer_G, optimizer_D, opt, imgs_shape,
                              gamma, lambda_k, k, Tensor,
                              imgs=imgs, aimGen=aimGen, aux_test_imgs=aux_test_imgs)
            k = log[3]
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(dataloader), log[0], log[1], log[2], log[3])
            )
        began.epoch_save(epoch, modPath, imgPath, generator, discriminator, imgs, fake_noise)
    mod.final_save(modPath, generator, discriminator)

def dcgan(opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
    dcgan = mod.dcgan()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    for i, (imgs,_) in dataloader:
        fake_noise = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        break

    # Initialize generator and discriminator
    generator = dcgan.Generator(opt)
    discriminator = dcgan.Discriminator(opt)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(mod.weights_init_normal)
    discriminator.apply(mod.weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            aux_test_imgs = None
            for j, (fake_imgs, _) in enumerate(aux_testloader):
                aux_test_imgs = fake_imgs
                break
            log = dcgan.train(generator, discriminator,optimizer_G, optimizer_D,
                              adversarial_loss, opt, imgs_shape,Tensor,
                              imgs=imgs, aimGen=aimGen, aux_test_imgs=aux_test_imgs)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), log[0], log[1])
            )
        dcgan.epoch_save(epoch, modPath, imgPath, generator, discriminator, imgs, fake_noise)
    mod.final_save(modPath, generator, discriminator)

def acgan_train(opt, dataloader, modPath, imgPath):
    acgan = mod.acgan()

    FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

    n_row = 8
    fake_noise = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    noise_label = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])))

    # Initialize generator and discriminator
    generator = acgan().Generator(opt)
    discriminator = acgan().Discriminator(opt)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            log = acgan.train(generator, discriminator, optimizer_G, optimizer_D,
                        adversarial_loss, auxiliary_loss, opt, imgs, labels,
                        FloatTensor, LongTensor)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), log[0], 100 * log[1], log[2])
            )
        acgan.epoch_save(epoch, modPath, imgPath, generator, discriminator, imgs, fake_noise, noise_label)
    mod.final_save(modPath, generator, discriminator)
