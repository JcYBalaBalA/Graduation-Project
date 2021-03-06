import torch,itertools
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils

class model_opt():
    # len(opt_list = 10)
    # n_epochs,batch_size,lr,b1,b2,n_cpu,
    # latent_dim,img_size,channels,sample_interval
    # n_classes可选
    def __init__(self, opt_list, cuda=False):
        self.n_epochs = opt_list[0]
        self.batch_size = opt_list[1]
        self.lr = opt_list[2]
        self.b1 = opt_list[3]
        self.b2 = opt_list[4]
        self.n_cpu = opt_list[5]
        self.latent_dim = opt_list[6]
        self.img_size = opt_list[7]
        self.channels = opt_list[8]
        self.sample_interval = opt_list[9]
        self.cuda = cuda
        if len(opt_list) == 11:
            self.n_classes = opt_list[10]

    def list_all_member(self):
        for name, value in vars(self).items():
            print('\t%s=%s ' % (name, value))

'''
class aae():
    def train(self, opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        fake_noise = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Initialize generator and discriminator
        encoder = self.Encoder(opt, Tensor)
        decoder = self.Decoder(opt)
        discriminator = self.Discriminator(opt)

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

        Batch = 0
        if dataloader != None:
            Batch += len(dataloader)
        if aux_testloader != None:
            Batch += len(aux_testloader)

        for epoch in range(opt.n_epochs):
            i = 0
            if dataloader != None:
                for i, (imgs, _) in enumerate(dataloader):
                    log = self.epoch(encoder, decoder, optimizer_G, optimizer_D,
                                    discriminator, adversarial_loss, pixelwise_loss,
                                    opt, Tensor,
                                    imgs=imgs, aimGen=aimGen, aux_test_imgs=None)
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, Batch, log[0], log[1])
                    )

            if aux_testloader != None:
                for j, (fake_imgs, _) in enumerate(aux_testloader):
                    log = self.epoch(encoder, decoder, optimizer_G, optimizer_D,
                                    discriminator, adversarial_loss, pixelwise_loss,
                                    opt, Tensor,
                                    imgs=None, aimGen=aimGen, aux_test_imgs=fake_imgs)
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i + j, Batch, log[0], log[1])
                    )

            if dataloader == None and aux_testloader == None:
                log = self.epoch(encoder, decoder, optimizer_G, optimizer_D, discriminator,
                                adversarial_loss, pixelwise_loss, opt, Tensor,
                                imgs=None, aimGen=aimGen, aux_test_imgs=None)

                print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, log[0], log[1])
                )
            self.epoch_save(epoch, modPath, imgPath, encoder, decoder, discriminator, fake_noise)

    # 训练函数
    # 每从datalodaer中调用一个bitch的数据就需要调用一次train函数
    # 返回两个模型的loss
    # for epoch in range(opt.n_epochs):
    #     for i, (imgs, _) in enumerate(dataloader):
    #         aae.epoch(encoder, decoder, discriminator, opt, imgs, Tensor)
    def epoch(self, encoder, decoder, optimizer_G, optimizer_D, discriminator,
              adversarial_loss, pixelwise_loss, opt, Tensor,
              imgs=None, aimGen=None, aux_test_imgs=None):

        # Adversarial ground truths
        valid = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = 0
        if imgs != None:
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
        else:
            real_imgs = z

        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)

        # 黑盒攻击时使用攻击目标生成器生成图片并加以真标签
        # 使用训练集辅助信息加以假标签
        if aimGen != None:
            aimgen_img = aimGen(z)
            real_loss += adversarial_loss(discriminator(aimgen_img), valid)
            if aux_test_imgs != None:
                aux_fake_img = Variable(aux_test_imgs.type(Tensor))
                fake_loss += adversarial_loss(discriminator(aux_fake_img), fake)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(fake_imgs), valid) \
                 + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        return d_loss.item(), g_loss.item()

    class Encoder(nn.Module):
        def __init__(self,opt,Tensor):
            super(aae.Encoder, self).__init__()
            self.Tensor = Tensor
            self.opt = opt
            self.img_shape = (opt.channels, opt.img_size, opt.img_size)
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.mu = nn.Linear(512, self.opt.latent_dim)
            self.logvar = nn.Linear(512, self.opt.latent_dim)

        def _reparameterization(self, mu, logvar):
            std = torch.exp(logvar / 2)
            sampled_z = Variable(self.Tensor(np.random.normal(0, 1, (mu.size(0), self.opt.latent_dim))))
            z = sampled_z * std + mu
            return z

        def forward(self, img):
            img_flat = img.view(img.shape[0], -1)
            x = self.model(img_flat)
            mu = self.mu(x)
            logvar = self.logvar(x)
            z = self._reparameterization(mu, logvar)
            return z



    class Decoder(nn.Module):
        def __init__(self,opt):
            super(aae.Decoder, self).__init__()
            self.opt = opt
            self.img_shape = (opt.channels, opt.img_size, opt.img_size)
            self.model = nn.Sequential(
                nn.Linear(self.opt.latent_dim, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, int(np.prod(self.img_shape))),
                nn.Tanh(),
            )

        def forward(self, z):
            img_flat = self.model(z)
            img = img_flat.view(img_flat.shape[0], *self.img_shape)
            return img


    class Discriminator(nn.Module):
        def __init__(self,opt):
            super(aae.Discriminator, self).__init__()
            self.opt = opt
            self.model = nn.Sequential(
                nn.Linear(self.opt.latent_dim, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            validity = self.model(z)
            return validity

    def epoch_save(self,epoch, modPath, imgPath, encoder, decoder, discriminator, fake_noise):
        torch.save(encoder, '%s/encoder_epoch_%06d.pth' % (modPath, epoch))
        torch.save(decoder, '%s/decoder_epoch_%06d.pth' % (modPath, epoch))
        torch.save(discriminator, '%s/netD_epoch_%06d.pth' % (modPath, epoch))
        fake = decoder(fake_noise)
        vutils.save_image(fake.data, '%s/fake_%06d.png' % imgPath, nrow=8, normalize=True)

    # def final_save(self, modPath, encoder, decoder, discriminator):
    #     torch.save(encoder, '%s/encoder_final.pth' % (modPath))
    #     torch.save(decoder, '%s/decoder_final.pth' % (modPath))
    #     torch.save(discriminator, '%s/discriminator_final.pth' % (modPath))
'''
# acgan 具有分类和生成功能，但我们不能对攻击目标生成器生成的img给以准确的标签
# 因此暂时不将其考虑在攻击实验中
class acgan():
    def train(self, opt, dataloader, modPath, imgPath):
        FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

        n_row = 8
        fake_noise = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        noise_label = Variable(LongTensor(np.array([num for _ in range(n_row) for num in range(n_row)])))

        # Initialize generator and discriminator
        generator = self.Generator(opt)
        discriminator = self.Discriminator(opt)

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
                log = self.epoch(generator, discriminator, optimizer_G, optimizer_D,
                                  adversarial_loss, auxiliary_loss, opt, imgs, labels,
                                  FloatTensor, LongTensor)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), log[0], 100 * log[1], log[2])
                )
            acgan.epoch_save(epoch, modPath, imgPath, generator, discriminator, fake_noise, noise_label)
        # final_save(modPath, generator, discriminator)

    # return D_loss D_accuracy G_loss
    def epoch(self, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss,
              auxiliary_loss, opt, imgs, labels, FloatTensor, LongTensor):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()
        return d_loss.item(), d_acc, g_loss.item()

    # def weights_init_normal(self,m):
    #     classname = m.__class__.__name__
    #     if classname.find("Conv") != -1:
    #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     elif classname.find("BatchNorm2d") != -1:
    #         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    class Generator(nn.Module):
        def __init__(self,opt):
            super(acgan.Generator, self).__init__()
            self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

            self.init_size = opt.img_size // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise, labels):
            gen_input = torch.mul(self.label_emb(labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self,opt):
            super(acgan.Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                """Returns layers of each discriminator block"""
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.conv_blocks = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4

            # Output layers
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
            self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax(dim=-1))

        def forward(self, img):
            out = self.conv_blocks(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            label = self.aux_layer(out)

            return validity, label

    def epoch_save(self, epoch, modPath, imgPath, generator, discriminator, fake_noise, labels):
        torch.save(generator, '%s/netG_epoch_%06d.pth' % (modPath, epoch))
        torch.save(discriminator, '%s/netD_epoch_%06d.pth' % (modPath, epoch))
        fake = generator(fake_noise, labels)
        vutils.save_image(fake.data, '%s/fake_%06d.png' % imgPath, nrow=8, normalize=True)

class began():
    def train(self, opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        fake_noise = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Initialize generator and discriminator
        generator = self.Generator(opt)
        discriminator = self.Discriminator(opt)

        if opt.cuda:
            generator.cuda()
            discriminator.cuda()

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        gamma = 0.75
        lambda_k = 0.001
        k = 0.0

        Batch = 0
        if dataloader != None:
            Batch += len(dataloader)
        if aux_testloader != None:
            Batch += len(aux_testloader)

        for epoch in range(opt.n_epochs):
            i = 0
            if dataloader != None:
                for i, (imgs, _) in enumerate(dataloader):
                    log = self.epoch(generator, discriminator, optimizer_G, optimizer_D, opt,
                                      gamma, lambda_k, k, Tensor,
                                      imgs=imgs, aimGen=aimGen, aux_test_imgs=None)
                    k = log[3]
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                        % (epoch, opt.n_epochs, i, Batch, log[0], log[1], log[2], log[3])
                    )

            if aux_testloader != None:
                for j, (fake_imgs, _) in enumerate(aux_testloader):
                    log = self.epoch(generator, discriminator, optimizer_G, optimizer_D, opt,
                                      gamma, lambda_k, k, Tensor,
                                      imgs=None, aimGen=aimGen, aux_test_imgs=fake_imgs)
                    k = log[3]
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                        % (epoch, opt.n_epochs, i + j, Batch, log[0], log[1], log[2], log[3])
                    )

            if dataloader == None and aux_testloader == None:
                log = self.epoch(generator, discriminator, optimizer_G, optimizer_D, opt,
                                  gamma, lambda_k, k, Tensor,
                                  imgs=None, aimGen=aimGen, aux_test_imgs=None)
                k = log[3]
                print(
                    "[Step %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                    % (epoch, opt.n_epochs, log[0], log[1], log[2], log[3])
                )
            self.epoch_save(epoch, modPath, imgPath, generator, discriminator, fake_noise)
        # final_save(modPath, generator, discriminator)

    #return D_loss G_loss M k
    def epoch(self, generator, discriminator, optimizer_G, optimizer_D, opt,
              gamma, lambda_k, k, Tensor, imgs=None, aimGen=None, aux_test_imgs=None):
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_loss_real = 0
        if imgs != None:
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            d_real = discriminator(real_imgs)
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_fake = discriminator(gen_imgs.detach())
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))

        # 黑盒攻击时使用攻击目标生成器生成图片并加以真标签
        # 使用训练集辅助信息加以假标签
        if aimGen != None:
            aimgen_img = aimGen(z)
            d_real = discriminator(aimgen_img)
            d_loss_real += torch.mean(torch.abs(d_real - aimgen_img))
            if aux_test_imgs != None:
                aux_fake_img = Variable(aux_test_imgs.type(Tensor))
                d_fake = discriminator(aux_fake_img.detach())
                d_loss_fake += torch.mean(torch.abs(d_fake - aux_fake_img))

        d_loss = d_loss_real - k * d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ----------------
        # Update weights
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).data
        return d_loss.item(), g_loss.item(), M, k

    # def weights_init_normal(self,m):
    #     classname = m.__class__.__name__
    #     if classname.find("Conv") != -1:
    #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     elif classname.find("BatchNorm2d") != -1:
    #         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    class Generator(nn.Module):
        def __init__(self,opt):
            super(began.Generator, self).__init__()

            self.init_size = opt.img_size // 4
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            out = self.l1(noise)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self,opt):
            super(began.Discriminator, self).__init__()

            # Upsampling
            self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
            # Fully-connected layers
            self.down_size = opt.img_size // 2
            down_dim = 64 * (opt.img_size // 2) ** 2
            self.fc = nn.Sequential(
                nn.Linear(down_dim, 32),
                nn.BatchNorm1d(32, 0.8),
                nn.ReLU(inplace=True),
                nn.Linear(32, down_dim),
                nn.BatchNorm1d(down_dim),
                nn.ReLU(inplace=True),
            )
            # Upsampling
            self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

        def forward(self, img):
            out = self.down(img)
            out = self.fc(out.view(out.size(0), -1))
            out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
            return out

    def epoch_save(self, epoch, modPath, imgPath, generator, discriminator, fake_noise):
        torch.save(generator, '%s/netG_epoch_%06d.pth' % (modPath, epoch))
        torch.save(discriminator, '%s/netD_epoch_%06d.pth' % (modPath, epoch))
        fake = generator(fake_noise)
        vutils.save_image(fake.data, '%s/fake_%06d.png' % imgPath, nrow=8, normalize=True)

class dcgan():
    def train(self, opt, modPath, imgPath, dataloader=None, aimGen=None, aux_testloader=None):
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        fake_noise = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Initialize generator and discriminator
        generator = self.Generator(opt)
        discriminator = self.Discriminator(opt)

        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        if opt.cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        Batch = 0
        if dataloader != None:
            Batch += len(dataloader)
        if aux_testloader != None:
            Batch += len(aux_testloader)

        for epoch in range(opt.n_epochs):
            i = 0
            if dataloader != None:
                for i, (imgs, _) in enumerate(dataloader):
                    log = self.epoch(generator, discriminator, optimizer_G, optimizer_D,
                                      adversarial_loss, opt, Tensor,
                                      imgs=imgs, aimGen=aimGen, aux_test_imgs=None)
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, Batch, log[0], log[1])
                    )

            if aux_testloader != None:
                for j, (fake_imgs, _) in enumerate(aux_testloader):
                    log = self.epoch(generator, discriminator, optimizer_G, optimizer_D,
                                      adversarial_loss, opt, Tensor,
                                      imgs=None, aimGen=aimGen, aux_test_imgs=fake_imgs)
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i + j, Batch, log[0], log[1])
                    )

            elif dataloader == None and aux_testloader == None:
                log = self.epoch(generator, discriminator, optimizer_G, optimizer_D,
                                  adversarial_loss, opt, Tensor,
                                  imgs=None, aimGen=aimGen, aux_test_imgs=None)
                print(
                    "[Step %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, log[0], log[1])
                )
            self.epoch_save(epoch, modPath, imgPath, generator, discriminator, fake_noise)
        # final_save(modPath, generator, discriminator)

    def epoch(self, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss,
              opt, Tensor, imgs=None, aimGen=None, aux_test_imgs=None):
        # Adversarial ground truths
        valid = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = 0
        if imgs != None:
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        # 黑盒攻击时使用攻击目标生成器生成图片并加以真标签
        # 使用训练集辅助信息加以假标签
        if aimGen != None:
            aimgen_img = aimGen(z)
            real_loss += adversarial_loss(discriminator(aimgen_img), valid)
            if aux_test_imgs != None:
                aux_test_imgs = Variable(aux_test_imgs.type(Tensor))
                fake_loss += adversarial_loss(discriminator(aux_test_imgs), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        return d_loss.item(), g_loss.item()

    # def weights_init_normal(m):
    #     classname = m.__class__.__name__
    #     if classname.find("Conv") != -1:
    #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     elif classname.find("BatchNorm2d") != -1:
    #         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    class Generator(nn.Module):
        def __init__(self,opt):
            super(dcgan.Generator, self).__init__()

            self.init_size = opt.img_size // 4
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self,opt):
            super(dcgan.Discriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

            return validity

    def epoch_save(self, epoch, modPath, imgPath, generator, discriminator, fake_noise):
        torch.save(generator, '%s/netG_epoch_%06d.pth' % (modPath, epoch))
        torch.save(discriminator, '%s/netD_epoch_%06d.pth' % (modPath, epoch))
        fake = generator(fake_noise)
        vutils.save_image(fake.data, '%s/fake_%06d.png' % imgPath, nrow=8, normalize=True)

# acgan began dcgan
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def final_save(modPath, generator, discriminator):
    torch.save(generator, '%s/netG_final.pth' % (modPath))
    torch.save(discriminator, '%s/netD_final.pth' % (modPath))