import os,torch
import models as mod
import getdata as gd


def train_interface(path="save", modle="began",dataset="MNIST"):
    modPath = path + "/models"
    imgPath = path + "/images"

    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/images", exist_ok=True)
    os.makedirs(path + "/models", exist_ok=True)

    # n_epochs batch_size lr b1 b2 n_cpu latent_dim img_size channels sample_interval n_classes(仅acgan有)
    opts = {
        "acgan" : [200,64,0.0002,0.5,0.999,8,100,32,1,400,10],
        "began" : [200,64,0.0002,0.5,0.999,8,62,32,1,400],
        "dcgan" : [200,64,0.0002,0.5,0.999,8,100,32,1,400]
    }
    opt = mod.model_opt(opts[modle])
    print(modle + "_opt:")
    opt.list_all_member()

    opt.cuda = opt.cuda if torch.cuda.is_available() else False

    trainloader, testloader = gd.getData(dataset,opt.batch_size,gd.channel_1_transform(opt.img_size))
    if modle == "acgan":
        acgan = mod.acgan()
        acgan.train(opt, trainloader, modPath, imgPath)
    elif modle == "began":
        began = mod.began()
        began.train(opt, modPath, imgPath, dataloader=trainloader)
    elif modle == "dcgan":
        dcgan = mod.dcgan()
        dcgan.train(opt, modPath, imgPath, dataloader=trainloader)

if __name__ == "__main__":
    train_interface()