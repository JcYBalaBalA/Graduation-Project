import torch,os
import train.models as mod
import attack.white_box as wb

# 黑盒攻击
# 可了解信息：
# 1.何种数据集
# 2.训练集大小
# 3.可以访问攻击对象的生成器

# 输入：
# GenPath：可访问到的生成模型路径 string
# aux_trainloader：作为辅助信息的train集 dataloader default = None
# aux_testloader：作为辅助信息的test集 dataloader default = None
# modName：需要使用何种模型攻击 string default = "dcgan"
# attack_modPath：训练出的攻击模型保存路径 string
def attack(attack_modName, opt_list, savepath, dataloader, GenPath,
           aux_trainloader=None, aux_testloader=None, cuda=False):
    modPath = savepath + "/models"
    imgPath = savepath + "/images"

    os.makedirs(savepath, exist_ok=True)
    os.makedirs(savepath + "/images", exist_ok=True)
    os.makedirs(savepath + "/models", exist_ok=True)

    opt = mod.model_opt(opt_list)

    aimGen = torch.load(GenPath)
    if cuda:
        aimGen.cuda()

    if attack_modName == "began":
        began = mod.began()
        began.train(opt, modPath, imgPath, dataloader=aux_trainloader,
                    aimGen=aimGen, aux_testloader=aux_testloader)
    elif attack_modName == "dcgan":
        dcgan = mod.dcgan()
        dcgan.train(opt, modPath, imgPath, dataloader=aux_trainloader,
                    aimGen=aimGen, aux_testloader=aux_testloader)

    # 用黑盒攻击生成的每一个分类器模型进行成员推理检测
    rs = {}
    D_list = os.listdir(modPath)
    for D in D_list:
        if D.startswith("netD"):
            rs[int(D.split("_")[-1])] = wb.attack(dataloader, len(dataloader), modPath + "\\" + D)
    return rs

# 实验用
def attack_show(attack_modName, opt_list, savepath, GenPath, trainloader, testloader,
           aux_trainloader=None, aux_testloader=None, cuda=False):
    modPath = savepath + "/models"
    imgPath = savepath + "/images"

    os.makedirs(savepath, exist_ok=True)
    os.makedirs(savepath + "/images", exist_ok=True)
    os.makedirs(savepath + "/models", exist_ok=True)

    opt = mod.model_opt(opt_list)
    print(attack_modName + "_opt:")
    opt.list_all_member()

    aimGen = torch.load(GenPath)
    if cuda:
        aimGen.cuda()

    if attack_modName == "began":
        began = mod.began()
        began.train(opt, modPath, imgPath, dataloader=aux_trainloader,
                    aimGen=aimGen, aux_testloader=aux_testloader)
    elif attack_modName == "dcgan":
        dcgan = mod.dcgan()
        dcgan.train(opt, modPath, imgPath, dataloader=aux_trainloader,
                    aimGen=aimGen, aux_testloader=aux_testloader)

    # 用黑盒攻击生成的每一个分类器模型进行成员推理检测
    rs = {}
    D_list = os.listdir(modPath)
    for D in D_list:
        if D.startswith("netD"):
            rs[int(D.split("_")[-1])] = wb.attack_show(trainloader, testloader, len(trainloader), modPath + "\\" + D)
    return rs
