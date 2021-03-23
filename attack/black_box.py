
import torch
import train.models as mod
# 黑盒攻击
# 可了解信息：
# 1.何种数据集
# 2.训练集大小
# 3.可以访问攻击对象的生成器

# 输入：
# GenPath：可访问到的生成模型路径 string
# aux_trainloader：作为辅助信息的train dataloader dataloader default = None
# aux_testloader：作为辅助信息的test dataloader dataloader default = None
# modName：需要使用何种模型攻击 string default = "dcgan"
# attack_modPath：训练出的攻击模型保存路径 string
def attack(modName, opt, modPath, imgPath, dataloader=None, aimGen=None,
           aux_trainloader=None, aux_testloader=None):
