import torch

# 白盒攻击
# 可了解信息：
# 1.何种数据集
# 2.训练集大小
# 3.可以访问攻击对象的分类器
# 预留两种接口
# 1.传入disPath，路径所代表的是通过torch.save(net,PATH)保存的模型
# 2.disState，所代表的是通过torch.save(net.state_dict,PATH)保存的参数
# 而disMod代表具有某些参数的目标模型，通过
# load_state_dict载入这些参数
# 3.传入disMod，直接传入所要攻击模型的路径

# 传入的trainloader及testloader仅代表获取数据时默认的训练集和测试集,若只需要一个输入可将testloader置为None
# return 长度为trainsize的样本数组
def attack(dataloader, trainsize, disMod, cuda=False):
    disMod = torch.load(disMod)

    if cuda:
        disMod.cuda()

    wb_predictions = []
    for i, (img, _) in enumerate(dataloader, 0):
        output = disMod(img)
        output = [x for x in output.detach().cpu().numpy()]
        output = list(zip(output, [img[i] for i in range(len(output))]))
        wb_predictions.extend(output)

    return [x[1] for x in sorted(wb_predictions, reverse=True)[:len(trainsize)]]

# 实验用
# return 白盒攻击预测准确度
def attack_show(trainloader, testloader, trainsize, disMod, cuda=False):
    disMod = torch.load(disMod)

    if cuda:
        disMod.cuda()

    wb_predictions = []
    for i, (img, _) in enumerate(trainloader, 0):
        output = disMod(img)
        output = [x for x in output.detach().cpu().numpy()]
        output = list(zip(output, ['train' for _ in range(len(output))]))
        wb_predictions.extend(output)

    for i, (img, _) in enumerate(testloader, 0):
        output = disMod(img)
        output = [x for x in output.detach().cpu().numpy()]
        output = list(zip(output, ['test' for _ in range(len(output))]))
        wb_predictions.extend(output)

    wb_predictions = [x[1] for x in sorted(wb_predictions, reverse=True)[:len(trainsize)]]
    return wb_predictions.count('train')/float(len(trainsize))
