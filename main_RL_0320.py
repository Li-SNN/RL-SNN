# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh
"""
import os
import sys
import time
from data.Dataset import getTrentoData,getHouston2018Data,getAugsburgData,getHoustonData,getBerlinData
# from dataload_Houston2013 import train_loader,test_loader

from utils.auxiliary import get_logger
from utils.hyper_pytorch import *
from datetime import datetime

import torch
import torch.nn.parallel
import warnings
warnings.filterwarnings('ignore')
from utils.start_0305 import test, train, output_metric, AvgrageMeter, accuracy

from models.encode_RL_0305_Res import ComplexNet as SNN
# from models.encode_RL_0305_Res_noguid import ComplexNet as SNN

np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径

seed = 1014
epochs = 50

# learn_rate = 0.0085
learn_rate = 0.0005
momentum = 0.9
# weight_decay = 0.0001
# class_number = 22
iter = 1

train_dataset = "Augsburg"
# train_dataset = "Berlin"
# train_dataset = "Houston2013"
# train_dataset = "Houston2018"

# Dataset.set_random_seed(0)
if train_dataset == "Houston2013":
    image_h = 349
    image_w = 1905
    train_loader, test_loader, trntst_loader, all_loader = getHoustonData(
        hsi_path="data/Houston2013/houston_hsi.mat",
        lidar_path="data/Houston2013/houston_lidar.mat",
        gt_path="data/Houston2013/houston_gt.mat",
        index_path="data/Houston2013/houston_index.mat",
        channels=20,
        windowSize=11,
        batch_size=64,
        num_workers=0)
elif train_dataset == "Trento":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = getTrentoData(
        hsi_path="data/Trento/trento_hsi.mat",
        lidar_path="data/Trento/trento_lidar.mat",
        gt_path="data/Trento/trento_gt.mat",
        index_path="data/Trento/trento_index.mat",
        channels=63,
        windowSize=11,
        batch_size=64,
        num_workers=0)
elif train_dataset == "Augsburg":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = getAugsburgData(
        hsi_path="data/Augsburg/augsburg_hsi.mat",
        lidar_path="data/Augsburg/augsburg_sar.mat",
        gt_path="data/Augsburg/augsburg_gt.mat",
        index_path="data/Augsburg/augsburg_index.mat",
        channels=40,
        windowSize=11,
        batch_size=64,
        num_workers=0)
elif train_dataset == "Berlin":
    image_h = 1723
    image_w = 476
    train_loader, test_loader, trntst_loader, all_loader = getBerlinData(
        hsi_path="data/Berlin/berlin_hsi.mat",
        lidar_path="data/Berlin/berlin_sar.mat",
        gt_path="data/Berlin/berlin_gt.mat",
        index_path="data/Berlin/berlin_index.mat",
        channels=40,
        windowSize=15,
        batch_size=64,
        num_workers=0)
else:
    image_h = 1202
    image_w = 4768
    train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(
        hsi_path="data/Houston2018/houston_hsi.mat",
        lidar_path="data/Houston2018/houston_lidar.mat",
        gt_path="data/Houston2018/houston_gt.mat",
        index_path="data/Houston2018/houston_index.mat",
        channels=40,
        windowSize=13,
        batch_size=64,
        num_workers=0)
def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    oa_list = []
    aa_list = []
    kappa_list = []
    each_acc_list = []
    train_time_list = []
    test_time_list = []

    torch.cuda.empty_cache()
    group_log_dir = os.path.join(log_dir, "Experiment_")  # logs组目录
    if not os.path.exists(group_log_dir):
        os.makedirs(group_log_dir)
    group_logger = get_logger(str(iter + 1), group_log_dir)
    random_state = seed + iter
    print('-------------------------------------------Iter %s----------------------------------' % (iter + 1))
    start(group_log_dir, logger=group_logger)

def start(group_log_dir, logger):
    print('进入main.py 中的start方法！')
    use_cuda = True
    # model = SNN(10,input_dim=40,num_classes=8) #SA最好精度在40个步长
    # model = SNN(2,input_dim=40,num_classes=8) #SA最好精度在40个步长
    # model = SNN(5,input_dim=40,num_classes=11) #SA最好精度在40个步长
    model = SNN(5,input_dim=40,num_classes=7) #SA最好精度在40个步长
    # model = SNN(5,input_dim=40,num_classes=8) #SA最好精度在40个步长
    # model = SNN(10,input_dim=40,num_classes=20) #SA最好精度在40个步长
    # model = DCMNet()
    # model = SpikingUGRF(15,5)

    print(model)
    model =model.cuda()

    # 定义损失函数和优化器
    # optimizer = torch.optim.SGD(model.parameters(), learn_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    optimizer = torch.optim.AdamW([
        {'params': model.branch_spec.parameters(), 'lr': 0.0085},
        {'params': model.branch_spat.parameters(), 'lr': 0.0085},
        {'params': model.branch_lidar.parameters(), 'lr': 0.0085},
        {'params': model.aux_spec.parameters(), 'lr': 0.0085},
        {'params': model.aux_spat.parameters(), 'lr': 0.0085},
        {'params': model.aux_lidar.parameters(), 'lr': 0.0085},
        {'params': model.classifier.parameters(), 'lr': 0.0085},
        # RL 策略网络：学习率降低 5-10 倍，只允许它缓慢微调
        {'params': model.policy.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}
        # {'params': model.policy.parameters(), 'lr': 0.0085, 'weight_decay': 1e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

    # 定义损失函数：注意这里主分类器用 reduction='none' 来获取逐样本的 reward
    # criterion_none = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    # criterion_mean = torch.nn.CrossEntropyLoss().cuda()
    # criterion = torch.nn.CrossEntropyLoss().cuda()

    # 加入 label_smoothing 防止 SNN 早期过度自信导致 RL 没东西学
    criterion_none = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1).cuda()
    # criterion_none = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1).cuda()
    # criterion_mean = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).cuda()
    criterion_mean = torch.nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

    best_oa = -1
    best_aa = -1
    best_kappa = -1
    best_each_acc = -1
    best_acc = -1
    # 定义两个数组,记录训练损失和验证损失
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    losses = AvgrageMeter()
    top1 = AvgrageMeter()
    global_moving_baseline = None
    train_start_time = time.time()  # 返回当前的时间戳
    boundary = 0.1
    # 定义两阶段的转折点
    STAGE_2_EPOCH = 40

    for epoch in range(epochs):
        model.train()
        losses.reset()  # 每一轮开始前重置计数器
        top1.reset()
        # 初始化记录三个分支权重的计量器
        w_spec_meter = AvgrageMeter()
        w_spat_meter = AvgrageMeter()
        w_lida_meter = AvgrageMeter()
        # for batch_idx, (data, labels) in enumerate(train_loader):
        for batch_idx, (hsi,lidar, labels) in enumerate(train_loader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            labels = labels.to(device)

            # data = data.to(device)
            # data = np.transpose(data, (0, 2, 3, 1))
            # hsi = data[..., 0:40]
            # hsi = np.transpose(hsi, (0, 3, 1, 2))
            # lidar = data[..., 40:]
            # lidar = np.transpose(lidar, (0, 3, 1, 2))
            # labels = labels.to(device)

            outputs, log_prob, aux_logits, entropy,w = model(hsi, lidar,boundary)
            # 计算并记录当前 batch 的三个分支平均权重
            batch_w = w.mean(dim=0).detach().cpu().numpy()
            w_spec_meter.update(batch_w[0], hsi.size(0))
            w_spat_meter.update(batch_w[1], hsi.size(0))
            w_lida_meter.update(batch_w[2], hsi.size(0))
            # 计算标准分类 Loss (主分支)
            ce_loss_batch = criterion_none(outputs, labels)  # [B]
            ce_loss = ce_loss_batch.mean()

            # 计算辅助分支 Loss (作为 RL 的基准)
            l_spec_b = criterion_none(aux_logits[0], labels)
            l_spat_b = criterion_none(aux_logits[1], labels)
            l_lida_b = criterion_none(aux_logits[2], labels)
            # 三个单模态分支的平均 Loss
            running_baseline = (l_spec_b + l_spat_b + l_lida_b) / 3.0

            with torch.no_grad():
                diff = running_baseline - ce_loss_batch
                # 使用归一化的 Advantage
                reward = (diff - diff.mean()) / (diff.std() + 1e-6)
                reward = torch.clamp(reward, -1.0, 1.0).detach()  # 强制截断在 [-1, 1]

            # 计算策略梯度
            entropy_weight = max(0.05 * (1 - epoch / 100), 0.005)
            # policy_loss = -(log_prob * reward).mean() - entropy_weight * entropy.mean()
            policy_loss = -(log_prob * reward).mean()
            # 前 20 轮只训练特征提取，20 轮后开启 RL 融合
            # if epoch<10:
            #     rl_weight=0.0
            # else:
            #     rl_weight = 0.3
            rl_weight = 0.1
            total_loss = ce_loss + 0.2 * (
                    l_spec_b.mean() + l_spat_b.mean() + l_lida_b.mean()) / 3.0 + rl_weight * policy_loss - entropy_weight * entropy.mean()

            optimizer.zero_grad()
            total_loss.backward()

            # 针对 SNN 和 RL 的混合训练，max_norm 建议设为 0.5 到 1.0 之间
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()


            prec1, _, _ = accuracy(outputs.data, labels.data, topk=(1,))
            losses.update(total_loss.item(), hsi.size(0))
            top1.update(prec1[0].item(), hsi.size(0))

            # 使用计数器的平均值进行日志记录和调度 ---
        current_train_loss = losses.avg
        current_train_acc = top1.avg

        valid_loss, valid_acc , test_acc1, test_obj, tar_v, pre_v= test(test_loader, model, criterion_mean, epoch, use_cuda,boundary)

        # from models.encode_RL_0305_Res import calculate_and_reset_firing_rate
        # overall_firing_rate, layer_firing_rates = calculate_and_reset_firing_rate(model)

        logger.info(
            'Epoch: %03d | Stage: %d | Train Loss: %.4f Acc: %.4f | Valid Loss: %.4f Acc: %.4f | W_Spec: %.3f W_Spat: %.3f W_Lidar: %.3f' % (
                epoch, 1 if epoch < STAGE_2_EPOCH else 2, current_train_loss, current_train_acc, valid_loss, valid_acc,
                w_spec_meter.avg, w_spat_meter.avg, w_lida_meter.avg))
        # for layer_name, rate in layer_firing_rates.items():
        #     logger.debug(f'Layer {layer_name} firing rate: {rate:.4f}')

        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)

        # scheduler.step(current_train_loss)
        scheduler.step(valid_loss) #使用validloss精度更好
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        # save model
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth_Trento.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100

    logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
    logger.info('best_CA: %s \n', best_each_acc)

    # train_end_time = time.time()
    # checkpoint = torch.load(group_log_dir + "/best_model.pth_Trento.tar")
    # best_acc = checkpoint['best_acc']
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    #
    # # 测试
    # test_start_time = time.time()
    # test_loss, test_acc, test_acc1, test_obj, tar_v, pre_v = test(test_loader, model, criterion, epoch, use_cuda)
    # OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    # print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
    # logger.info('AA: %f, OA: %f, kappa: %f\n '% (OA_TE * 100, AA_TE * 100, Kappa_TE))
    # test_end_time = time.time()
    # logger.info("Final:   Loss: %s  Accuracy: %s", test_loss, test_acc)
    #
    # train_time = train_end_time - train_start_time
    # test_time = test_end_time - test_start_time
    # # logger.debug('classification:\n %s\n confusion:\n%s\n ' % (classification, confusion))
    # logger.info("Train time:%s , Test time:%s", train_time, test_time)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()