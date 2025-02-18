import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric_duapre import compress_wiki, compress, calculate_map, calculate_top_map, p_topK, ContrastiveLoss, compress_nus, calculate_top_map_nus
# import datasetspre as datasets
import settingsnuspre as settings
from models3pre3 import ImgNetHY, TxtNetHY, DeTxtNet, DeImgNet, GenHash, GetTxtNet, GetImgNet, FuseTransEncoder, GetITNet
from load_data import get_loader_flickr, get_loader_nus, get_loader_coco, get_loader_flickr_CLIP, get_loader_flickr_fea, get_loader_nuswide_CLIP, get_loader_coco_CLIP
import os.path as osp
import sys
from tools import build_G_from_S, generate_robust_S
from RL_brain2 import SarsaLambdaTable
import utils_ucch as utilsuc
import scipy
import scipy.spatial

import pandas as pd
from RL_brainQ import QLearningTable
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(torch.exp(emb_i), dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(torch.exp(emb_j), dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2).cuda() #+ scipy.spatial.distance.cdist(representations.unsqueeze(1).detach().numpy(), representations.unsqueeze(0).detach().numpy(), 'hamming').cuda()    # simi_mat: (2*bs, 2*bs)

        # similarity_matrix = torch.where(similarity_matrix < 0.3, 0, 1)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
class Session:
    def __init__(self, train_loader, test_loader, database_loader, train_dataset, test_dataset, database_dataset, data_train, a, a2, a3):

        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(settings.GPU_ID)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.database_dataset = database_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.database_loader = database_loader
        self.I_tr, self.T_tr, self.L_tr = data_train

        # txt_feat_len = datasets.txt_feat_len
        txt_feat_len = self.T_tr.shape[1]
        img_feat_len = self.I_tr.shape[1]
        self.ContrastiveLoss = ContrastiveLoss(batch_size=settings.BATCH_SIZE, device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"))

        self.CodeNet_I = ImgNetHY(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.FeatNet_I = ImgNetHY(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.CodeNet_T = TxtNetHY(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)


        self.GetNet_I = GetImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.GetNet_T = GetTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.GetNet = GetITNet(code_len=settings.CODE_LEN, img_feat_len=5000)

        self.DeCodeNet_I = DeImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.DeCodeNet_T = DeTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.genHash = GenHash(code_len=settings.CODE_LEN, txt_feat_len=settings.BATCH_SIZE)
        num_layers, nhead = 2,  4

        self.FuseTrans = FuseTransEncoder(num_layers, 1024, nhead, settings.CODE_LEN).cuda()

        # self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG},
                                          {'params': self.CodeNet_I.alexnet.classifier.parameters(),
                                           'lr': settings.LR_IMG}],
                                         momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
            self.opt_DeI = torch.optim.SGD([{'params': self.DeCodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG}],
                                           momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE"or settings.DATASET == "MSCOCO":
            # self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=settings.LR_IMG, betas=(0.5, 0.999))
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
            self.opt_GI = torch.optim.Adam(self.GetNet_I.parameters(), lr=settings.LR_IMG, betas=(0.5, 0.999))
            self.opt_DeI = torch.optim.Adam(self.DeCodeNet_I.parameters(), lr=settings.LR_IMGTXT, betas=(0.5, 0.999))

        self.optimizer_FuseTrans = torch.optim.Adam(self.FuseTrans.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        self.opt_GT = torch.optim.Adam(self.GetNet_T.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))
        self.gen_H = torch.optim.Adam(self.genHash.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))
        self.opt_DeT = torch.optim.Adam(self.DeCodeNet_T.parameters(), lr=settings.LR_IMGTXT, betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.GetNet.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))

        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        self.img_norm = img_norm
        self.txt_norm = txt_norm
        self.cS = img_norm.mm(txt_norm.t()).cuda()


        f_norm = torch.concat((img_norm, txt_norm), dim=1).cuda()
        self.gsI = np.corrcoef(img_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gsI = (torch.tensor(self.gsI, dtype=torch.float32))
        self.gsT = np.corrcoef(txt_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gsT = (torch.tensor(self.gsT, dtype=torch.float32))
        self.gs = np.corrcoef(f_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gs = (torch.tensor(self.gs, dtype=torch.float32))
        self.gsid = torch.eye(4992)
        self.ContrastiveLoss = ContrastiveLoss(batch_size=settings.BATCH_SIZE, device=self.device)

    def train(self, epoch, l1, l2, l3, l4, l5, l6, l7):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        self.FuseTrans.cuda().train()

        self.CodeNet_I.set_alpha(1)
        self.CodeNet_T.set_alpha(1)
        self.GetNet.set_alpha(1)



        for No, (F_I, F_T, _, index_) in enumerate(self.train_loader): #No, (img, txt, _, index_)
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            F_I = Variable(torch.FloatTensor(F_I.numpy()).cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.optimizer_FuseTrans.zero_grad()

            S = self.gs[index_, :][:, index_].cuda()
            alpha1 = l6
            alpha2 = l7
            alpha3 = l5
            S1 = alpha2*self.gsI[index_, :][:, index_].cuda() + (1-alpha2)*self.gsT[index_, :][:, index_].cuda()
            S1 = torch.sign((S1 - alpha3)) * 0.5 + 0.5
            S = alpha1*(alpha2*self.gsI[index_, :][:, index_].cuda() + (1-alpha2)*self.gsT[index_, :][:, index_].cuda())+ (1-alpha1)*F.normalize(S).mm(F.normalize(S.t()))*S1
            S = S * 2 - 1

            code_I = self.CodeNet_I(F_I.cuda())
            code_T = self.CodeNet_T(F_T.cuda())


            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            GS = self.gs[index_, :] * (1 - self.gsid[index_, :])

            max_indices = np.argmax(GS, axis=1)


            temp_tokens = torch.concat((self.img_norm[max_indices], self.txt_norm[max_indices]), dim=1).cuda().unsqueeze(0)
            hashB, hashH, img, txt = self.FuseTrans(temp_tokens)

            B_H = F.normalize(hashH)

            BI_BI = B_I.mm(B_I.t()).cuda()
            BT_BT = B_T.mm(B_T.t()).cuda()
            BI_BT = B_I.mm(B_T.t()).cuda()
            B_B = B_H.mm(B_H.t()).cuda()


            loss1 = F.mse_loss(BI_BI.cuda(), S.cuda()) +F.mse_loss(BI_BT.cuda(), S.cuda()) + F.mse_loss(BT_BT.cuda(), S.cuda()) + 1 * F.mse_loss(B_B.cuda(), S.cuda())
            loss8 = 1*self.ContrastiveLoss(B_I, B_T) + 1*self.ContrastiveLoss(B_H, B_T) + 1*self.ContrastiveLoss(B_I, B_H)
            loss3 = F.mse_loss(code_I.cuda(), hashB.cuda()) + F.mse_loss(code_T.cuda(), hashB.cuda())





            loss = l4 * loss8 + l1 * loss1 + l2 * loss3
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.optimizer_FuseTrans.step()



    def eval(self, l1, l2, l3, l4, l5, l6, l7):
        # self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T, self.database_dataset,
                                                                   self.test_dataset)

        if settings.DATASET == "MIRFlickr":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T,
                                                              self.database_dataset, self.test_dataset)
        if settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_nus(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T,
                                                              self.database_dataset, self.test_dataset)

        MAP_I2T5 = calculate_top_map_nus(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I5 = calculate_top_map_nus(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_I2T = 0#calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = 0#calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        self.logger.info('MAP: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (l1, l2, l3, l4, l5, l6, l7, MAP_I2T5, MAP_T2I5, MAP_I2T5+MAP_T2I5))
        return MAP_I2T5+MAP_T2I5

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

    def cal_similarity2(self, F_I, F_T):
        a1 = settings.BETA
        a2 = 0.6
        K = 3000
        batch_size = F_I.size(0)
        size = batch_size
        top_size = K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = a1 * S_I + (1 - a1) * S_T

        m, n1 = S1.sort()
        S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t())/2
        S = a2 * S1 + (1 - a2) * S2

        return S


    def crossview_contrastive_Loss(self, view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        # bn, k = view1.size()
        assert (view2.size(0) == _ and view2.size(1) == k)

        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise
        # p_i_j = compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).detach()
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).detach()

        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS

        loss = - p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))

        # loss = loss.sum()
        loss = torch.sum(loss)

        return loss


def main():

    if settings.DATASET == "MIRFlickr":
        dataloader, data_train = get_loader_flickr_CLIP(settings.BATCH_SIZE)
        # dataloader, data_train = get_loader_flickr_fea(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "NUSWIDE":
        dataloader, data_train, dataloader2 = get_loader_nuswide_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "MSCOCO":
        dataloader, data_train = get_loader_coco_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']

    train_loader = train_dataset

    test_loader = test_dataset

    database_loader = database_dataset
    n_features = 6
    n_actions = 3 ** n_features

    RL = QLearningTable(actions=list(range(n_actions)))
    observation = torch.ones(1, n_features)
    RL.epsilon = 0.9
    max_e = 500
    max_status = 1.8
    max_value = 0
    l5 = 1
    l6 = 0.9
    l2 = 1
    l3 = 1
    l4 = 1
    l7 = 0.6
    q_table = pd.DataFrame(columns=list(range(1)), dtype=np.float64)

    for i in range(1000000):
        step = 0
        observation = torch.ones(1, n_features)#*10**5
        observation[0, 3:] = 0.5

        e_step = 0
        a = 10 ** (-6)
        b = -0.1
        while True:


            action = RL.choose_action(str(observation))


            observation_ = observation.clone()
            action1 = action
            bits = 10
            for j in range(n_features-3):
                if action1 % 3 == 0:
                    if observation_[0, j] / bits > 0.00001:
                        observation_[0, j] = float(observation[0, j]) / bits
                    else:
                        observation_[0, j] = 0
            #
                elif action1 % 3 == 1 and observation_[0, j] * bits < 100000:
                    observation_[0, j] = observation[0, j] * bits
                if observation_[0, j] > 0.9:
                    observation_[0, j] = int(observation_[0, j])
                action1 = action1 // 3
            setpj = 3

            for j in range(3):
                 if action1 % 3 == 0:
                     if observation_[0, setpj+j] - 0.1 > -0.1:
                         observation_[0, setpj+j] = observation[0, setpj+j] - 0.1
                     else:
                         observation_[0, setpj+j] = 0

                 elif action1 % 3 == 1 and observation_[0, setpj+j] + 0.1 <= 1.1:
                     observation_[0, setpj+j] = observation[0, setpj+j] + 0.1
                 action1 = action1 // 3



            l1, l2, l3, l4, l5, l6, l7 = 1.000000, 100.000000, 1.000000, 0.100000, 0.700000, 0.300000, 0.400000
            l1, l2, l4, l5, l6, l7  = observation_[0, :]
            state = str(observation_)

            status = False

            chae = 0.01
            if state not in q_table.index:
            # if True:
                l1, l2, l3, l4, l5, l6, l7 = float(l1), float(l2), float(l3), float(l4), float(l5), float(l6), float(l7)
                sess = Session(train_loader, test_loader, database_loader, train_dataset,
                               test_dataset, database_dataset, data_train, l5, l5, l7)
                A = 0
                B = 0
                Bmax = 0
                currmax = 0
                for epoch in range(settings.NUM_EPOCH):
                    # train the Model
                    sess.train(epoch, l1, l2, l3, l4, l5, l6, l7)
                    # eval the Model
                    if ((epoch + 1) % settings.EVAL_INTERVAL == 0):
                        B = sess.eval(l1, l2, l3, l4, l5, l6, l7)
                        if max_value < B:
                            max_value = B
                        if B >= Bmax:
                            Bmax = B

                        # sess.save_checkpoints(step=epoch + 1)
                        if B > A:
                            A = B
                        if B < currmax:
                            break
                        else:
                            currmax = B
                        # if (B < 1.8):
                        #     break

                # append new state to q table
                q_table = q_table.append(
                    pd.Series(
                        Bmax,
                        index=q_table.columns,
                        name=state,
                    )
                )
            else:
                #step = step - 1
                A = 0
                B = q_table.loc[str(observation_), 0]
                # chae = 0.01
                if B > A:
                    A = B

                print('^^^^^^exit:%s,B:%.6f' % (str(observation_),  B))


            # A = sess.eval(l1, l2, l3, l4, l5, l6, l7) - 1.8
            # A = A - max_status  # 1.8
            # reward = A * 100
            A = A - max_status

            if A < 0:
                reward = -2 ** ((-A) * 20)
            else:
                reward = 2 ** (A * 20)
            if reward < -100:
                reward = -100

            print('%.6f,%.6f,%.6f,%.6f,max:%.6f,value:%.6f' % (action, reward, i, RL.epsilon, max_status, max_value))


            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            # action = action_
            e_step = e_step + 1

            if step > max_e:  # or A > 0.16:
                # print(observation)
                break
            step += 1
    


if __name__ == '__main__':
    main()
