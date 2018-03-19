import os, time, pickle, argparse, network, util, itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from util import DatasetProcessing
from torchvision import models
import CalcHammingRanking as CalcHR
import cv2
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='256*256',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--input_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--output_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--input_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--output_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nb', type=int, default=9, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop', type=bool, default=True, help='random crop True or False')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=30, help='train epochs num')
parser.add_argument('--decay_epoch', type=int, default=25, help='learning rate decay start epoch num')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--bit', type=int, default=64, help='bit')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrH', type=float, default=0.0002, help='learning rate, default=0.0005')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
parser.add_argument('--lambdaC', type=float, default=10, help='lambdaC for hash loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
opt = parser.parse_args()
print('------------ Options -------------')
for k, v in sorted(vars(opt).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results save path
root = opt.dataset + '_' + opt.save_root + '/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'test_results'):
    os.mkdir(root + 'test_results')
if not os.path.isdir(root + 'test_results/AtoB'):
    os.mkdir(root + 'test_results/AtoB')
if not os.path.isdir(root + 'test_results/BtoA'):
    os.mkdir(root + 'test_results/BtoA')

if not os.path.isdir(root + 'train_results'):
    os.mkdir(root + 'train_results')
if not os.path.isdir(root + 'train_results/AtoB'):
    os.mkdir(root + 'train_results/AtoB')
if not os.path.isdir(root + 'train_results/BtoA'):
    os.mkdir(root + 'train_results/BtoA')

DATA_DIR = './data/256x256'
DATABASE_FILE = 'img_base.txt'
TRAIN_FILE = 'train_img.txt'
TEST_FILE = 'test_img.txt'
nclasses = 125
DATABASE_LABEL = 'train_label.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'


def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
  
    target_onehot = torch.FloatTensor(target.size(0), nclasses)

    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S




# data_loader
transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


#dset_data = DatasetProcessing(DATA_DIR, DATABASE_FILE, DATABASE_LABEL, transformations)
dset_train = DatasetProcessing(DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)
dset_test = DatasetProcessing(DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

num_train, num_test =  len(dset_train), len(dset_test)

#data_loader = DataLoader(dset_train,batch_size=opt.batch_size,shuffle=False,num_workers=4)
test_loader = DataLoader(dset_test,batch_size=opt.batch_size,shuffle=False,num_workers=4)
train_loader = DataLoader(dset_train,batch_size=opt.batch_size,shuffle=True,num_workers=4)

### training phase
# parameters setting

# FOR HA
B_real_ = torch.randn(num_train, opt.bit)
B_fake_ = torch.randn(num_train, opt.bit)
H_I_ = torch.zeros(num_train, opt.bit)
H_S_ = torch.zeros(num_train, opt.bit)

B_I = torch.sign(torch.sign(B_real_))
B_S = torch.sign(torch.sign(B_fake_))

#B_I = np.load('./data/TU/B_128.npy')
#B_S = B_I
#B_I = np.sign(B_I + 1e-10)
#B_S = np.sign(B_S + 1e-10)
#B_I = torch.from_numpy(B_I)
#B_S = torch.from_numpy(B_S)

#train_labels = LoadLabel(TRAIN_LABEL, DATA_DIR)

#train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
train_labels = LoadLabel(DATABASE_LABEL,DATA_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
Y = train_labels_onehot

#Sim = CalcSim(train_labels_onehot, train_labels_onehot)
# T_S = np.load('64ex-T_S.npy')
# D_I = np.load('64ex-H_I.npy')
# map_1 = CalcHR.CalcMap(T_S, D_I,test_labels_onehot.numpy(), train_labels_onehot.numpy())
# print(map_1)
# network
G_A = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb,flag = False)
G_B = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb, flag = False)
D_A = network.discriminator(opt.input_ndc, opt.bit, opt.ndf)

net = models.resnet18(pretrained = False)
net.fc= nn.Linear(2048, opt.bit)

net_dict = net.state_dict()

pretrain_dict = torch.load('./ex_sketch_64HA_param.pkl')
pretrain_dict = {k[7:] : v for k, v in pretrain_dict.items() if  k[7:] in net_dict}
net_dict.update(pretrain_dict) 
net.load_state_dict(net_dict)   


H_A = net

D_B = network.discriminator(opt.input_ndc, opt.bit, opt.ndf)

G_A.weight_init(mean=0.0, std=0.02)
G_B.weight_init(mean=0.0, std=0.02)
D_A.weight_init(mean=0.0, std=0.02)
D_B.weight_init(mean=0.0, std=0.02)


G_A = nn.DataParallel(G_A)
G_B = nn.DataParallel(G_B)
D_A = nn.DataParallel(D_A)
D_B = nn.DataParallel(D_B)
H_A = nn.DataParallel(H_A)

G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()
H_A.cuda()

# G_A.train()
# G_B.train()
# D_A.train()
# D_B.train()
# H_A.train()

print('---------- Networks initialized -------------')
#util.print_network(G_A)
#util.print_network(G_B)
#util.print_network(D_A)
#util.print_network(D_B)
#util.print_network(H_A)
#util.print_network(H_B)
print('-----------------------------------------------')

# loss
BCE_loss = nn.BCELoss().cuda()
MSE_loss = nn.MSELoss().cuda()
L1_loss = nn.L1Loss().cuda()

# Adam optimizer
G_optimizer = optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_A_optimizer = optim.Adam(D_A.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
D_B_optimizer = optim.Adam(D_B.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
H_A_optimizer = optim.Adam(H_A.parameters(), lr=opt.lrH, betas=(opt.beta1, opt.beta2))

# image store
fakeA_store = util.ImagePool(50)
fakeB_store = util.ImagePool(50)

train_hist = {}
train_hist['D_A_losses'] = []
train_hist['D_B_losses'] = []
train_hist['G_A_losses'] = []
train_hist['G_B_losses'] = []
train_hist['A_cycle_losses'] = []
train_hist['B_cycle_losses'] = []
train_hist['H_A_Hash_losses'] = []
train_hist['H_B_Hash_losses'] = []
train_hist['A_Cons_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')


file = open('fg_log'+str(opt.bit)+'.log','a')
start_time = time.time()
map_ = 0.0


for epoch in range(opt.train_epoch):

    G_A.eval()
    
    H_A.eval()
    H_A_optimizer.param_groups[0]['lr']  = opt.lrH*(0.1**(epoch//5))
    D_B_S_time = time.time()
    # # D step
    print('start...')
    temp1 = 2*Y.t().mm(Y)+ torch.eye(nclasses)
    temp1 = temp1.inverse()
    temp1 = temp1.mm(Y.t())
    D = temp1.mm((B_I+B_S))
    D_ETIME = time.time()
    print('[D STEP] TIME: %3.5f' %(D_ETIME-D_B_S_time))
    # #B step SKETCH2IMAGE

    # for iteration, batch in enumerate(train_loader, 0):
    #     realA = batch[0]
    #     realB = batch[1]
    #     train_label = batch[2]
    #     batch_ind = batch[3]
        
    #     #if opt.resize_scale:
    #     #    realA = util.imgs_resize(realA, opt.input_size)
    #     #    realB = util.imgs_resize(realB, opt.input_size)
    #     # print(realA.shape)
    #     realA = Variable(realA.cuda(), volatile=True)
    #     realB = Variable(realB.cuda(), volatile=True)

    #     fakeB,mask = G_A(realA)
        
    #     H_A_real = H_A(realB)
    #     H_A_fake = H_A(fakeB)
    #     for i, ind in enumerate(batch_ind):
    #         H_I_[ind,:] = H_A_real.data[i]
    #         H_S_[ind,:] = H_A_fake.data[i]
    B_I = torch.sign(Y.mm(D) + 1e-5 * H_I_)
    B_S = torch.sign(Y.mm(D) + 1e-5 * H_S_)
      
    D_B_E_time = time.time()
    print('[D STEP AND B STEP] TIME: %3.5f' %(D_B_E_time-D_B_S_time))

    G_A.train()
    G_B.train()
    D_A.train()
    D_B.train()
    H_A.train()
# F step
    H_A_Hash_losses = []
    H_B_Hash_losses = []
    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    A_cycle_losses = []
    B_cycle_losses = []
    B_Cons_losses  = []
    A_Cons_losses  = []
    epoch_start_time = time.time()
    num_iter = 0
    if (epoch+1) > opt.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= opt.lrG / (opt.train_epoch - opt.decay_epoch)
    
    if (epoch + 1) > 20 and (epoch + 1) < 25:
        H_A_optimizer.param_groups[0]['lr']  = opt.lrH*0.1
    #elif  (epoch + 1) > 25:
    #    H_A_optimizer.param_groups[0]['lr']  = opt.lrH*0.01

    start_batch_time = time.time()
    for iteration, batch in enumerate(train_loader, 0):
        realA = batch[0]
        realB = batch[1]
        train_label = batch[2]
        batch_ind = batch[3]
        train_label = torch.squeeze(train_label)
        if opt.resize_scale:
            realA = util.imgs_resize(realA, opt.resize_scale)
            realB = util.imgs_resize(realB, opt.resize_scale)

        if opt.crop:
            realA = util.random_crop(realA, opt.input_size)
            realB = util.random_crop(realB, opt.input_size)

        if opt.fliplr:
            realA = util.random_fliplr(realA)
            realB = util.random_fliplr(realB)

        realA, realB = Variable(realA.cuda()), Variable(realB.cuda())

        # train generator G
        G_optimizer.zero_grad()

        # generate real A to fake B; D_A(G_A(A))
        fakeB, maskA = G_A(realA)
        D_A_result = D_A(fakeB)
        G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))
        # mask = maskA.cpu().data.numpy()
        # mask = np.squeeze(mask)
        # mask = mask.transpose(1,2,0)
        # mask = cv2.resize(mask,(256,256))
        # mask = mask.transpose(2,0,1)
        # mask = np.expand_dims(mask,1)
        # mask = torch.Tensor(mask)
        # maskA = Variable(mask.cuda())        
        
 
        # reconstruct fake B to rec A; G_B(G_A(A))
        recA,_ = G_B(fakeB)
        A_cycle_loss = L1_loss(recA, realA) * opt.lambdaA 

        # generate real B to fake A; D_A(G_B(B))
        fakeA,maskB = G_B(realB)
        D_B_result = D_B(fakeA)
        G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))

        # reconstruct fake A to rec B G_A(G_B(B))
        recB,_ = G_A(fakeA)
        B_cycle_loss = L1_loss(recB, realB) * opt.lambdaB 

        G_loss = G_A_loss + A_cycle_loss + G_B_loss + B_cycle_loss
        G_loss.backward()
        G_optimizer.step()

        train_hist['G_A_losses'].append(G_A_loss.data[0])
        train_hist['G_B_losses'].append(G_B_loss.data[0])
        train_hist['A_cycle_losses'].append(A_cycle_loss.data[0])
        train_hist['B_cycle_losses'].append(B_cycle_loss.data[0])
        
        
        G_A_losses.append(G_A_loss.data[0])
        G_B_losses.append(G_B_loss.data[0])
        A_cycle_losses.append(A_cycle_loss.data[0])
        B_cycle_losses.append(B_cycle_loss.data[0])



        # train discriminator D_A
        D_A_optimizer.zero_grad()

        D_A_real = D_A(realB)
        
        #realD_A = kronecker_product(D_A_real, D_A_real)
        realD_A  = D_A_real

        D_A_real_loss = MSE_loss(realD_A, Variable(torch.ones(realD_A.size()).cuda()))
        
        fakeB = fakeB_store.query(fakeB)
        #fakeB,_ = G_A(realA)
        D_A_fake = D_A(fakeB)
        #print(D_A_fake) 
        #fake_D_A = kronecker_product(D_A_real, D_A_fake)
        fake_D_A = D_A_fake 

        D_A_fake_loss = MSE_loss(fake_D_A, Variable(torch.zeros(fake_D_A.size()).cuda()))
        

        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5 

        D_A_loss.backward()
        D_A_optimizer.step()
  
        train_hist['D_A_losses'].append(D_A_loss.data[0])
        D_A_losses.append(D_A_loss.data[0])
        
        # train discriminator D_B
        D_B_optimizer.zero_grad()

        D_B_real = D_B(realA)
        D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))
     
  
        fakeA = fakeA_store.query(fakeA)
        #fakeA,_ = G_B(realB)
        D_B_fake = D_B(fakeA)
        D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))

        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5 
        D_B_loss.backward()
        D_B_optimizer.step()

        train_hist['D_B_losses'].append(D_B_loss.data[0])
        D_B_losses.append(D_B_loss.data[0])

        #H_A
        H_A_optimizer.zero_grad()
        fakeB,_ = G_A(realA)

        H_A_fake = H_A(fakeB)
        H_A_real = H_A(realB)

        tempI = torch.zeros(H_A_real.data.size())
        tempS = torch.zeros(H_A_real.data.size())
        regterm_3 = 0.0
        for i, ind in enumerate(batch_ind):
            tempI[i,:] = B_I[ind, :]
            tempS[i,:] = B_S[ind, :]
            H_I_[ind, :] = H_A_real.data[i]
            H_S_[ind, :] = H_A_fake.data[i]
            k = i
            while(k == i):
                p = np.random.randint(len(batch_ind))
           
                k = p

            real_code = (H_A_real[i,:] + H_A_fake[i,:])/2
            fake_code = H_A_fake[i,:]

            loss_pos =  (H_A_real[i,:] - H_A_fake[i,:]).pow(2).sum()
            loss_neg =  (H_A_real[i,:] - H_A_fake[k,:]).pow(2).sum()
            regterm_3 += F.relu(20 + loss_pos - loss_neg)

        tempI = Variable(tempI.cuda())
        tempS = Variable(tempS.cuda())

        regterm_1 = 0.0
        regterm_2 = 0.0 
        regterm_3 = 0.0 
        #regterm_4 = 0.0
                

            


        real_code = H_A_real #+ H_A_fake)/2
    
        regterm_1 = (tempI-real_code).pow(2).sum()

        regterm_2 = (tempS-H_A_fake).pow(2).sum()
       
        regterm_4 = (real_code - H_A_fake).pow(2).sum()

        # regterm_4 = (H_A_real.mm(H_A_real.t())/opt.bit - Variable(torch.eye(len(batch_ind)).cuda())).pow(2).sum()

        # regterm_5 = (H_A_fake.mm(H_A_fake.t())/opt.bit - Variable(torch.eye(len(batch_ind)).cuda())).pow(2).sum()

        # regterm_6 = H_A_real.pow(2).sum() + H_A_fake.pow(2).sum()


        H_A_hash_loss = (regterm_1 + regterm_2  + regterm_4 + 0.1*regterm_3)/opt.batch_size
        
        H_A_hash_loss.backward()
        H_A_optimizer.step()

        train_hist['H_A_Hash_losses'].append(H_A_hash_loss.data[0])
        H_A_Hash_losses.append(H_A_hash_loss.data[0])
        
        end_batch_time = time.time()
        print(epoch,'[F STEP] TIME: %3.5f, [BATCH LOSS] : %3.5f' %((end_batch_time-start_batch_time),(H_A_hash_loss.data[0])))
        
        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    print(
    '[%d/%d] - ptime: %.2f, loss_D_A: %.3f, loss_D_B: %.3f, loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f, H_A_hash_loss: %.3f' % (
        (epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_A_losses)),
        torch.mean(torch.FloatTensor(D_B_losses)), torch.mean(torch.FloatTensor(G_A_losses)),
        torch.mean(torch.FloatTensor(G_B_losses)), torch.mean(torch.FloatTensor(A_cycle_losses)),
        torch.mean(torch.FloatTensor(B_cycle_losses)),
        torch.mean(torch.FloatTensor(H_A_Hash_losses))
        ))
    
    T_S = np.zeros([num_test,opt.bit],dtype=np.float32)
    T_I = np.zeros([num_test,opt.bit],dtype=np.float32)
    G_A.eval()
    H_A.eval()
    ts_time = time.time()
    for iteration, batch in enumerate(test_loader, 0):
        realA = batch[0]
        realB = batch[1]
        batch_ind = batch[3]
        #realA = util.imgs_resize(realA,opt.input_size)
        # realB = util.imgs_resize(realB,opt.input_size)
        realA = Variable(realA.cuda(), volatile=True)
        realB = Variable(realB.cuda(), volatile=True)
        fakeB,maskT = G_A(realA)

        # mask = maskT.cpu().data.numpy()
        # mask = np.squeeze(mask)
        # mask = mask.transpose(1,2,0)
        # mask = cv2.resize(mask,(256,256))
        # mask = mask.transpose(2,0,1)
        # mask = np.expand_dims(mask,1)
        # mask = torch.Tensor(mask)
        # maskT = Variable(mask.cuda(), volatile=True) 

        H_A_fake = H_A(fakeB)
        T_S[batch_ind.numpy(),:] = torch.sign(H_A_fake.cpu().data).numpy()
        
        H_A_real = H_A(realB)
        T_I[batch_ind.numpy(),:] = torch.sign(H_A_fake.cpu().data).numpy()
        
        
    ds_time = time.time()
    # print('LoadTestData',ds_time-ts_time)
    # D_I = np.zeros([num_train,opt.bit],dtype=np.float32)
    # td_time = time.time()
    # for iteration, batch in enumerate(data_loader, 0):
    #     realB = batch[1]
    #     batch_ind = batch[3]
    #     realB = util.imgs_resize(realB,opt.input_size)
    #     realB= Variable(realB.cuda(), volatile=True)
    #     H_A_real = H_A(realB)
    #     D_I[batch_ind.numpy(),:] = torch.sign(H_A_real.cpu().data).numpy()
    # de_time = time.time()
    # print('LoadData',de_time-td_time)
    

    
    # B = B_I
    # B_ = B_S
    # B = B.cpu().numpy()
    # B_ = B_.cpu().numpy()
    # D_I = np.sign((H_I_.cpu().numpy()))

    # map
    #map_1 = CalcHR.CalcMap(T_S, D_I,test_labels_onehot.numpy(), train_labels_onehot.numpy())
    # map@50
    #map_2 = CalcHR.CalcTopMap(T_S, D_I,test_labels_onehot.numpy(), train_labels_onehot.numpy(),200)
    
     # map
    # map_11 = CalcHR.CalcMap(T_S, D_I,test_labels_onehot.numpy(), dataset_labels_onehot.numpy())
    # # map@50
    # map_22 = CalcHR.CalcTopMap(T_S, D_I,test_labels_onehot.numpy(), dataset_labels_onehot.numpy(),200)
    # # retrieval acc

    #precision@50
    #pr_20 = CalcHR.CalcTopAcc(T_S, D_I,test_labels_onehot.numpy(), train_labels_onehot.numpy(),200)
    # pr_50 = CalcHR.CalcTopAcc(T_S, H_I,test_labels_onehot.numpy(), train_labels_onehot.numpy(),100)
    acc_1 = CalcHR.CalcReAcc(T_S,T_I,1)
    acc_5 = CalcHR.CalcReAcc(T_S,T_I,5)
    acc_10 = CalcHR.CalcReAcc(T_S,T_I,10)



    loss = torch.mean(torch.FloatTensor(H_A_Hash_losses))
    
    print('[LOSS]: %3.5f' % loss)
    print('[Retrieval Phase] ACC(retrieval train_data): %3.5f' % acc_1)
    print('[Retrieval Phase] @5ACC(retrieval train_data): %3.5f' % acc_5)
    print('[Retrieval Phase] @10ACC(retrieval train_data): %3.5f' % acc_10)
    

    loss = round(loss,4)
    map_1 = round(acc_1,4)
    map_2 = round(acc_5,4)
    pr_20 = round(acc_10,4)

    file.write(str(loss)+'  '+str(map_1)+'  '+str(map_2)+'  '+str(pr_20)+ '\n')
  
    if acc_10 > map_:
        
        np.save(str(opt.bit)+'fg-T_S.npy',T_S)
        np.save(str(opt.bit)+'fg-T_I.npy',T_I)
        
        map_ = acc_10

    if (epoch+1) % 10 == 0:
        # test A to B
        n = 0
        for realA, _, _, _, in test_loader:
            n += 1
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_input.png'
            plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realA = Variable(realA.cuda(), volatile=True)
            genB,mask = G_A(realA)
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_output.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recA,_ = G_B(genB)
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_recon.png'
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

         #test B to A
        n = 0
        for _, realB, _, _ in test_loader:
            n += 1
            path = opt.dataset + '_results/test_results/BtoA/' + str(n) + '_input.png'
            plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realB = Variable(realB.cuda(), volatile=True)
            genA,_ = G_B(realB)
            path = opt.dataset + '_results/test_results/BtoA/' + str(n) + '_output.png'
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recB,_ = G_A(genA)
            path = opt.dataset + '_results/test_results/BtoA/' + str(n) + '_recon.png'
            plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    else:
        n = 0
        for real in train_loader:
            realA = real[0]
            realB = real[1]
            n += 1
            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_inputA.png'
            plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
            
            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_inputB.png'
            plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
            
            realA = Variable(realA.cuda(), volatile=True)
            genB,mask = G_A(realA)
            mask = mask[0].cpu().data.numpy()
            mask = np.squeeze(mask)
            #mask = mask.transpose(1,2,0)
            mask = cv2.resize(mask,(256,256))
            mask = np.expand_dims(mask,2)
            #mask = mask.transpose(2,0,1)
            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_mask.png'
            plt.imsave(path, ((realA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)*mask)

            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_output.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recA,_ = G_B(genB)
            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_recon.png'
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            if n > 9:
                break

        # test B to A
        n = 0
        for real in train_loader:
            realA = real[0]
            realB = real[1]
            n += 1
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_inputB.png'
            plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
          
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_inputA.png'
            plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
            
            realB = Variable(realB.cuda(), volatile=True)
            genA,mask = G_B(realB)
            mask = mask[0].cpu().data.numpy()
            mask = np.squeeze(mask)
            #mask = mask.transpose(1,2,0)
            mask = cv2.resize(mask,(256,256))
            #mask = mask.transpose(2,0,1)
            mask = np.expand_dims(mask,2)
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_mask.png'
            plt.imsave(path, ((realB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)*mask)
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_output.png'
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recB,_ = G_A(genA)
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_recon.png'
            plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            if n > 9:
                break

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G_A.state_dict(), root + model +str(opt.bit)+ 'generatorA_param.pkl')
torch.save(G_B.state_dict(), root + model +str(opt.bit)+ 'generatorB_param.pkl')
torch.save(D_A.state_dict(), root + model +str(opt.bit)+ 'discriminatorA_param.pkl')
torch.save(D_B.state_dict(), root + model +str(opt.bit)+ 'discriminatorB_param.pkl')
torch.save(H_A.state_dict(), root + model +str(opt.bit)+ 'HA_param.pkl')

