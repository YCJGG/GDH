import os, time, pickle, argparse, network, util, itertools
import torch
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument('--dataset', required=False, default='shoes',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--input_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--output_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--input_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--output_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nb', type=int, default=9, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop', type=bool, default=True, help='random crop True or False')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=300, help='train epochs num')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')
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

DATA_DIR = './data/shoes'
TRAIN_FILE = 'train_img.txt'
TEST_FILE = 'test_img.txt'

# data_loader
transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

dset_train = DatasetProcessing(DATA_DIR, TRAIN_FILE, transformations)
dset_test = DatasetProcessing(DATA_DIR, TEST_FILE, transformations)
num_train, num_test =  len(dset_train), len(dset_test)




train_loader = DataLoader(dset_train,batch_size=opt.batch_size,shuffle=True,num_workers=4)
test_loader = DataLoader(dset_test,batch_size=opt.batch_size,shuffle=False,num_workers=4)


### training phase
# parameters setting

# FOR HA
B_real_ = torch.randn(num_train, opt.bit)
B_fake_ = torch.randn(num_train, opt.bit)
H_I_ = torch.zeros(num_train, opt.bit)
H_S_ = torch.zeros(num_train, opt.bit)

B_I = torch.sign(torch.sign(B_real_))
B_S = torch.sign(torch.sign(B_fake_))


# network
G_A = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb)


net = models.resnet18(pretrained = False)
net.fc= nn.Linear(2048, 64)

net_dict = net.state_dict()

pretrain_dict = torch.load('./ex_sketch_64HA_param.pkl')
pretrain_dict = {k[7:] : v for k, v in pretrain_dict.items() if  k[7:] in net_dict}
net_dict.update(pretrain_dict) 
net.load_state_dict(net_dict)   
H_A = net

G_A_dict = G_A.state_dict()
pretrain_dict = torch.load('./shoes_64generatorA_param.pkl')
pretrain_dict = {k[7:] : v for k, v in pretrain_dict.items() if  k[7:] in G_A_dict}
G_A_dict.update(pretrain_dict)
G_A.load_state_dict(G_A_dict)

G_A = nn.DataParallel(G_A)
H_A = nn.DataParallel(H_A)

G_A.cuda()
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


H_A_optimizer = optim.Adam(H_A.parameters(), lr=opt.lrH, betas=(opt.beta1, opt.beta2),weight_decay=1e-5)
#H_A_optimizer = optim.Adam(H_A.parameters(), lr=opt.lrH)
# image store
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


file = open('rank_log'+str(opt.bit)+'.log','a')
start_time = time.time()
acc_ = 0.0
acc = 0.0
acc1 = 0.0
acc1_ = 0.0
G_A.eval()
for epoch in range(opt.train_epoch):

   
    H_A.train()

# F step
    H_A_Hash_losses = []
    H_B_Hash_losses = [0]
    D_A_losses = [0]
    D_B_losses = [0]
    G_A_losses = [0]
    G_B_losses = [0]
    A_cycle_losses = [0]
    B_cycle_losses = [0]
    B_Cons_losses  = [0]
    A_Cons_losses  = [0]
    epoch_start_time = time.time()
    num_iter = 0
    loss = 0
 
    H_A_optimizer.param_groups[0]['lr']  = opt.lrH*(0.1**(epoch//80))
    start_batch_time = time.time()
    
    for iteration, batch in enumerate(train_loader, 0):
        realA = batch[0]
        realB = batch[1]
        batch_ind = batch[2]
        
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
        
        fakeB = G_A(realA)
        H_A_optimizer.zero_grad()

        H_A_real = H_A(realB)
        #H_A_fake = H_A(fakeB)
        H_A_fake = H_A(realA)

        Bbatch_real = torch.sign(H_A_real)
        Bbatch_fake = torch.sign(H_A_fake)
    
        regterm_3 = 0.0 
        for i, ind in enumerate(batch_ind):
            k = i
            while(k == i):
                p = np.random.randint(len(batch_ind))
           
                k = p

            real_code = (H_A_real[i,:] + H_A_fake[i,:])/2
            fake_code = H_A_fake[i,:]

            loss_pos =  (H_A_real[i,:] - H_A_fake[i,:]).pow(2).sum()
            loss_neg =  (H_A_real[i,:] - H_A_fake[k,:]).pow(2).sum()
            regterm_3 += F.relu(18 + loss_pos - loss_neg)
            
        regterm_1 = (Bbatch_real - H_A_real).pow(2).sum() + (Bbatch_fake - H_A_fake).pow(2).sum()
        regterm_2 = (H_A_real - H_A_fake).pow(2).sum()

        H_A_hash_loss = regterm_3 
        H_A_hash_loss.backward()
        H_A_optimizer.step()

        train_hist['H_A_Hash_losses'].append(H_A_hash_loss.data[0])
        H_A_Hash_losses.append(H_A_hash_loss.data[0])
        
        end_batch_time = time.time()
        print('[F STEP] TIME: %3.5f, [BATCH LOSS] : %3.5f' %((end_batch_time-start_batch_time),(H_A_hash_loss.data[0])))
    
       

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
    T_S = np.zeros([num_test,opt.bit],dtype=np.float32)
    T_I = np.zeros([num_test,opt.bit],dtype=np.float32)
    T_SS = np.zeros([num_test,opt.bit],dtype=np.float32)
    T_II = np.zeros([num_test,opt.bit],dtype=np.float32)
    H_A.eval()
    for iteration, batch in enumerate(test_loader, 0):
        realA = batch[0]
        realB = batch[1]
        batch_ind = batch[2]
        realA = util.imgs_resize(realA,opt.input_size)
        realB = util.imgs_resize(realB,opt.input_size)
        realA = Variable(realA.cuda(), volatile=True)
        realB = Variable(realB.cuda(), volatile=True)
        fakeB = G_A(realA)
        #H_A_fake = H_A(fakeB)
       
        H_A_fake = H_A(realA)
        H_A_real = H_A(realB)
        T_S[batch_ind.numpy(),:] = torch.sign(H_A_fake.cpu().data).numpy()
        #T_I[batch_ind.numpy(),:] = (torch.sign(H_A_real.cpu().data).numpy()+torch.sign(H_A_fake.cpu().data).numpy())/2
        T_I[batch_ind.numpy(),:] = torch.sign(H_A_real.cpu().data).numpy()
        T_SS[batch_ind.numpy(),:] = H_A_fake.cpu().data.numpy()
        T_II[batch_ind.numpy(),:] = H_A_real.cpu().data.numpy()
        #T_II[batch_ind.numpy(),:] = (H_A_real.cpu().data.numpy()+H_A_fake.cpu().data.numpy())/2


 
    # retrieval acc
    acc_1 = CalcHR.CalcReAcc(T_S,T_I,1)
    #acc_5 = CalcHR.CalcReAcc(T_S,T_I,5)
    acc_10 = CalcHR.CalcReAcc(T_S,T_I,10)
    #acc_20 = CalcHR.CalcReAcc(T_S,T_I,20)    

    aacc_1 = CalcHR.CalcReAcc_2(T_SS,T_II,1)
    #aacc_5 = CalcHR.CalcReAcc_2(T_SS,T_II,5)
    aacc_10 = CalcHR.CalcReAcc_2(T_SS,T_II,10)
    #aacc_20 = CalcHR.CalcReAcc_2(T_SS,T_II,20)
  
    loss = torch.mean(torch.FloatTensor(H_A_Hash_losses))
    
    
    print(epoch+1,'[Retrieval Phase] @loss): %3.5f' % loss)
    print(epoch+1,'[Retrieval Phase] @1_acc(retrieval train_data): %3.5f' % acc_1)
    #print(epoch+1,'[Retrieval Phase] @5_acc(retrieval train_data): %3.5f' % acc_5)
    print(epoch+1,'[Retrieval Phase] @10_acc(retrieval train_data): %3.5f' % acc_10)
    #print(epoch+1,'[Retrieval Phase] @20_acc(retrieval train_data): %3.5f' % acc_20)
    
    print(epoch+1,'[Retrieval Phase] @1_acc(retrieval train_data): %3.5f' % aacc_1)
    #print(epoch+1,'[Retrieval Phase] @5_acc(retrieval train_data): %3.5f' % aacc_5)
    print(epoch+1,'[Retrieval Phase] @10_acc(retrieval train_data): %3.5f' % aacc_10)
    #print(epoch+1,'[Retrieval Phase] @20_acc(retrieval train_data): %3.5f' % aacc_20)

    loss = round(loss,4)
    acc_1 = round(acc_1,4)
    #acc_5 = round(acc_5,4)
    acc_10 = round(acc_10,4)
    #acc_20 = round(acc_20,4)
  

    file.write(str(acc_1)+'  '+str(acc_10)+'  '+ str(aacc_1)+ '  '+str(aacc_10)+ '\n')
  
    if acc_10 > acc_:
       
        np.save(str(opt.bit)+'b10rank-T_S.npy',T_S)
        np.save(str(opt.bit)+'b10rank-T_I.npy',T_I)
       
        acc_ = acc_10
    
    if aacc_10 > acc:
       
        np.save(str(opt.bit)+'r10rank-T_SS.npy',T_SS)
        np.save(str(opt.bit)+'r10rank-T_IS.npy',T_II)
       
        acc = aacc_10
    if acc_1 > acc1_:
       
        np.save(str(opt.bit)+'b1rank-T_S.npy',T_S)
        np.save(str(opt.bit)+'b1rank-T_I.npy',T_I)
       
        acc1_ = acc_1
    
    if aacc_1 > acc1:
       
        np.save(str(opt.bit)+'r1rank-T_SS.npy',T_SS)
        np.save(str(opt.bit)+'r1rank-T_IS.npy',T_II)
       
        acc1 = aacc_1
