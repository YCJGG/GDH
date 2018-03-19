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
parser.add_argument('--dataset', required=False, default='chairs',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--input_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--output_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--input_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--output_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nb', type=int, default=9, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop', type=bool, default=True, help='random crop True or False')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=200, help='train epochs num')
parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--bit', type=int, default=128, help='bit')
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

DATA_DIR = './data/chairs'
TRAIN_FILE = 'img.txt'
TEST_FILE = 'test.txt'

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
G_B = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb)
D_A = network.discriminator(opt.input_ndc, opt.output_ngc, opt.ndf)

net = models.resnet18(pretrained = True)
net.fc= nn.Linear(2048, opt.bit)

# net_dict = net.state_dict()

# pretrain_dict = torch.load('./pre_resnet18.pkl')
# pretrain_dict = {k[7:] : v for k, v in pretrain_dict.items() if  k[7:] in net_dict}
# net_dict.update(pretrain_dict) 
# net.load_state_dict(net_dict)   

# net.fc= nn.Linear(2048, opt.bit)
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


file = open('rank_log'+str(opt.bit)+'.log','a')
start_time = time.time()
map_ = 0.0

for epoch in range(opt.train_epoch):

    G_A.train()
    G_B.train()
    D_A.train()
    D_B.train()
    H_A.train()

# F step
    H_A_Hash_losses = [0]
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
    if (epoch+1) > opt.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= opt.lrG / (opt.train_epoch - opt.decay_epoch)
        #H_A_optimizer.param_groups[0]['lr']  = opt.lrH*(0.1**(epoch//opt.decay_epoch))
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
        
        # G STEP
    
        
        # train generator G
        G_optimizer.zero_grad()

        # generate real A to fake B; D_A(G_A(A))
        fakeB = G_A(realA)
        D_A_result = D_A(fakeB)
        G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))
            
        # reconstruct fake B to rec A; G_B(G_A(A))
        recA = G_B(fakeB)
        A_cycle_loss = L1_loss(recA, realA) * opt.lambdaA 

        # generate real B to fake A; D_A(G_B(B))
        fakeA = G_B(realB)
        D_B_result = D_B(fakeA)
        G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))

        # reconstruct fake A to rec B G_A(G_B(B))
        recB = G_A(fakeA)
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
        #fakeB = G_A(realA)
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
        #fakeA = G_B(realB)
        D_B_fake = D_B(fakeA)
        D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))

        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5 
        D_B_loss.backward()
        D_B_optimizer.step()

        train_hist['D_B_losses'].append(D_B_loss.data[0])
        D_B_losses.append(D_B_loss.data[0])

       
       

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    print(
    '[%d/%d] - ptime: %.2f, loss_D_A: %.3f, loss_D_B: %.3f, loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f' % (
        (epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_A_losses)),
        torch.mean(torch.FloatTensor(D_B_losses)), torch.mean(torch.FloatTensor(G_A_losses)),
        torch.mean(torch.FloatTensor(G_B_losses)), torch.mean(torch.FloatTensor(A_cycle_losses)),
        torch.mean(torch.FloatTensor(B_cycle_losses))
        ))
    

    if (epoch+1) % 10 == 0:
        # test A to B
        n = 0
        for realA, _, _, in test_loader:
            n += 1
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_input.png'
            plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realA = Variable(realA.cuda(), volatile=True)
            genB = G_A(realA)
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_output.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recA = G_B(genB)
            path = opt.dataset + '_results/test_results/AtoB/' + str(n) + '_recon.png'
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

         #test B to A
        n = 0
        for _, realB, _ in test_loader:
            n += 1
            path = opt.dataset + '_results/test_results/BtoA/' + str(n) + '_input.png'
            plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realB = Variable(realB.cuda(), volatile=True)
            genA = G_B(realB)
            path = opt.dataset + '_results/test_results/BtoA/' + str(n) + '_output.png'
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recB = G_A(genA)
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
            genB = G_A(realA)
        
            path = opt.dataset + '_results/train_results/AtoB/' + str(n) + '_output.png'
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recA = G_B(genB)
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
            genA = G_B(realB)
            path = opt.dataset + '_results/train_results/BtoA/' + str(n) + '_output.png'
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recB = G_A(genA)
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

