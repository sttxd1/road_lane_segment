import torch
import numpy as np
import time
import argparse
from pathlib import Path

from models.segnet import SegNet
from models.enet import ENet
from models.resnet38 import ResNet38
# from models.bisenet import BiSeNet
from models.enet_k import ENet as ENet_k
from models.resnet38_k import ResNet38 as ResNet38_k
from loss import DiscriminativeLoss
from dataset import tuSimpleDataset
import sys
sys.path.insert(0, './')

parser = argparse.ArgumentParser(description="Train model")

parser.add_argument('--train-path', default="../TUSimple/train_set")
parser.add_argument('--test-path', default="../TUSimple/test_set")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--img-size', type=int, nargs='+', default=[224, 224], help='image resolution: [width height]')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--weight', type=float, default=None)
parser.add_argument('--model', type=str, default='enet')

args = parser.parse_args()

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 2
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epoch
LOG_INTERVAL = 10
SIZE = [args.img_size[0], args.img_size[1]] #[224, 224]

def train():
    # refer from : https://github.com/Sayan98/pytorch-segnet/blob/master/src/train.py
    is_better = True
    prev_loss_train = float('inf')
    prev_loss_test = float('inf')
    
    model.train()
    num1 = 0
    num2 = 0
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()
        loss_f = []
        model.train()
        for batch_idx, (imgs, sem_labels, ins_labels) in enumerate(train_dataloader):
            loss = 0

            img_tensor = torch.autograd.Variable(imgs).cuda()
            sem_tensor = torch.autograd.Variable(sem_labels).cuda()
            ins_tensor = torch.autograd.Variable(ins_labels).cuda()

            num1 += torch.sum(sem_tensor==0)
            num2 += torch.sum(sem_tensor==1)
            optimizer.zero_grad()

            # Predictions
            sem_pred, ins_pred = model(img_tensor)

            # Discriminative Loss
            disc_loss = criterion_disc(ins_pred, ins_tensor, [5] * len(img_tensor))
            loss += disc_loss

            # CrossEntropy Loss
            ce_loss = criterion_ce(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS),sem_tensor.view(-1))
            loss += ce_loss

            loss.backward()
            optimizer.step()

            loss_f.append(loss.cpu().data.numpy())

            if batch_idx % LOG_INTERVAL == 0:
                print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

                info = {'loss': loss.item(), 'ce_loss': ce_loss.item(), 'disc_loss': disc_loss.item(), 'epoch': epoch}
                print(info)
        dt = time.time() - t_start
        scheduler.step()
        is_better = np.mean(loss_f) < prev_loss_train
        if is_better:
            prev_loss_train = np.mean(loss_f)
            # print("\t\tBest Model.")
            torch.save(model.state_dict(), f"results/{args.prefix}_model_best_train.pth")
        print("Train: Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s, Lr: {:2f}".format(epoch+1, np.mean(loss_f), dt, optimizer.param_groups[0]['lr']))
        print("Evaluating...")
        t_start = time.time()
        loss_f = []
        model.eval()
        for batch_idx, (imgs, sem_labels, ins_labels) in enumerate(test_dataloader):
            loss = 0

            img_tensor = torch.autograd.Variable(imgs).cuda()
            sem_tensor = torch.autograd.Variable(sem_labels).cuda()
            ins_tensor = torch.autograd.Variable(ins_labels).cuda()
            sem_pred, ins_pred = model(img_tensor)
            disc_loss = criterion_disc(ins_pred, ins_tensor, [5] * len(img_tensor))
            loss += disc_loss
            ce_loss = criterion_ce_eval(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS),sem_tensor.view(-1))
            loss += ce_loss
            loss_f.append(loss.cpu().data.numpy())

            
        print("Test: Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s, Lr: {:2f}".format(epoch+1, np.mean(loss_f), dt, optimizer.param_groups[0]['lr']))
        # is_better = np.mean(loss_f) < prev_loss_test
        # # scheduler.step()
        # if is_better:
        #     prev_loss_test = np.mean(loss_f)
        #     print("\t\tBest Model.")
        #     torch.save(model.state_dict(), f"results/{args.prefix}_model_best_test.pth")
    print(num1, num2)


if __name__ == "__main__":
   Path("results/").mkdir(parents=True, exist_ok=True)
   train_path = args.train_path
   train_dataset = tuSimpleDataset(train_path, size=SIZE)
   train_len = int(len(train_dataset)*0.8)
   train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset)-train_len])
   print(len(train_dataset))
   train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
   print(len(test_dataset))
   test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=8)
   if args.model == 'enet': 
       model = ENet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
   if args.model == 'segnet':
       model = SegNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
   if args.model == 'enet_k': 
       model = ENet_k(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
#    if args.model == 'bisenet': 
#        model = BiSeNet(32, 'resnet18').cuda()
   if args.model == 'resnet38': 
       model = ResNet38().cuda()
   if args.model == 'resnet38_k': 
       model = ResNet38_k().cuda()

   if args.weight is not None:
       criterion_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([args.weight, 1-args.weight])).cuda()
   else:
       criterion_ce = torch.nn.CrossEntropyLoss().cuda()
   criterion_ce_eval = torch.nn.CrossEntropyLoss().cuda()
   criterion_disc = DiscriminativeLoss(delta_var=0.1,
                                       delta_dist=0.6,
                                       norm=2,
                                       usegpu=True).cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40,50,60,70,80], gamma=0.9)

   train()
