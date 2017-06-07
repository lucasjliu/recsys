import torch
import torch.multiprocessing as mp
import train
import train.data
from train.args import *
import model
import random
import sys

net = model.JoinConvNet(n1, c, t, o)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=mmt, weight_decay=wd)

num_process = 1
cuda_available = torch.cuda.is_available()

random.seed(4)
train.data.load(sparsity=2)

train_set, test_set = train.train_test_split(
	train.data.collection, test_ratio, random_seed)
getvar = train.JointDataset.getvar
collate = train.JointDataset.collate
trainer = train.Trainer(optim, loss_fn, getvar)

if num_process == 1:
	net = model.JoinConvNet(n1, c, t, o)
	optim = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=mmt, weight_decay=wd)
	trainer = train.Trainer(optim, loss_fn, getvar)

	torch.manual_seed(random_seed+1)
	if cuda_available:
		torch.cuda.manual_seed(random_seed+1)
		#net = torch.nn.DataParallel(net).cuda()
		net = net.cuda()
		loss_fn = loss_fn.cuda()
	train_loader = torch.utils.data.DataLoader(train_set, 
		batch_size=batch_size, shuffle=True, 
		collate_fn=collate, pin_memory=cuda_available)
	test_loader = torch.utils.data.DataLoader(test_set, 
		batch_size=batch_size, shuffle=True,
		collate_fn=collate, pin_memory=cuda_available)
	trainer.train(net, train_loader, test_loader, epochs, log_interval)
sys.exit()


net.share_memory()

processes = []
for rank in range(1, num_process+1):
	torch.manual_seed(random_seed+rank)
	train_loader = torch.utils.data.DataLoader(train_set, 
		batch_size=batch_size, shuffle=True, collate_fn=collate)
	test_loader = torch.utils.data.DataLoader(test_set, 
		batch_size=batch_size, shuffle=True, collate_fn=collate)
	p = mp.Process(target=trainer.train, 
		args=(net, train_loader, test_loader, epochs, log_interval))
	p.start()
	processes.append(p)
for p in processes:
	p.join()