import torch
import torch.multiprocessing as mp
import train
import model
from train.args import *
from train.load_data import collection

net = model.JoinConvNet(n1, c, t, o)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=lr)

num_process = 1
cuda_available = torch.cuda.is_available()

train_set, test_set = train.train_test_split(
	collection, test_ratio, random_seed)
getvar = train.JointDataset.getvar
collate = train.JointDataset.collate
trainer = train.Trainer(optim, loss_fn, getvar)

if num_process == 1:
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
	exit()

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

'''
mse_train = []
mse_test = []

for e in range(epochs):
	cnt = len(I_train)
	for idx in I_train:
		pred, y = predict(net, load_data.collection, idx)
		loss = train(net, optim, loss_fn, pred, y)
		print(cnt, loss.data[0], end=',')
		cnt -= 1
	#mse_train.append(mean_squared_error(predict_batch(net, I_train), y_train))
	mse_test.append(mean_squared_error(predict_batch(net, I_test), y_test))
	print(i, "th iteration:", mse_train[-1], mse_test[-1])

from matplotlib import pyplot as plt
x = np.arange(1, epochs)
with plt.style.context('fivethirtyeight'):
	#plt.plot(x, mse_train, label='train')
	plt.plot(x, mse_test, label='test')
plt.legend()
plt.show()
'''