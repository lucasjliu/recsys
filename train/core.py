#
#	by Jiahuan Liu
#	06/2017
#
import sklearn.model_selection
import torch
import torch.utils.data
from torch.autograd import Variable
import os
import time
import datetime
from .data import collection

class JointDataset(torch.utils.data.Dataset):
	def __init__(self, collection, indices):
		self.collection = collection
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		cell = self.collection.data[self.indices[idx]]
		keys = [cell[key_name] for key_name in self.collection.key_names]
		target = cell[self.collection.target_name]
		data = [self.collection.concate_by_key(*pair, self.collection.data_name)
			for pair in zip(self.collection.key_names, keys)]
		return data, target
	
	@staticmethod
	def getvar(data, target, for_test=False):
		n_col = len(data[0])
		if torch.cuda.is_available():
			data = [ (Variable(torch.FloatTensor(
				JointDataset.pad([[row[i]] for row in data])).cuda(), 
				volatile=for_test)) for i in range(n_col) ]
		else:
			data = [ (Variable(torch.FloatTensor(
				JointDataset.pad([[row[i]] for row in data])), 
				volatile=for_test)) for i in range(n_col) ]
		target = torch.FloatTensor(target)
		if torch.cuda.is_available():
			target = target.cuda(async=True)
		target = Variable(target)
		return data, target
	
	@staticmethod
	def collate(batch):
		return ([row[0] for row in batch], [row[1] for row in batch])
	
	@staticmethod
	def pad(data, value=0):
		# only support 4D list, padding on dim 2
		maxlen = max([len(row[0]) for row in data])
		paddings = [value] * len(data[0][0][0])
		for row in data:
			for _ in range(maxlen - len(row[0])):
				row[0].append(paddings)
		return data

def train_test_split(collection, test_ratio, random_seed):
	indices = list(range(len(collection.data)))
	train_indices, test_indices, _, _ = sklearn.model_selection.train_test_split(
		indices, indices, test_size=test_ratio, random_state=random_seed)
	train_set = JointDataset(collection, train_indices)
	test_set = JointDataset(collection, test_indices)
	return train_set, test_set

def timestamp():
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

class Trainer:
	def __init__(self, optimizer, loss_fn, getvar):
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.getvar = getvar
	
	def train(self, model, train_loader, test_loader, epochs, log_interval=1):
		for epoch in range(1, epochs + 1):
			self.train_epoch(epoch, model, train_loader, log_interval)
			self.test_epoch(model, test_loader)
	
	def train_epoch(self, epoch, model, data_loader, log_interval):
		model.train()
		pid = os.getpid()
		train_loss = 0
		for batch_idx, (data, target) in enumerate(data_loader, 1):
			data, target = self.getvar(data, target)
			self.optimizer.zero_grad()
			pred = model(data)
			loss = self.loss_fn(pred, target)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.data[0]
			if batch_idx % log_interval == 0:
				sample_num = len(data_loader.dataset)
				print('[{}][{}] Epoch {}, {}/{} ({:.1f}%) Loss: {:.6f}'.format(
					pid, timestamp(), epoch, min(batch_idx * data_loader.batch_size, sample_num), sample_num,
					100. * batch_idx / len(data_loader), loss.data[0]))
		train_loss /= len(data_loader)
		print('\n[{}] Train set average loss: {:.4f}\n'.format(timestamp(), train_loss))

	def test_epoch(self, model, data_loader):
		model.eval()
		test_loss = 0
		for data, target in data_loader:
			data, target = self.getvar(data, target, for_test=True)
			pred = model(data)
			test_loss += self.loss_fn(pred, target).data[0]
		test_loss /= len(data_loader)
		print('\n[{}] Test set average loss: {:.4f}\n'.format(timestamp(), test_loss))