import sklearn.model_selection
import torch
import torch.utils.data
from torch.autograd import Variable
from .load_data import *
import os

'''
def get_X(collection, I, embed_name=None):
	X = []
	for idx in I:
		cell = collection.data[idx]
		keys = [cell[key_name] for key_name in collection.key_names]
		X.append([collection.concate_by_key(*pair, embed_name) 
			for pair in zip(collection.key_names, keys)])
	return X

def get_y(collection, I, field_name):
	return [collection.data[i][field_name] for i in I]

def train_test_split(collection, test_ratio, seed):
	item_idx = list(range(len(collection.data)))
	I_train, I_test, _, _ = train_test_split(
		item_idx, item_idx, test_size=test_ratio, random_state=seed)
	X_train = get_X(collection, I_train)
	X_test = get_X(collection, I_test)
	y_train = get_y(collection, I_train, "overall")
	y_test = get_y(collection, I_test, "overall")
'''

make_var = lambda x, volatile: torch.autograd.Variable(torch.FloatTensor(x), volatile=volatile)

class JointDataset(torch.utils.data.Dataset):
	def __init__(self, collection, indices):
		self.collection = collection
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		cell = collection.data[self.indices[idx]]
		keys = [cell[key_name] for key_name in collection.key_names]
		target = cell[collection.target_name]
		data = [collection.concate_by_key(*pair, collection.data_name)
			for pair in zip(collection.key_names, keys)]
		return data, target
	
	@staticmethod
	def getvar(data, target, for_test=False):
		n_col = len(data[0])
		data = [ (Variable(torch.FloatTensor(
			JointDataset.pad([[row[i]] for row in data])), 
			volatile=for_test)) for i in range(n_col) ]
		target = Variable(torch.FloatTensor([target]))
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

class Trainer:
	def __init__(self, optimizer, loss_fn, getvar):
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.getvar = getvar
	
	def train(self, model, train_loader, test_loader, epochs, log_interval=1):
		for epoch in range(1, epochs + 1):
			self.train_epoch(epoch, model, train_loader, log_interval)
			self.test_epoch(model, test_loader)
	
	@staticmethod
	def timestamp():
		return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

	def train_epoch(self, epoch, model, data_loader, log_interval):
		model.train()
		pid = os.getpid()
		for batch_idx, (data, target) in enumerate(data_loader):
			data, target = self.getvar(data, target)
			self.optimizer.zero_grad()
			pred = model(data)
			loss = self.loss_fn(pred, target)
			loss.backward()
			self.optimizer.step()
			if batch_idx % log_interval == 0:
				print('{} [{}] Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
					timestamp(), pid, epoch, batch_idx * len(target), len(data_loader.dataset),
					100. * batch_idx / len(data_loader), loss.data[0]))

	def test_epoch(self, epoch, model, data_loader, log_interval):
		model.eval()
		test_loss = 0
		for data, target in data_loader:
			data, target = self.getvar(data, target, for_test=True)
			pred = model(data)
			test_loss += self.loss_fn(pred, target).data[0]
		test_loss /= len(data_loader)
		print('\n{} Test set: Average loss: {:.4f}\n'.format(timestamp(), test_loss))