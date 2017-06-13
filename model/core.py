#
#	by Jiahuan Liu
#	06/2017
#
import pandas
import json
import random

class Datasrc:
	def __getitem__(self, key):
		raise NotImplementedError
	def __len__():
		raise NotImplementedError
	def list():
		raise NotImplementedError

class JsonDatasrc(Datasrc):
	def __init__(self, file_path, file_open_func=open, load_item_func=json.loads):
		self.raw_json = []
		for line in file_open_func(file_path):
			self.raw_json.append(load_item_func(line))
			self.raw_json[-1]["id"] = len(self.raw_json) - 1
		self.size = len(self.raw_json)

	def describe(self, exclude=[]):
		data_frame = pandas.DataFrame(self.raw_json)
		for del_col in exclude: del data_frame[del_col]
		print(data_frame.describe(include='all'))

	def count_by_col(self, key_name):
		count = {}
		for cell in self.raw_json:
			key_id = cell[key_name]
			if count.get(key_id) is None:
				count[key_id] = 0
			count[key_id] += 1
		print(pandas.DataFrame(
			count[key] for key in count.keys()).describe())

	def __getitem__(self, key):
		return self.raw_json[key]

	def get(self, key):
		return self.raw_json.get(key)

	def list(self):
		return iter(self.raw_json)

	def __len__(self):
		return self.size

	def mv_col(self, name, new_name):
		for cell in self.list():
			if new_name: 
				cell[new_name] = cell[name]
			del cell[name]
	
	def rm_col(self, name):
		self.mv_col(name, None)

	def shuffle(self):
		random.shuffle(self.raw_json)
		self.resetid()
	
	def resetid(self):
		for i in range(len(self.raw_json)):
			self.raw_json[i]["id"] = i

class _KeyIndex:
	def __init__(self):
		self.id = {}
		self.list = []

	def add(self, key, item_id):
		if self.id.get(key) is None:
			self.id[key] = len(self.id)
			self.list.append([])
		self.list[self.id[key]].append(item_id)

	def __getitem__(self, key):
		return self.id[key]

	def get(self, key):
		return self.id.get(key)

	def get_list(self, key):
		return self.list[self.id[key]]

class KVCollection:
	def __init__(self, dataset, key_names, field_names=[]):
		self.data = dataset
		self.key_names = key_names
		self.key_num = len(key_names)
		self.key_idx = {key_name: _KeyIndex() for key_name in key_names}
		self.field_names = field_names
		self.target_name = None

	def list(self, field_names=None):
		if not field_names: field_names = self.field_names
		for cell in self.data.list():
			yield ([self.key_idx[key_name].get(cell[key_name])
				for key_name in self.key_names],
				[cell[key_name] for key_name in self.key_names],
				[cell[field_name] for field_name in field_names])

	def cycle(self, epochs, field_names=None):
		if not field_names: field_names = self.field_names
		while epochs > 0:
			for entry in self.list(field_names):
				yield entry
			epochs -= 1

	def build_key_index(self):
		for cell in self.data.list():
			for key_name in self.key_names:
				self.key_idx[key_name].add(cell[key_name], cell["id"])

	def find_by_key(self, key_name, key):
		return self.key_idx[key_name].get_list(key)

	def get_by_key(self, key_name, key):
		return [self.data[i] for i in self.find_by_key(key_name, key)]

import nltk
import gensim

class NlpUtil:
	@staticmethod
	def tokenize(str):
		return nltk.word_tokenize(str)

	@staticmethod
	def load_word2vec(file_path):
		return gensim.models.KeyedVectors.load_word2vec_format(
			file_path, binary=True)

class TextCollection(KVCollection):
	default_embed_name = "embed"

	def __init__(self, dataset, key_names, field_names):
		KVCollection.__init__(self, dataset, key_names, field_names)
		self.data_name = TextCollection.default_embed_name

	def build_embeddings(self, embeds, field_name):
		for cell in self.data.list():
			cell[self.default_embed_name] = [embeds[word]
				for word in NlpUtil.tokenize(cell[field_name])
				if word in embeds]

	def concate_by_key(self, key_name, key, embed_name=None):
		if embed_name is None: embed_name = self.data_name
		res = []
		for cell in self.get_by_key(key_name, key):
			for word in cell[embed_name]:
				res.append(word.tolist())
		return res

import torch
import torch.nn as nn
import functools

class Net(torch.nn.Module):
	def num_flat_features(self, x):
		return functools.reduce(lambda x, y: x * y, x.size()[1:])

class TextConvNet(Net):
	def __init__(self, num_neuron, embed_size, words_window_size, output_size):
		Net.__init__(self)
		self.conv = nn.Conv2d(1, num_neuron, (words_window_size, embed_size))
		self.fc = nn.Linear(num_neuron, output_size)

	def forward(self, x):
		x = self.conv(x)
		x, _ = x.max(dim=2)
		x = nn.functional.relu(x)
		x = x.view(-1, self.num_flat_features(x))
		x = nn.functional.relu(self.fc(x))
		return x

class FactorsMatrix(torch.nn.Module):
	def __init__(self, N, F):
		torch.nn.Module.__init__(self)
		self.linear = torch.nn.Linear(N, F)

	@staticmethod
	def one_hot(size, idx):
		return torch.zeros(size).scatter_(0, torch.LongTensor([idx]), 1).view(1, -1)
	
	def get_row(self, idx):
		one_hot = self.one_hot(self.linear.in_features, idx)
		if torch.cuda.is_available(): one_hot = one_hot.cuda()
		row = self.linear(torch.autograd.Variable(one_hot))
		return row
	
	def forward(self, i, j):
		vi, vj = self.get_row(i), self.get_row(j)
		return vi.dot(vj)

class FactMachine(torch.nn.Module):
	def __init__(self, D, F):
		torch.nn.Module.__init__(self)
		self.w0 = torch.nn.Linear(1, 1)
		self.w = torch.nn.Linear(D, 1)
		self.V = FactorsMatrix(D, F)

	def forward(self, X):
		m = len(X)
		ones = torch.ones(m, 1) 
		if torch.cuda.is_available(): ones = ones.cuda()
		y = self.w0(torch.autograd.Variable(ones))
		y = y.add(self.w(X))
		dim = self.V.linear.in_features
		for k in range(m):
			x = X[k]
			for i in range(dim):
				for j in range(i+1, dim):
					y[k] = y[k].add( x[i].dot(x[j]).mul(self.V(i, j)) )
		return y

class MatFact2d(torch.nn.Module):
	def __init__(self, M, N, F):
		torch.nn.Module.__init__(self)
		self.matA = FactorsMatrix(M, F)
		self.matB = FactorsMatrix(N, F)

	def forward(self, ia, ib):
		return self.matA.get_row(ia).dot(self.matB.get_row(ib))

class JoinConvNet(torch.nn.Module):
	def __init__(self, num_neuron=80, embed_size=300, words_window_size=3, output_size=30):
		torch.nn.Module.__init__(self)
		self.conv1 = TextConvNet(num_neuron, embed_size, words_window_size, output_size)
		self.conv2 = TextConvNet(num_neuron, embed_size, words_window_size, output_size)
		#self.mf = FactMachine(output_size * 2, output_size * 2)

	def forward(self, x):
		o = [self.conv1(x[0]), self.conv2(x[1])]
		#'''
		m = len(x[0])
		y = torch.autograd.Variable(torch.ones(m, 1))
		for k in range(m):
			y[k] = o[0][k].dot(o[1][k])
		return y
		#'''
		#return self.mf( torch.cat([o[0], o[1]], dim=1) )