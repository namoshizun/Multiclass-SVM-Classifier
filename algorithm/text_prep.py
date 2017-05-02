import pandas as pd
import numpy as np 
import sys
from collections import Counter
MAX = sys.maxsize


def read_selected_features(path):
	with open(path, 'r') as source:
		return source.read().splitlines()


def rebuild_data(data_path, labels_path):
	print('==== reading training data ====')
	data = pd.read_csv(data_path, index_col=0, header=None)

	print('==== reading training labels ====')
	labels = pd.read_csv(labels_path, index_col=0, header=None)
	data.index = data.index.rename('app')
	labels.index = labels.index.rename('app')
	assert(len(data) == len(labels))

	print('==== rebuilding ====')
	tagged = ['t'+str(i) for i in range(len(data.columns))]
	data.columns = tagged
	labels.columns = ['labels']
	data = data.join(labels)

	print('==== removing redundant features ====')
	selected_fetures = read_selected_features('./gt_0.01_attrs.txt')
	redundant_features = list(set(tagged) - set(selected_fetures))
	data.drop(redundant_features, axis=1, inplace=True)
	return data


def save(df, path, encoding='ascii'):
	print('==== saving ====')
	df.to_csv(path, encoding=encoding)


def slice_classes(df, classes=None, subset_size=None):
	print('==== slicing data by classes ====')
	classes = classes or df['labels'].values
	subset_size = subset_size or MAX

	df = df[df['labels'].isin(classes)]
	label_stats = Counter(df['labels'].values)
	free_slots = Counter({x : subset_size for x in classes})

	to_drop = []
	for app, data in df.iterrows():
		label = data[-1]
		label_stats[label] -= 1
		
		if free_slots[label] < 0:
			to_drop.append(app)

	df.drop(to_drop, inplace=True)
	return df


def main(data, labels, classes, subset_size):
	df = rebuild_data(data, labels)
	sub_df = slice_classes(df, classes, subset_size)
	sub_df.set_index(['labels'], inplace=True)
	save_path = './nihao.csv'
	save(sub_df, save_path)
	print('DONE')
	return df


if __name__ == '__main__':
	training_data = './data/training_data.csv'
	training_labels = './data/training_labels.csv'
	samples_per_class = None
	# selected_classes = None
	selected_classes = [
		# 'Health and Fitness',
		'Education',
		'Media and Video',
		'Books and Reference',
		'Social',
		'Lifestyle',
		# 'Casual',
		# 'Music and Audio',
		# 'Comics',
		# 'Shopping',
		# 'Transportation',
		# 'Business',
		# 'Libraries and Demo',
		# 'Communication',
		# 'Personalization',
		# 'Travel and Local',
		# 'News and Magazines',
		# 'Tools',
		# 'Cards and Casino',
		# 'Finance',
		# 'Brain and Puzzle',
		# 'Entertainment',
		# 'Business',
		# 'Arcade and Action',
		# 'Medical',
		# 'Productivity',
		# 'Racing',
		# 'Sports',
		# 'Weather',
		# 'Sports Games'
	]

	res = main(training_data, training_labels, selected_classes, samples_per_class)
