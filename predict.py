
import pandas as pd
import numpy as np
import pickle
import argparse
import json
import sys
from scipy.sparse import csr_matrix


print('Loading model...', file=sys.stderr)
model = pickle.load(open('model.pkl', 'rb'))
print('Finished loading model.', file=sys.stderr)

print('Loading matrix...', file=sys.stderr)
orders = pd.read_csv('data/orders.csv', sep=';')
users = pd.read_csv('data/users.csv', sep=';')
products = pd.read_csv('data/products.csv', sep=';')

last_orders_1 = orders.groupby('user_id').agg({'order_id':'last'}).reset_index()
train_orders = orders[~orders.order_id.isin(list(last_orders_1['order_id']))]
last_orders_2 = train_orders.groupby('user_id').agg({'order_id':'last'}).reset_index()

last_orders = pd.concat([last_orders_1, last_orders_2])

test_orders = orders[orders.order_id.isin(list(last_orders['order_id']))]
train_orders = orders[~orders.order_id.isin(list(last_orders['order_id']))]

data_test = np.array([1]*len(test_orders))
row_ind_test = np.array(test_orders['user_id'])
col_ind_test = np.array(test_orders['product_id'])
item_user_data_test = csr_matrix((data_test, (row_ind_test, col_ind_test)), \
								shape=(len(users)+1, len(products)+1)).T.tocsr()

train_orders['cnt'] = (np.log2(train_orders['day']+1) + 1)/ (np.log2(30) + 1)
data_train = np.array(train_orders['cnt'])
row_ind_train = np.array(train_orders['user_id'])
col_ind_train = np.array(train_orders['product_id'])
item_user_data_train = csr_matrix((data_train, (row_ind_train, col_ind_train)), \
								shape=(len(users)+1, len(products)+1)).T.tocsr()

user_items = item_user_data_train.T.tocsr()
print('Finished loading matrix.', file=sys.stderr)


while 1:
	print('Waiting for userID...', file=sys.stderr)
	uid = int(sys.stdin.readline())
	print('Got userID.', file=sys.stderr)
	res = []
	if (uid == -1):
		for _, row in recs.iterrows():
			user_id = int(row[0])
			r = res.get(user_id, [])
			product_id = int(row[1])
			score = row[2]
			d = dict()
			d['Score'] = score
			d['ProductID'] = product_id
			r.append(d)
			res[user_id] = r
	else:
		recommendations = model.recommend(uid, user_items, N=100)
		pr_sc = []
		for t in recommendations:
			product_id = int(t[0])
			score = t[1] 
			score = t[1] 
			d = dict()
			d['Score'] = str(score)
			d['ProductID'] = product_id
			pr_sc.append(d)
		res = pr_sc

	print(json.dumps(res))
	sys.stdout.flush()
