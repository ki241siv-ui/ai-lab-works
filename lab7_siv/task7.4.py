import json
import numpy as np
from sklearn import covariance, cluster

company_symbols_map = {
    'TOT': 'Total', 'XOM': 'Exxon', 'CVX': 'Chevron', 'COP': 'ConocoPhillips',
    'VLO': 'Valero Energy', 'MSFT': 'Microsoft', 'IBM': 'IBM', 'AAPL': 'Apple',
    'HPQ': 'HP', 'DELL': 'Dell', 'GOOG': 'Google', 'AMZN': 'Amazon',
    'JPM': 'JPMorgan Chase', 'AIG': 'AIG', 'AXP': 'American Express', 'BAC': 'Bank of America'
}

input_file = 'company_symbol_mapping.json'
with open(input_file, 'w') as f:
    json.dump(company_symbols_map, f)

with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

X = np.random.randn(len(symbols), 100) 
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV()
with np.errstate(invalid='ignore'):
    edge_model.fit(X.T)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))