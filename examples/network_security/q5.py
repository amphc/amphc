# Preprocessing
# Cell 1
import pandas as pd 
import gc

# Cell 2
df = pd.read_csv(
    '~/Downloads/gowiththeflow_20190826.csv',
    header = 0, 
    names= ['ts', 'src', 'dst', 'port', 'bytes']
)
df.info()

# Cell 3
def is_internal(s):
    return s.str.startswith(('12.', '13.', '14.')) 

df['src_int'] = is_internal(df['src'])
df['dst_int'] = is_internal(df['dst'])

df['ts']      = pd.to_datetime(df.ts, unit='ms')
df['hour']    = df.ts.dt.hour.astype('uint8')
df['minute']  = df.ts.dt.minute.astype('uint8')
df['port']    = df['port'].astype('uint8')
df.head()

# Cell 4
all_ips = set(df['src'].unique()) | set(df['dst'].unique())
print('Unique src:', df['src'].nunique())
print('Unique dst:', df['dst'].nunique())
print('Total Unique IPs:', len(all_ips))

ip_type = pd.CategoricalDtype(categories=all_ips)
df['src'] = df['src'].astype(ip_type)
df['dst'] = df['dst'].astype(ip_type)
gc.collect()
df.info()

# Question 5: Internal P2P
# Cell 28
import networkx
from networkx.algorithms.approximation.clique import large_clique_size 
from collections import Counter

# Cell 29
internal_edges_all = df[
  df['src_int'] & df['dst_int']
].drop_duplicates(['src', 'dst', 'port'])
internal_ports = internal_edges_all.port.unique()

# Cell 30
port_upper_bounds = []
for p in internal_ports:
    internal_edges = internal_edges_all\
        .pipe(lambda x: x[x['port'] == p])\
        .drop_duplicates(['src', 'dst'])

    edges = set()
    for l, r in zip(internal_edges.src, internal_edges.dst):
        k = min((l, r), (r, l))
        edges.add(k)
    
    degrees = Counter()
    for (l, r) in edges:
        degrees[l] += 1
        degrees[r] += 1
    
    max_clique_size = 0
    min_degrees = len(degrees)
    for idx, (node, degree) in enumerate(degrees.most_common()):
        min_degrees = min(min_degrees, degree)
        if min_degrees >= idx:
            max_clique_size = max(max_clique_size, idx+1)
        if min_degrees < max_clique_size:
            break
            
    port_upper_bounds.append((p, max_clique_size + 1))

# Cell 31
port_upper_bounds.sort(key = lambda x: -x[-1])
port_upper_bounds[:5]

# Cell 32
max_port = 0
curr_max_clique = 0
for p, max_clique_upper_bound in port_upper_bounds:
    if curr_max_clique > max_clique_upper_bound: break
    
    internal_edges = internal_edges_all\
        .pipe(lambda x: x[x['port'] == p])\
        .drop_duplicates(['src', 'dst'])
  
    internal_nodes = set(internal_edges.src) | set(internal_edges.dst)
    G = networkx.Graph()
    G.add_nodes_from(internal_nodes)
    for l, r in zip(internal_edges.src, internal_edges.dst):
        G.add_edge(l, r)        
        
    _size = large_clique_size(G) 
    if curr_max_clique < _size:
        curr_max_clique = _size
        max_port = p

# Cell 33
print('Port {} has approx. max clique size {}'.format(max_port, curr_max_clique))
