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

# Answers until Q3
blacklist_ips = ['13.37.84.125', '12.55.77.96', '12.30.96.87']

# Question 4: Private C&C channel
# Cell 24
df[~df['src_int']]\
  .drop_duplicates(('src', 'port'))\
  .groupby('port').size()\
  .sort_values()\
  .head()

# Cell 25
df[~df['src_int'] & (df['port'] == 113)][['src', 'dst', 'port']]

# Cell 26
df[(df['src'] == '15.104.76.58') & (df['dst'] == '14.47.74.88')]\
    [['src', 'dst', 'port']]
