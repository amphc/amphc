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

# Answers until Q9
blacklist_ips = ['13.37.84.125', '12.55.77.96', '12.30.96.87', '14.45.67.46', '14.51.84.50', '13.42.70.40']

# Question 10: Lateral Spy
# Cell 50
int_df = df[df['src_int'] & df['dst_int']]\
    .pipe(lambda x: x[~x.src.isin(blacklist_ips)])\
    .drop_duplicates(('src', 'dst', 'port'))

# Cell 51
print('Unique dsts')
int_df\
  .drop_duplicates(['src', 'dst'])\
  .groupby('src').size()\
  .sort_values(ascending=False).head()

# Cell 52
print('Unique ports')
int_df\
  .drop_duplicates(['src', 'port'])\
  .groupby('src').size()\
  .sort_values(ascending=False).head()

# Cell 53
dst_port_df = int_df\
    .groupby(['dst', 'port'])\
    .src.apply(list).dropna()

dst_port_df.sample(10)

# Cell 54
dst_port_df.pipe(lambda x: x[x.map(len) == 1])\
    .to_frame().reset_index()\
    .explode('src')\
    .src.value_counts()
