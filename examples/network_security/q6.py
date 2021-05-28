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

# Question 6: Malware Controller
# Cell 34
single_dst = df[~df['src_int'] & df['dst_int']]\
    .drop_duplicates(['src', 'dst'])\
    .src.value_counts()\
    .pipe(lambda x: x[x == 1])\
    .index

print('Count of "little reason" src:', len(single_dst))

# Cell 35
df[~df['src_int'] & df['dst_int']]\
    .pipe(lambda x: x[x.src.isin(single_dst)])\
    .drop_duplicates(['src', 'dst'])\
    .groupby('dst').size()\
    .where(lambda x: x > 0).dropna()

# Cell 36
df[~df['src_int'] & df['dst_int']]\
  .pipe(lambda x: x[x.src.isin(single_dst)])\
  .drop_duplicates(['src', 'dst'])\
  .head()
