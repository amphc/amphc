# Preprocessing
# Cell 1
import pandas as pd 
import matplotlib.pyplot as plt
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

# Question 8: Botnet Inside
# Cell 40
periodic_callbacks = df[df['src_int'] & ~df['dst_int']]\
  .drop_duplicates(['dst', 'minute'])\
  .groupby('dst').size()\
  .pipe(lambda x: x[(x > 0) & (x <= 4)])\
  .sort_values()

periodic_callbacks

# Cell 41
fig, (ax_l, ax_r) = plt.subplots(ncols=2, sharey=True, figsize=(12,6))

df[df.dst.isin(periodic_callbacks.index)]\
    .set_index('ts')\
    .resample('Min').size()\
    .plot(title='Connections over time to C&C(min interval)', ax=ax_l)

df[df.dst == '14.53.122.55']\
    .set_index('ts')\
    .resample('Min').size()\
    .plot(title='Connections over time to 14.53.122.55 (benign)', ax=ax_r)

# Cell 43
df[~df['dst_int']]\
    .groupby('dst')\
    .bytes.std()\
    .sort_values()\
    .head(10)

# Cell 44
df[~df['dst_int']]\
    .groupby('port').size()\
    .sort_values()\
    .head(10)

# Cell 45
df.loc[
    df.dst.isin(periodic_callbacks.index),
    ['src', 'dst', 'bytes']
].head()

# Cell 46
df[df.dst.isin(periodic_callbacks.index)]\
    .ts.diff()\
    .dt.total_seconds()\
    .plot.hist(bins=50)
