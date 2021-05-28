# Preprocessing
# Cell 1
import pandas as pd 
import numpy as np
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

# Answers until Q2
blacklist_ips = ['13.37.84.125', '12.55.77.96']

# Question 3: Discover Data Exfiltration 3
# Cell 16
src_port_bytes_df = df[
        ~df['src'].isin(blacklist_ips)     # Not including previous answers
        & df['src_int'] & ~df['dst_int']   # Outbound
    ].groupby(['src', 'port'])\
        .bytes.sum()\
        .reset_index()

ports = src_port_bytes_df['port'].unique()
print('Number of unique ports:', len(ports))

# Cell 17
src_port_bytes_df[src_port_bytes_df.port == 113]

# Cell 18
src_port_bytes_df.groupby('port')\
    .bytes.sum()\
    .sort_values(ascending=False)\
    .plot.bar(figsize=(16,4), rot=0, title="Outbound bytes per port")\
    .set_ylabel('Total outbound bytes')

# Cell 19
fig, axs = plt.subplots(ncols=3, nrows=3, sharey=True, figsize=(12,6))

for idx, p in enumerate(src_port_bytes_df.port.head(9)):
    src_port_bytes_df[src_port_bytes_df.port == p]\
        .bytes.plot.hist(title='Distribution for port {}'.format(p), ax = axs[idx % 3][idx // 3])\
        .set_xlabel('total outbound bytes')

    plt.tight_layout()

# Cell 20
src_port_bytes_df\
  .groupby('port')\
  .apply(lambda x: np.max((x.bytes - x.bytes.mean()) / x.bytes.std()))\
  .sort_values(ascending=True)\
  .tail(10)\
  .plot.barh(title='Top z-score value per port')\
  .set_xlabel('Max z-score')

# Cell 21
src_124 = src_port_bytes_df\
  .pipe(lambda x: x[x['port'] == 124])\
  .sort_values('bytes', ascending=False).head(1)

src_124

# Cell 22
ax = src_port_bytes_df[src_port_bytes_df.port == 124]\
    .bytes.plot.hist(bins=50, title='Distribution of outbound data usage for port 124')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_124.iloc[0, 2], linestyle='--')
plt.text(src_124.iloc[0, 2], 100, '12.30.96.87', rotation=90, horizontalalignment='right')
