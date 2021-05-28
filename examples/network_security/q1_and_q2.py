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

# Question 1: Discover Data Exfiltration 1
# Cell 6
src_bytes_out = df[df['src_int'] & ~df['dst_int']]\
  .groupby('src')\
  .bytes.sum()\
  .pipe(lambda x: x[x > 0])\
  .sort_values(ascending=False)

src_bytes_out.to_frame().head()

# Cell 7
src_bytes_out.head(10)\
    .sort_values()\
    .plot.barh(title='Top 10 high outbound traffic srcs')\
    .set_xlabel('total outbound bytes')

# Cell 8
ax = src_bytes_out\
  .plot.hist(bins=50, title='Outbound traffic per src')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_bytes_out.iloc[0], linestyle='--')
plt.text(src_bytes_out.iloc[0], 100, '13.37.84.125', rotation=90, horizontalalignment='right')

# Answers until Q1
blacklist_ips = ['13.37.84.125']

# Question 2: Discover Data Exfiltration 2
# Cell 10
df.groupby('hour').size()\
  .plot.bar(title='Activity per hour')\
  .set_ylabel('Connection counts')

# Cell 11
off_hours_activity = df[
    ~df['src'].isin(blacklist_ips)          # Not including previous answers
    & df['src_int'] & ~df['dst_int']        # Outbound
    & (df['hour'] >= 0) & (df['hour'] < 16) # Off hours
].groupby('src')\
  .bytes.sum()\
  .sort_values(ascending=False)\
  .where(lambda x: x > 0)

off_hours_activity.head()

# Cell 12
off_hours_activity.head(10)\
    .sort_values()\
    .plot.barh(title='Top 10 off hours high outbound traffic srcs')\
    .set_xlabel('total outbound bytes')

# Cell 13
ax = off_hours_activity.plot.hist(bins=50, title='Off hours outbound traffic')
ax.set_xlabel('total outbound bytes')
_ = ax.axvline(off_hours_activity.iloc[0], linestyle='--')
plt.text(off_hours_activity.iloc[0], 40, '12.55.77.96', rotation=90, horizontalalignment='right')

# Cell 14
ax = src_bytes_out\
  .plot.hist(bins=50, title='Outbound traffic per src')

ax.set_xlabel('total outbound bytes')
_ = ax.axvline(src_bytes_out.loc['12.55.77.96'], color='k', linestyle='--')
plt.text(src_bytes_out.loc['12.55.77.96'], 100, '12.55.77.96', rotation=90, horizontalalignment='right')