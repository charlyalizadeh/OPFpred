import pandas as pd

decompositions = pd.read_csv('./data/decompositions.csv')
instances = pd.read_csv('./data/instances.csv')
instances = instances[['instance_name', 'graph.degree_max', 'graph.degree_mean',
                       'graph.degree_min', 'graph.degree_var',
                       'graph.density', 'graph.diameter',
                       'graph.global_clustering_coefficient', 'graph.ne', 'graph.nv',
                       'loads_imag.max', 'loads_imag.mean',
                       'loads_imag.median', 'loads_imag.min', 'loads_imag.var',
                       'loads_real.max', 'loads_real.mean', 'loads_real.median',
                       'loads_real.min', 'loads_real.var']]

# Add target
df_cholesky = decompositions[(decompositions['nb_added_edges'] == 0) & (decompositions['dec_type'] == 'cholesky')]
choleskytimes = dict(zip(df_cholesky['instance_name'], df_cholesky['solver.solving_time']))
decompositions['target'] = decompositions.apply(lambda row: row['solver.cholesky_diff'] / choleskytimes[row['instance_name']], axis=1)

# Merge instances and decompositions
df_merged = pd.merge(decompositions, instances, on='instance_name')
df_merged['merge_treshold'].fillna(-1, inplace=True)
df_merged['origin_nb_added_edges'].fillna(-1, inplace=True)
df_merged['origin_dec_type'].fillna("", inplace=True)
df_merged_clean_OPF = df_merged.dropna(axis=1)
column_removed = list(set(df_merged.columns) - set(df_merged_clean_OPF.columns))
print(f'Number of columns removed: {len(column_removed)}')
print(f'Number of keeped columns: {len(df_merged_clean_OPF.columns)}')

# Remove column with only one value
for c in df_merged_clean_OPF.columns:
    if len(df_merged_clean_OPF[c].unique()) == 1:
        df_merged_clean_OPF.drop(c, inplace=True, axis=1)

# Remove not used features
rm_columns = [
    'oid',
    'nb_lc',
    'solver.objective',
    'solver.cholesky_diff',
    'category'
]
df_merged_clean_OPF.drop(rm_columns, axis=1, inplace=True, errors='ignore')
for c in df_merged_clean_OPF.columns:
    if c.startswith('kernel'):
        df_merged_clean_OPF.drop(c, inplace=True, axis=1)

df_merged_clean_OPF.to_csv('./data/OPFDataset.csv', index=False)
