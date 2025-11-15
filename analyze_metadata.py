import pandas as pd

# Load metadata
df = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

print("="*80)
print("HEK293T METADATA ANALYSIS")
print("="*80)

print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

print("\n" + "="*80)
print("SAMPLE OF DATA (First 20 rows)")
print("="*80)
print(df.head(20).to_string())

print("\n" + "="*80)
print("UNIQUE VALUES PER COLUMN")
print("="*80)
for col in df.columns:
    print(f"{col:20s}: {df[col].nunique():6d} unique values")

print("\n" + "="*80)
print("EXPERIMENT STRUCTURE")
print("="*80)
print("\nExperiment numbers:")
print(df['experiment_no'].value_counts().sort_index())

print("\n" + "="*80)
print("CELL LINE & TREATMENT INFO")
print("="*80)
print("\nCell lines:")
print(df['cell_id'].value_counts())

print("\nPerturbation time (pert_itime):")
print(df['pert_itime'].value_counts())

print("\nPerturbation dose (pert_idose):")
print(df['pert_idose'].value_counts())

print("\n" + "="*80)
print("TREATMENT ANALYSIS")
print("="*80)
print(f"\nTotal unique treatments: {df['treatment'].nunique()}")
print("\nControl/Baseline samples:")
control_samples = df[df['treatment'].isin(['DMSO', 'Blank', 'RNA'])]
print(control_samples['treatment'].value_counts())

print("\n" + "="*80)
print("EXAMPLE: One compound across experiments")
print("="*80)
# Pick a compound that appears multiple times
compound_example = df[df['treatment'].str.startswith('HY_', na=False)]['treatment'].value_counts().head(1).index[0]
print(f"\nExample compound: {compound_example}")
example_data = df[df['treatment'] == compound_example][['experiment_no', 'treatment', 'pert_idose', 'pert_itime', 'sample_plate']]
print(example_data.to_string())
print(f"\nThis compound appears in {len(example_data)} samples")

print("\n" + "="*80)
print("PLATE STRUCTURE")
print("="*80)
print(f"\nNumber of unique plates: {df['sample_plate'].nunique()}")
print("\nPlates per experiment:")
plates_per_exp = df.groupby('experiment_no')['sample_plate'].nunique()
print(plates_per_exp)

print("\n" + "="*80)
print("REPLICATES ANALYSIS")
print("="*80)
# Group by treatment and dose to see replicates
treatment_counts = df.groupby(['treatment', 'pert_idose']).size().reset_index(name='count')
print("\nSample of treatment replicates:")
print(treatment_counts[treatment_counts['treatment'].str.startswith('HY_', na=False)].head(20).to_string())

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"Experiments: {df['experiment_no'].nunique()}")
print(f"Unique compounds/treatments: {df['treatment'].nunique()}")
print(f"Cell line: {df['cell_id'].unique()[0]}")
print(f"Perturbation time: {df['pert_itime'].unique()[0]} hours")
print(f"Dose: {df['pert_idose'].unique()[0]}")

