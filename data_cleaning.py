import pandas as pd
import numpy as np


# load data sets
gas = pd.read_excel('C:\\Users\\omarz\\OneDrive\\Desktop\\hackathon\\gas_flows.xlsx')
parameters = pd.read_excel('C:\\Users\\omarz\\OneDrive\\Desktop\\hackathon\\process_parameters.xlsx')
specs = pd.read_excel('C:\\Users\\omarz\\OneDrive\\Desktop\\hackathon\\IOP_specs.xlsx')


# merging data
merged_data = pd.merge(parameters, specs, on='ANALYSIS_TIMESTAMP')
merged_data = pd.merge(merged_data, gas, on='ANALYSIS_TIMESTAMP')

print(merged_data.shape)

#choose only columns with numbers
cols = merged_data.select_dtypes(include=[np.number]).columns
for col in cols:
    if merged_data[col].dtype in [np.float64, np.int64]:  # for numbers
        Q1 = merged_data[col].quantile(0.25)
        Q3 = merged_data[col].quantile(0.75)
        IQR = Q3-Q1
        # Replace outliers with blank
        merged_data[col] = merged_data[col].mask((merged_data[col] > Q3+1.5*IQR), other=Q3+1.5*IQR)
        merged_data[col] = merged_data[col].mask((merged_data[col] < Q1-1.5*IQR), other=Q1-1.5*IQR)


for col in merged_data.columns:
    if merged_data[col].dtype in [np.float64, np.int64]:  # for numbers
        mean = merged_data[col].mean()
        # Replace blanks with mean (statistical imputation)
        merged_data[col].fillna(mean, inplace=True)


merged_data = merged_data.drop('ANALYSIS_TYPE', axis=1)

#remove any column where the mode of column makes up 60% of the data inside it
percentage = (merged_data == merged_data.mode().loc[0]).sum()/len(merged_data)
merged_data = merged_data.loc[:, percentage <= 0.6]
print(merged_data.shape)

#standarize the data
for col in merged_data.columns:
    if (merged_data[col].dtype in [np.float64, np.int64]) & (col != 'C') & (col != 'MET'):
        merged_data[col] = (merged_data[col] - merged_data[col].min()) / (merged_data[col].max() - merged_data[col].min())

print(merged_data.head().to_string())


merged_data.to_csv("C:\\Users\\omarz\\OneDrive\\Desktop\\hackathon\\cleaned_merged_data.csv", index=False)


