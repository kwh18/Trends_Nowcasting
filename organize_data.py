import pandas as pd
import numpy as np

states_df = pd.read_excel('states_empty.xlsx', index_col=0)


states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

states_lfp = ['CA', 'MD', 'MN', 'NE', 'NH', 'NM', 'OH', 'OK', 'PA', 'TX', 'UT', 'RI', 'WV']

state_full = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

# for state in states:
#     try:
#         PCS = pd.read_excel(f'pre_employment/US-{state}/top_True_rising_True/2004-01-2023-12/delete_series_with_0s__weighting_none__lags_0/principal_comp_3.xlsx', index_col=0)
#         states_df[f'gt_premp_{state.lower()}_pc1'] = PCS['comp_0']
#         states_df[f'gt_premp_{state.lower()}_pc2'] = PCS['comp_1']
#         states_df[f'gt_premp_{state.lower()}_pc3'] = PCS['comp_2']
#     except FileNotFoundError:
#         PCS = np.nan
    

# states_df.to_excel('L:/kyle/trends_nowcast/trends_db/Trends_Preemployment_states.xlsx')


# for state in states:
#     mini_local = pd.ExcelFile('mini_local.xlsx')
#     vars = pd.read_excel(mini_local, 'vars')
#     factor_spec = pd.read_excel(mini_local, 'factor_spec')
#     options = pd.read_excel(mini_local, 'options')

#     vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', np.nan, np.nan, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, 1]

#     factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})

#     options.iloc[1,1] = f'{state.lower()}na'

#     with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local.xlsx') as writer:
#         vars.to_excel(writer,'vars', index=False)
#         factor_spec.to_excel(writer,'factor_spec', index=False)
#         options.to_excel(writer,'options', index=False)
#     print(f'Finished mini_{state.lower()}na_local setup file')

    
#     mini_local_trends = pd.ExcelFile('mini_local_trends.xlsx')
#     vars = pd.read_excel(mini_local_trends, 'vars')
#     factor_spec = pd.read_excel(mini_local_trends, 'factor_spec')
#     options = pd.read_excel(mini_local_trends, 'options')

#     vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 1', 'Google Trends', f'gt_premp_{state.lower()}_pc1', np.nan, np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 2', 'Google Trends', f'gt_premp_{state.lower()}_pc2', np.nan, np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 3', 'Google Trends', f'gt_premp_{state.lower()}_pc3', np.nan, np.nan, 1, 1, 1, 1]

#     factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})
    
#     options.iloc[1,1] = f'{state.lower()}na'

#     with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local_trends.xlsx') as writer:
#         vars.to_excel(writer,'vars', index=False)
#         factor_spec.to_excel(writer,'factor_spec', index=False)
#         options.to_excel(writer,'options', index=False)
#     print(f'Finished mini_{state.lower()}na_local_trends setup file')

    
#     mini_local_trends = pd.ExcelFile('mini_local_trends.xlsx')
#     vars = pd.read_excel(mini_local_trends, 'vars')
#     factor_spec = pd.read_excel(mini_local_trends, 'factor_spec')
#     options = pd.read_excel(mini_local_trends, 'options')

#     vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, np.nan, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 1', 'Google Trends', f'gt_premp_{state.lower()}_pc1', np.nan, np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 2', 'Google Trends', f'gt_premp_{state.lower()}_pc2', np.nan, np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 3', 'Google Trends', f'gt_premp_{state.lower()}_pc3', np.nan, np.nan, 1, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 1', 'Google Trends', f'gt_premp_us_pc1', np.nan, np.nan, np.nan, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 2', 'Google Trends', f'gt_premp_us_pc2', np.nan, np.nan, np.nan, 1, 1, 1]
#     vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 3', 'Google Trends', f'gt_premp_us_pc3', np.nan, np.nan, np.nan, 1, 1, 1]

#     factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})
    
#     options.iloc[1,1] = f'{state.lower()}na'

#     with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local_trends_all.xlsx') as writer:
#         vars.to_excel(writer,'vars', index=False)
#         factor_spec.to_excel(writer,'factor_spec', index=False)
#         options.to_excel(writer,'options', index=False)
#     print(f'Finished mini_{state.lower()}na_local_trends_all setup file')

for state in states_lfp:
    mini_local = pd.ExcelFile('mini_local.xlsx')
    vars = pd.read_excel(mini_local, 'vars')
    factor_spec = pd.read_excel(mini_local, 'factor_spec')
    options = pd.read_excel(mini_local, 'options')

    vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', 0, 0, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, 1]
    
    factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})

    options.iloc[1,1] = f'{state.lower()}na'

    with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local.xlsx') as writer:
        vars.to_excel(writer,'vars', index=False)
        factor_spec.to_excel(writer,'factor_spec', index=False)
        options.to_excel(writer,'options', index=False)
    print(f'Finished mini_{state.lower()}na_local setup file')

    
    mini_local_trends = pd.ExcelFile('mini_local_trends.xlsx')
    vars = pd.read_excel(mini_local_trends, 'vars')
    factor_spec = pd.read_excel(mini_local_trends, 'factor_spec')
    options = pd.read_excel(mini_local_trends, 'options')

    vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', 0, 0, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 1', 'Google Trends', f'gt_premp_{state.lower()}_pc1', np.nan, np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 2', 'Google Trends', f'gt_premp_{state.lower()}_pc2', np.nan, np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 3', 'Google Trends', f'gt_premp_{state.lower()}_pc3', np.nan, np.nan, 1, 1, 1, 1]

    factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})
    
    options.iloc[1,1] = f'{state.lower()}na'

    with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local_trends.xlsx') as writer:
        vars.to_excel(writer,'vars', index=False)
        factor_spec.to_excel(writer,'factor_spec', index=False)
        options.to_excel(writer,'options', index=False)
    print(f'Finished mini_{state.lower()}na_local_trends setup file')

    
    mini_local_trends = pd.ExcelFile('mini_local_trends.xlsx')
    vars = pd.read_excel(mini_local_trends, 'vars')
    factor_spec = pd.read_excel(mini_local_trends, 'factor_spec')
    options = pd.read_excel(mini_local_trends, 'options')

    vars.loc[len(vars.index)] = [f'All Employees: Total Nonfarm in {state_full[state]}', f'{state} payroll employment', f'{state.lower()}na', np.nan, 1, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Initial Claims in {state_full[state]}', f'{state} claims', f'{state.lower()}iclaims', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Unemployment Rate in {state_full[state]}', f'{state} household employment', f'{state.lower()}ur', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Labor Force Participation for {state_full[state]}', f'{state} payroll employment', f'lfp{state.lower()}', 0, 0, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Total Personal Income in {state_full[state]}', f'{state} Income', f'{state.lower()}otot', np.nan, np.nan, 1, 1, np.nan, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 1', 'Google Trends', f'gt_premp_{state.lower()}_pc1', np.nan, np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 2', 'Google Trends', f'gt_premp_{state.lower()}_pc2', np.nan, np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - {state} - Principal Component 3', 'Google Trends', f'gt_premp_{state.lower()}_pc3', np.nan, np.nan, 1, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 1', 'Google Trends', f'gt_premp_us_pc1', np.nan, np.nan, np.nan, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 2', 'Google Trends', f'gt_premp_us_pc2', np.nan, np.nan, np.nan, 1, 1, 1]
    vars.loc[len(vars.index)] = [f'Google Trends - Pre-Employment - US - Principal Component 3', 'Google Trends', f'gt_premp_us_pc3', np.nan, np.nan, np.nan, 1, 1, 1]

    factor_spec = factor_spec.rename(columns={'Unnamed: 0':np.nan})
    
    options.iloc[1,1] = f'{state.lower()}na'

    with pd.ExcelWriter(f'L:/kyle/trends_nowcast/example/local_setups/mini_{state.lower()}na_local_trends_all.xlsx') as writer:
        vars.to_excel(writer,'vars', index=False)
        factor_spec.to_excel(writer,'factor_spec', index=False)
        options.to_excel(writer,'options', index=False)
    print(f'Finished mini_{state.lower()}na_local_trends_all setup file')