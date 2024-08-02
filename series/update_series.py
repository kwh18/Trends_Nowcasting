import os
from fredapi import Fred
fred = Fred(api_key='881ee24b7dbe2413187263ce8d4f553e')
import pandas as pd
import numpy as np

path = os.path.dirname(__file__)+'/'

states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# Employment
# US Nonfarm Payrolls (SA Log Diff)
nfp_US = fred.get_series('PAYEMS', '12/1/2003')
nfp_US = pd.DataFrame(nfp_US, columns=['series'])
nfp_US = np.log(nfp_US).diff().dropna()
nfp_US.to_excel(f'{path}/nfp_US.xlsx')

# Every State Nonfarm Payrolls (SA Log Diff)
for state in states:
    nfp = fred.get_series(state+'NA', '12/1/2003')
    nfp = pd.DataFrame(nfp, columns=['series'])
    nfp = np.log(nfp).diff().dropna()
    nfp.to_excel(f'{path}/nfp_US-{state}.xlsx')

# Housing
# New Home Sales (SAAR)
new_home_sales_US = fred.get_series('HSN1F', '12/1/2003')
new_home_sales_US = pd.DataFrame(new_home_sales_US, columns=['series'])
new_home_sales_US.to_excel(f'{path}/new_home_sales_US.xlsx')

# Vehicles
# Total Vehicle Sales (SAAR)
total_vehicle_sales_US = fred.get_series('TOTALSA', '12/1/2003')
total_vehicle_sales_US = pd.DataFrame(total_vehicle_sales_US, columns=['series'])
total_vehicle_sales_US.to_excel(f'{path}/total_vehicle_sales_US.xlsx')