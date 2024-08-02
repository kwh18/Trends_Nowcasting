from apiclient.discovery import build

import os
import time
from statsmodels.tsa.seasonal import STL
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from os import path
import pandas as pd
import numpy as np

path = os.path.dirname(__file__)+'/'

SERVER = 'https://trends.googleapis.com'

API_VERSION = 'v1beta'
DISCOVERY_URL_SUFFIX = '/$discovery/rest?version=' + API_VERSION
DISCOVERY_URL = SERVER + DISCOVERY_URL_SUFFIX


def connect():
  service = build('trends', 'v1beta',
                  developerKey='INSERT GOOGLE API HERE',
                  discoveryServiceUrl=DISCOVERY_URL)
  return service


def get_data(new_pull:bool=True,
             include_top:bool=False,
             include_rising:bool=False,
             weight_option:str='none',
             lags_to_include:int=0,
             terms:list=[],
             parent_folder:str='',
             start_date:str='2004-01',
             end_date:str='',
             geo_codes:list=['US'],
             series:str='',
             n_princ_comp:int=3,
             deal_with_0:str='delete_series'):

  if new_pull:
    # Connects to Trends API
    service = connect()

    for geo_code in geo_codes:
      top_terms=[]
      rising_terms=[]
      if include_top:
        print(f'Pulling top terms for {geo_code}')
        top_terms=get_top_queries(service, terms, geo_code, start_date, end_date, keep_top=6)
      if include_rising:
        print(f'Pulling rising terms for {geo_code}')
        rising_terms=get_rising_queries(service, terms, geo_code, start_date, end_date, keep_top=6)
      full_terms = list(set(terms + top_terms + rising_terms))

      # Make separate folder for each geo_code
      if not os.path.exists(f'{path}/{parent_folder}/{geo_code}/top_{str(include_top)}_rising_{str(include_rising)}/{start_date}-{end_date}'):
        os.makedirs(f'{path}/{parent_folder}/{geo_code}/top_{str(include_top)}_rising_{str(include_rising)}/{start_date}-{end_date}')
      data_folder = path+'/'+parent_folder+'/'+geo_code+'/top_'+str(include_top)+'_rising_'+str(include_rising)+'/'+start_date+'-'+end_date
        
        # Places raw output file into folder
      print(f'Pulling time series for {geo_code} with include_top={include_top} and include_rising={include_rising}. Full set of terms: {full_terms}')
      get_time_series(service,
                      full_terms,
                      start_date,
                      end_date,
                      geo_code,
                      data_folder)
        
  for geo_code in geo_codes:
    if not os.path.exists(f'{path}/{parent_folder}/{geo_code}/top_{str(include_top)}_rising_{str(include_rising)}/{start_date}-{end_date}/{deal_with_0}_with_0s__weighting_{weight_option}__lags_{str(lags_to_include)}'):
      os.makedirs(f'{path}/{parent_folder}/{geo_code}/top_{str(include_top)}_rising_{str(include_rising)}/{start_date}-{end_date}/{deal_with_0}_with_0s__weighting_{weight_option}__lags_{str(lags_to_include)}')
    data_folder = path+'/'+parent_folder+'/'+geo_code+'/top_'+str(include_top)+'_rising_'+str(include_rising)+'/'+start_date+'-'+end_date
    component_folder = path+'/'+parent_folder+'/'+geo_code+'/top_'+str(include_top)+'_rising_'+str(include_rising)+'/'+start_date+'-'+end_date+'/'+deal_with_0+'_with_0s__weighting_'+weight_option+'__lags_'+str(lags_to_include)
    # Deals with 0 values in dataframe
    if deal_with_0=='delete_series':
      # delete_series deletes all keywords that contain any 0 value
      dealing_with_0(data_folder, component_folder, option="delete_series")
      seasonal_adjustment(component_folder, option="resid")
    elif deal_with_0=='create_NaN':
      # create_NaN replaces all 0 values with NaN
      dealing_with_0(data_folder, component_folder, option="create_NaN")
      seasonal_adjustment(component_folder, option="YoY")
    elif deal_with_0=='interpolate':
      # create_NaN replaces all 0 values with NaN
      dealing_with_0(data_folder, component_folder, option="interpolate")
      seasonal_adjustment(component_folder, option="resid")
    else:
      raise ValueError('Issue with deal_with_0 option')
              
    # Return principal components of data
    principal_components(component_folder, geo_code, n_princ_comp, weight_option, series)


def get_time_series(service, terms, start_date, end_date, geo_code, data_folder):

  full_series = pd.DataFrame(columns=terms)

  for term in terms:
    response = service.getGraph(terms=term, restrictions_startDate=start_date, restrictions_endDate=end_date, restrictions_geo = geo_code).execute()
    series = response['lines'][0]['points']
    series = pd.DataFrame(series)
    series['date'] == pd.to_datetime(series['date'])
    series.set_index('date', inplace=True)
    full_series[term]=series['value']

    time.sleep(0.5)
  
  full_series.to_excel(f'{data_folder}/raw_data.xlsx')

  cleaned_series = full_series.drop(columns = full_series.columns[full_series.eq(0).mean()>0.1])
  thresh = 6
  to_drop = cleaned_series.diff().eq(0).rolling(thresh).sum().eq(thresh).any()
  cleaned_series = cleaned_series.loc[:, ~to_drop]

  cleaned_series += 1

  print(f'Terms left after cleaning: {cleaned_series.columns}')

  cleaned_series.to_excel(f'{data_folder}/cleaned_data.xlsx')




def get_top_queries(service, terms, geo_code, start_date, end_date, keep_top):
  top_list = []

  for term in terms:
    # Top queries, no restrictions.
    response = service.getTopQueries(term=term, restrictions_startDate=start_date, restrictions_endDate=end_date, restrictions_geo = geo_code).execute()
    if response:
      list = response['item']
      top = [sub['title'] for sub in list]
      top = top[0:keep_top-1]
      top_list = top_list + top
      print(f'Top terms for {term} in {geo_code}: {top}')
    else:
      print(f'Cannot find top terms for {term} in {geo_code}')

    time.sleep(0.5)
  
  return top_list


def get_rising_queries(service, terms, geo_code, start_date, end_date, keep_top):
  rising_list = []

  for term in terms:
    # Top queries, no restrictions.
    response = service.getRisingQueries(term=term, restrictions_startDate=start_date, restrictions_endDate=end_date, restrictions_geo = geo_code).execute()
    if response:
      list = response['item']
      rising = [sub['title'] for sub in list]
      rising = rising[0:keep_top-1]
      rising_list = rising_list + rising
      print(f'Rising terms for {term} in {geo_code}: {rising}')
    else:
      print(f'Cannot find rising terms for {term} in {geo_code}')
    time.sleep(0.5)

  return rising_list


def dealing_with_0(data_folder, component_folder, option):
  pruned_df = pd.read_excel(f'{data_folder}/raw_data.xlsx', index_col="date")
  if option == "delete_series":
    # Deletes any series that contains a 0
    pruned_df = pruned_df.loc[:, (pruned_df != 0).all(axis=0)]

  elif option == "create_NaN" or option == 'interpolate':
    # Replaces all 0s in sheet with NaN
    pruned_df = pruned_df.loc[:, (pruned_df != 0).any(axis=0)]
    pruned_df.replace(0, np.nan, inplace=True)

  elif option == "none":
    pruned_df

  else:
    print("Option for dealing with zeros not selected. Defaulting to 'none'")
  
  if option == 'interpolate':
    pruned_df = pruned_df.interpolate().fillna(method='bfill')
            
  pruned_df.to_excel(f'{component_folder}/pruned_data.xlsx')


def seasonal_adjustment(component_folder, option):
  adjusted_df = pd.read_excel(f'{component_folder}/pruned_data.xlsx', index_col="date")
  for column in adjusted_df:
    if option == "YoY":
      # Differences data by 12 months. Necessary for series with NaNs
      adjusted_df[column] = adjusted_df[column].diff(periods=12)
    else:
      adjusted_df.dropna(axis=0,how='any', inplace=True) # Drops rows that have any NaN
      stl = STL(adjusted_df[column],robust=True, period=12).fit()

    if option == "deseason":
      # Removes seasonal component of data
      adjusted_df[column] = adjusted_df[column] - stl.seasonal

    elif option == "resid":
      # Removes seasonal and trend components
      adjusted_df[column] = stl.resid

    elif option == "trend":
      # Removes seasonal and residual components
      adjusted_df[column] = stl.trend

    adjusted_df.to_excel(f'{component_folder}/seasonally_adjusted_data.xlsx')


def principal_components(component_folder, geo_code, n_comp, weight_option, series):
  forecast_df = pd.read_excel(f'{component_folder}/seasonally_adjusted_data.xlsx', index_col="date")
  n_comp = min(n_comp, len(forecast_df.columns))
  no_problem = True

  if not weight_option=='none':
    if series == 'big3':
      forecast_df['series'] = pd.read_excel(f'{path}/series/big3_dom_sales_pct_yoy.xlsx', index_col=0)
    elif series == 'new_home_sales':
      forecast_df['series'] = pd.read_excel(f'{path}/series/new_home_sales_US.xlsx', index_col=0)
    elif series == 'nfp':
      forecast_df['series'] = pd.read_excel(f'{path}/series/nfp_{geo_code}.xlsx', index_col=0)
    elif series == 'tot_veh_sales':
      forecast_df['series'] = pd.read_excel(f'{path}/series/total_vehicle_sales_US.xlsx', index_col=0)
    else:
      raise ValueError('Not a recognized series name, cannot weight on info')


    forecast_df.dropna(axis=0,how='all', inplace=True) # Drops rows that are all NaN
    forecast_df.dropna(axis='columns',thresh=n_comp+1, inplace=True) # Drops columns with too many NaNs for PCA
    forecast_df = forecast_df.dropna(subset=['series'])
    n_comp = min(n_comp, len(forecast_df.columns))
    series_df = forecast_df.copy()['series']
    forecast_df.drop(['series'], axis=1, inplace=True)
    ridge_X = forecast_df.copy().shift(1)
    ridge_X.dropna(axis=0, how='all', inplace=True)
    ridge_y = series_df.copy().tail(-1)
                
    if weight_option == 'ridge':
      rid = Ridge()
      if len(ridge_X) == len(ridge_y) and len(ridge_X)>0:
        rid.fit(ridge_X, ridge_y)
        pca_weights = rid.coef_
      else:
        no_problem = False
        pca_weights=[0]*len(forecast_df.columns)
                
    elif weight_option == 'llf':
      pca_weights=[]
      if len(forecast_df.columns)>0 and len(forecast_df)>0:
        for column in forecast_df:
          keyword_series = forecast_df[column].copy().dropna(axis=0, how='any', inplace=True)
          model = ARIMA(series_df, exog=keyword_series, order=(1,0,0))
          res = model.fit()
          pca_weights.append(res.llf)
          model = ARIMA(series_df, order=(1,0,0))
          res = model.fit()
          pca_weights = [x - res.llf for x in pca_weights]
      else:
        no_problem = False
        pca_weights=[0]*len(forecast_df.columns)
        

    else:
      raise ValueError('Not a recognized weight_option')            

    i=0
    for column in forecast_df:
      if pca_weights[i]<=0:
        forecast_df = forecast_df.drop(column, axis=1)
      i+=1
    if weight_option == 'ridge':
      pca_weights = [x for x in pca_weights if x>0]
    elif weight_option == 'llf':
      pca_weights = [x for x in pca_weights if x>0]

    if len(forecast_df.columns)>0 and len(forecast_df)>0:
      stand_pca = PCA(forecast_df, ncomp=n_comp, method='eig',missing = 'fill-em', standardize=True, normalize=False, weights=pca_weights)
      princip_comp = stand_pca.factors
    else:
      no_problem = False


  else:
    forecast_df.dropna(axis=0,how='all', inplace=True) # Drops rows that are all NaN
    forecast_df.dropna(axis='columns',thresh=n_comp+1, inplace=True) # Drops columns with too many NaNs for PCA
    forecast_df.dropna(axis=0,thresh=min(n_comp,len(forecast_df.columns))+1, inplace=True) # Drops rows with too many NaNs for PCA

    n_comp = min(n_comp, len(forecast_df.columns))

    if len(forecast_df.columns)>0 and len(forecast_df)>0:
      stand_pca = PCA(forecast_df, ncomp=n_comp, method='eig',missing = 'fill-em', standardize=True, normalize=False)
      princip_comp = stand_pca.factors
    else:
      no_problem = False

  if no_problem:          
    princip_comp.to_excel(f'{component_folder}/principal_comp_{n_comp}.xlsx')
  else:
    print(f'Problem with principal components for component folder = {component_folder}')



if __name__ == '__main__':


  pre_employment_terms = ['jobs', 'government jobs', 'part time jobs', 'online jobs',
                       'career', 'top jobs', 'jobs hiring', 'job', 'job search', 'job vacancies']

  after_employment_terms = ['rollover 401k','roll 401k IRA', 'COBRA coverage',
                         'qualifying life event', 'fsa', 'moving tax deductible','movers', 'paycheck calculator']
  
  homebuying_terms = ['realtor','zillow','redfin','home appraisal','home inspector','real estate agent',
                    'mortage application','downpayment','heloc','refinance mortgage','downpayment assistance', 'fha loan', 'buyers agent',
                    'earnest money','escrow mortgage']

  vehicle_terms = ['buying a car', 'buying a vehicle', 'car loan', 'used car', 'car title', 'vehicle title', 'carmax']

  big3 = ['ford', 'ford dealership', 'chevy', 'chevy dealership',
        'buick', 'buick dealership', 'cadillac', 'cadillac dealership',
        'gmc', 'gmc dealership', 'chrysler', 'chrysler dealership',
        'dodge', 'dodge dealership', 'jeep', 'jeep dealership']
  
  # get_data(new_pull = False,
  #         include_top = True,
  #         include_rising = False,
  #         #weight_option = 'ridge',
  #         #weight_option = 'llf',
  #         #lags_to_include = 3,
  #         #lags_to_include = 6,
  #         #lags_to_include = 12,
  #         terms = pre_employment_terms,
  #         parent_folder = 'pre_employment',
  #         #start_date = '2004-01',
  #         end_date = '2023-07',
  #         geo_codes = ['US','US-MI'],
  #         # geo_codes = ['US', 'US-MI', 'US-AL', 'US-AK', 'US-AZ', 'US-AR', 'US-CA', 'US-CO', 'US-CT', 'US-DE', 'US-DC', 'US-FL', 'US-GA', 'US-HI',
  #         #               'US-ID', 'US-IL', 'US-IN', 'US-IA', 'US-KS', 'US-KY', 'US-LA', 'US-ME', 'US-MD', 'US-MA', 'US-MN', 'US-MS', 'US-MO', 'US-MT',
  #         #               'US-NE', 'US-NV', 'US-NH', 'US-NJ', 'US-NM', 'US-NY', 'US-NC', 'US-ND', 'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI',
  #         #               'US-SC', 'US-SD', 'US-TN', 'US-TX', 'US-UT', 'US-VT', 'US-VA', 'US-WA', 'US-WV', 'US-WI', 'US-WY'],       
  #         series = 'nfp',
  #         #series = 'new_home_sales',
  #         #series = 'big3',
  #         #series = 'tot_veh_sales',
  #         #n_princ_comp = 8,
  #         #deal_with_0 = 'create_NaN',
  #         deal_with_0 = 'interpolate',
  #       )
  

  get_data(new_pull = True,
          include_top = True,
          include_rising = True,
          weight_option = 'none',
          lags_to_include = 0,
          terms = pre_employment_terms,
          parent_folder = 'pre_employment',
          end_date = '2023-12',
          geo_codes = ['US', 'US-MI', 'US-AL', 'US-AK', 'US-AZ', 'US-AR', 'US-CA', 'US-CO', 'US-CT', 'US-DE', 'US-DC', 'US-FL', 'US-GA',
                        'US-HI', 'US-ID', 'US-IL', 'US-IN', 'US-IA', 'US-KS', 'US-KY', 'US-LA', 'US-ME', 'US-MD', 'US-MA', 'US-MN', 'US-MS', 'US-MO', 'US-MT',
                        'US-NE', 'US-NV', 'US-NH', 'US-NJ', 'US-NM', 'US-NY', 'US-NC', 'US-ND', 'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI',
                        'US-SC', 'US-SD', 'US-TN', 'US-TX', 'US-UT', 'US-VT', 'US-VA', 'US-WA', 'US-WV', 'US-WI', 'US-WY'],   
          # geo_codes = ['US', 'US-MI', 'US-CA', 'US-LA', 'US-NC', 'US-UT'],       
          series = 'nfp',
          deal_with_0 = 'delete_series',
        )

