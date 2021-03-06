def forecast_pib():
    
    import pandas as pd
    import numpy as np
    import requests
    import json
    import time
    from prophet import Prophet
    from keys import token_inegi, token_banxico
    
    ####################################################################
    #### This file is used to check which is the last recorded quarter 
    #### from previous forecasts. If a new record is observed, then the
    #### file is updated and everything gets re-forecasted.
    ####################################################################

    print('00 - Checking last INEGI records')
    print('...')

    #Import INEGI Last Values file
    lv= pd.read_csv('INEGI_LastValues.csv')

    #Last records
    last_year_record= int(lv[(lv['Var'] == 'PIB') & (lv['timestamp'] == lv['timestamp'].max())]['Y'])
    last_quarter_record= int(lv[(lv['Var'] == 'PIB') & (lv['timestamp'] == lv['timestamp'].max())]['Q'])

    ####################################################################
    #### Query INEGI's API to get latest macroeconomic data for GDP, 
    #### industrial GDP and Investment. If new data is identified,
    #### leave a new record in INEGI_LastValues.csv
    ####################################################################

    #Query
    query= f'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/6207061899/es/0700/false/BISE/2.0/{token_inegi}?type=json'
    response= requests.get(query)

    #Transform JSON to dataframe
    dat= pd.json_normalize(response.json()['Series'][0]['OBSERVATIONS']).rename(columns= {'OBS_VALUE': 'pib'})

    dat['y']= pd.to_numeric([dat['TIME_PERIOD'][i].split('/')[0] for i in range(0, len(dat))])
    dat['q']= pd.to_numeric([dat['TIME_PERIOD'][i].split('/')[1] for i in range(0, len(dat))])
    dat['pib']= pd.to_numeric(dat['pib'])
    dat['yq']= [str(dat['y'][i])+'Q'+str(dat['q'][i]) for i in range(0, len(dat))]

    #Check that last observation is the one registered 
    #If new observations are published, then update INEGI_lastValues.py file

    y_check= dat.iloc[-1, -3] == last_year_record #Year check
    q_check= dat.iloc[-1, -2] == last_quarter_record #Quarter check

    #Validation and updates
    if (y_check == True) & (q_check == True):
        print('... Last YQ has not changed. Forecasting with last registered values')
        
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]

    elif (y_check == True) & (q_check == False):
        print('...Year is same, but new Q data published. Forecasting with new quarter data')
        print('Updating INEGI_LastValues.csv file!')
        
        lv= lv.append({'Var': 'PIB', 
                    'Y': dat.iloc[-1, -3], 
                    'Q': dat.iloc[-1, -2], 
                    'timestamp': time.time()}, ignore_index= True)
        lv.to_csv('INEGI_LastValues.csv', index= False)
            
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]
            
    elif (y_check == False):
        print('...New year, new data published. Forecasting with new quarter data')
        print('Updating INEGI_LastValues.csv file!')
        
        lv= lv.append({'Var': 'PIB', 
                    'Y': dat.iloc[-1, -3], 
                    'Q': dat.iloc[-1, -2], 
                    'timestamp': time.time()}, ignore_index= True)
        lv.to_csv('INEGI_LastValues.csv', index= False)
            
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]
            
    ##Filter selected columns
    dat= dat[['yq', 'pib']]

    ####################################################################
    #### Facebook Prophet model to forecast automatically GDP growth
    #### based on new historic recors , towards 2032
    ####################################################################

    print('01 - forecasting INEGI records...')

    #Prepare data
    ts= dat[['yq', 'pib']].rename(columns= {'yq': 'ds',  'pib': 'y'})
    ts['ds'] = pd.PeriodIndex(ts['ds'], freq='Q').to_timestamp()

    #Fit Model
    m= Prophet(seasonality_mode='multiplicative')
    m.fit(ts)

    #Forecast period
    #Remaining quarters of the year to be forecasted
    if pib_lastValue[1] == 1:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q2', 'Q3', 'Q4']]
    elif pib_lastValue[1] == 2:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q3', 'Q4']]
    elif pib_lastValue[1] == 3:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q4']]
    else:
        remaining_qs = []

    #Next years to be forecasted
    future_ds= np.append(remaining_qs, 
                        [str(y)+q for y in range(pib_lastValue[0] + 1, 2033) for q in ['Q1', 'Q2', 'Q3', 'Q4']])

    #Forecast to 2023
    future= pd.DataFrame(columns=['ds'])
    future['ds']= pd.PeriodIndex(future_ds, freq='Q').to_timestamp()
    forecast = m.predict(future)

    #Integrate historic w/ forecast dataframes
    fnl= pd.DataFrame(columns= ['ds', 'y', 'yhat'])
    fnl['ds'] = pd.PeriodIndex(ts['ds'], freq='Q')
    fnl['y'] = ts['y']

    forecast['y'] = np.nan
    forecast['ds'] = pd.PeriodIndex(forecast['ds'], freq='Q')
    f= forecast[['ds', 'y', 'yhat']]

    fnl= pd.concat([fnl, f]).reset_index(drop= True)
    fnl['year'] = pd.PeriodIndex(fnl['ds'], freq='Q').year
    fnl['quarter'] = pd.PeriodIndex(fnl['ds'], freq='Q').quarter

    index_cut= fnl[(fnl['year'] == pib_lastValue[0]) & (fnl['quarter'] == pib_lastValue[1])].index[0]
    fnl['full'] = np.append(fnl.iloc[:index_cut+1, 1], 
                            fnl.iloc[index_cut+1:, 2])

    ####################################################################
    #### Banxico API query to retrieve latest records of macroeconomic  
    #### expectations survey.
    ####################################################################

    print('02 - Getting BANXICOs latest macro expectations survey data')

    #Proyecci??n a??o en curso
    url= 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SR14448/datos/oportuno'
    response= requests.get(url, headers= {'Bmx-Token': token_banxico})
    proy_a??o_en_curso= pd.to_numeric(response.json()['bmx']['series'][0]['datos'][0]['dato']) / 100
    a??o= int(response.json()['bmx']['series'][0]['datos'][0]['fecha'][-4:])

    #Proyecci??n pr??ximo a??o
    url= 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SR14455/datos/oportuno'
    response= requests.get(url, headers= {'Bmx-Token': token_banxico})
    proy_a??o_proximo= pd.to_numeric(response.json()['bmx']['series'][0]['datos'][0]['dato'])/ 100

    #Proyecci??n en 2 a??os
    url= 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SR14462/datos/oportuno'
    response= requests.get(url, headers= {'Bmx-Token': token_banxico})
    proy_en_2_a??os= pd.to_numeric(response.json()['bmx']['series'][0]['datos'][0]['dato'])/ 100

    #Proyecci??n d??cada
    url= 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SR14469/datos/oportuno'
    response= requests.get(url, headers= {'Bmx-Token': token_banxico})
    proy_decada= pd.to_numeric(response.json()['bmx']['series'][0]['datos'][0]['dato'])/ 100

    ####################################################################
    #### Ajustar valores de proyecciones PIB a expectativas del sector
    #### de Banxico
    ####################################################################

    print('03 - Adjusting naive forecasts to BANXICOs data')

    ################
    ### AJUSTAR 2022
    X = (proy_a??o_en_curso + 1) * fnl.loc[fnl['year'] == a??o-1, 'full'].sum() / fnl.loc[fnl['year'] == a??o, 'full'].sum()
    fnl.loc[fnl['year'] == a??o, 'full'] = fnl.loc[fnl['year'] == a??o, 'full'] * X

    ################
    ### AJUSTAR 2023
    X = (proy_a??o_proximo + 1) * fnl.loc[fnl['year'] == a??o, 'full'].sum() / fnl.loc[fnl['year'] == a??o+1, 'full'].sum()
    fnl.loc[fnl['year'] == a??o+1, 'full'] = fnl.loc[fnl['year'] == a??o+1, 'full'] * X

    ################
    ### AJUSTAR 2024
    X = (proy_en_2_a??os + 1) * fnl.loc[fnl['year'] == a??o+1, 'full'].sum() / fnl.loc[fnl['year'] == a??o+2, 'full'].sum()
    fnl.loc[fnl['year'] == a??o+2, 'full'] = fnl.loc[fnl['year'] == a??o+2, 'full'] * X

    ######################
    ### AJUSTAR 2025-2032

    yoy= fnl.groupby('year', as_index=True)[['full']].sum()
    yoy['pct_var'] = np.append([np.nan], [(yoy.iloc[i,0] /yoy.iloc[i-1,0] -1) * 100 for i in range(1, len(yoy))])

    X= (((proy_decada * 100) * 11) - (yoy.loc[(yoy.index <= a??o+2) & (yoy.index >= a??o) , 'pct_var'].sum()) - (yoy.loc[yoy.index > a??o+2, 'pct_var'].sum())) / 8
    yoy.loc[yoy.index > a??o+2, 'pct_var'] = yoy.loc[yoy.index > a??o+2, 'pct_var'] + X

    for year in range(a??o+3,2033):
        X = (yoy.loc[yoy.index== year, 'pct_var'] / 100 +1) * fnl.loc[fnl['year'] == year-1, 'full'].sum() / fnl.loc[fnl['year'] == year, 'full'].sum()
        X= X[X.index[0]]
        fnl.loc[fnl['year'] == year, 'full'] = fnl.loc[fnl['year'] == year, 'full'] * X

    print('04 - Forecasts complete!')

    return fnl[['year', 'quarter', 'full']]

########################################################################################################################################

def forecast_pct_pib_industrial():

    import pandas as pd
    import numpy as np
    import requests
    import json
    import time

    from prophet import Prophet
    from functions import forecast_pib
    from keys import token_inegi, token_banxico

    ####################################################################
    #### Query and forecast GDP total values using previously defined
    #### function. Values are afterwards going  to be used to forecast 
    #### industrial GDP.
    ####################################################################

    print('0.00 - Calculating GDP forecasts')
    print('-----')
    pib= forecast_pib()

    ####################################################################
    #### This file is used to check which is the last recorded quarter 
    #### from previous forecasts. If a new record is observed, then the
    #### file is updated and everything gets re-forecasted.
    ####################################################################

    print('-----')
    print('1.00 - Checking las INEGI records')

    #Import INEGI Last Values file
    lv= pd.read_csv('INEGI_LastValues.csv')

    #Last records
    last_year_record= int(lv[(lv['Var'] == 'PIB_Industrial') & (lv['timestamp'] == lv.loc[lv['Var'] == 'PIB_Industrial', 'timestamp'].max())]['Y'])
    last_quarter_record= int(lv[(lv['Var'] == 'PIB_Industrial') & (lv['timestamp'] == lv.loc[lv['Var'] == 'PIB_Industrial', 'timestamp'].max())]['Q'])

    ####################################################################
    #### Query INEGI's API to get latest macroeconomic data for GDP, 
    #### industrial GDP and Investment. If new data is identified,
    #### leave a new record in INEGI_LastValues.csv
    ####################################################################

    print('2.00 - Querying INEGI for new Industrial GDP data')

    #Query
    query= f'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/6207062155/es/0700/false/BISE/2.0/{token_inegi}?type=json'
    response= requests.get(query)

    #Transform JSON to dataframe
    dat= pd.json_normalize(response.json()['Series'][0]['OBSERVATIONS']).rename(columns= {'OBS_VALUE': 'pib_industrial_var_anual'})

    dat['y']= pd.to_numeric([dat['TIME_PERIOD'][i].split('/')[0] for i in range(0, len(dat))])
    dat['q']= pd.to_numeric([dat['TIME_PERIOD'][i].split('/')[1] for i in range(0, len(dat))])
    dat['pib_industrial_var_anual']= pd.to_numeric(dat['pib_industrial_var_anual'])
    dat['yq']= [str(dat['y'][i])+'Q'+str(dat['q'][i]) for i in range(0, len(dat))]

    ########################
    #Check that last observation is the one registered 
    #If new observations are published, then update INEGI_lastValues.py file

    y_check= dat.iloc[-1, -3] == last_year_record #Year check
    q_check= dat.iloc[-1, -2] == last_quarter_record #Quarter check

    #Validation and updates
    if (y_check == True) & (q_check == True):
        print('----')
        print('Last YQ has not changed. Forecasting with last registered values')
        
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]

    elif (y_check == True) & (q_check == False):
        print('----')
        print('Year is same, but new Q data published. Forecasting with new quarter data')
        print('Updating INEGI_LastValues.csv file')
        
        lv= lv.append({'Var': 'PIB', 
                    'Y': dat.iloc[-1, -3], 
                    'Q': dat.iloc[-1, -2], 
                    'timestamp': time.time()}, ignore_index= True)
        lv.to_csv('INEGI_LastValues.csv', index= False)
            
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]
            
    elif (y_check == False):
        print('----')
        print('New year, new data published. Forecasting with new quarter data')
        print('Updating INEGI_LastValues.csv file')
        
        lv= lv.append({'Var': 'PIB', 
                    'Y': dat.iloc[-1, -3], 
                    'Q': dat.iloc[-1, -2], 
                    'timestamp': time.time()}, ignore_index= True)
        lv.to_csv('INEGI_LastValues.csv', index= False)
            
        pib_lastValue = [dat.iloc[-1, -3], dat.iloc[-1, -2]]

    ########################
    ##Filter selected columns
    print('----')
    print('3.00 - Calculating dependent variable')

    dat= dat[['yq', 'pib_industrial_var_anual']]

    #Can only obtain % yoy variation values from INEGI's API, so here are base absolute values
    base_1993= [1689645.0, 1665390.7, 1653018.0, 1715589.1]

    #Calculate remaining absolute values
    dat['pib_industrial_abs'] = np.nan
    dat.iloc[0:4, -1] = base_1993
    for i in range(4, len(dat)):
        dat.iloc[i, -1] = dat.iloc[i-4, -1] * (1+ dat.iloc[i, -2] / 100)

    #Attaching PIB total absolute values (query from previous function)
    index= pib.loc[(pib['year'] == pib_lastValue[0]) & (pib['quarter'] == pib_lastValue[1]), ].index[0]
    dat['pib'] = pib.iloc[:index+1,-1]

    #calculate Dependent variable (pct share of industrial gdp)
    dat['pct_pib_secondary_industrial'] = dat['pib_industrial_abs'] / dat['pib']
    dat= dat[['yq', 'pct_pib_secondary_industrial']]

    ####################################################################
    #### Facebook Prophet model to forecast automatically GDP growth
    #### based on new historic recors , towards 2032
    ####################################################################

    #Prepare data
    ts= dat[['yq', 'pct_pib_secondary_industrial']][66:].rename(columns= {'yq': 'ds',  'pct_pib_secondary_industrial': 'y'})
    ts['ds'] = pd.PeriodIndex(ts['ds'], freq='Q').to_timestamp()
    ts['cap'] = 0.165
    ts['floor'] = 0.155

    #Fit Model
    m= Prophet(growth = 'logistic')
    m.fit(ts)

    #Forecast period
    #Remaining quarters of the year to be forecasted
    if pib_lastValue[1] == 1:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q2', 'Q3', 'Q4']]
    elif pib_lastValue[1] == 2:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q3', 'Q4']]
    elif pib_lastValue[1] == 3:
        remaining_qs = [str(pib_lastValue[0])+i for i in ['Q4']]
    else:
        remaining_qs = []

    #Next years to be forecasted
    future_ds= np.append(remaining_qs, 
                        [str(y)+q for y in range(pib_lastValue[0] + 1, 2033) for q in ['Q1', 'Q2', 'Q3', 'Q4']])

    #Forecast to 2023
    future= pd.DataFrame(columns=['ds'])
    future['ds']= pd.PeriodIndex(future_ds, freq='Q').to_timestamp()
    future['cap'] = 0.168
    future['floor'] = 0.155
    forecast = m.predict(future)

    #Integrate historic w/ forecast dataframes
    fnl= pd.DataFrame(columns= ['ds', 'y', 'yhat'])
    fnl['ds'] = pd.PeriodIndex(ts['ds'], freq='Q')
    fnl['y'] = ts['y'].reset_index(drop=True)

    forecast['y'] = np.nan
    forecast['ds'] = pd.PeriodIndex(forecast['ds'], freq='Q')
    f= forecast[['ds', 'y', 'yhat']]

    fnl= pd.concat([fnl, f]).reset_index(drop= True)
    fnl['year'] = pd.PeriodIndex(fnl['ds'], freq='Q').year
    fnl['quarter'] = pd.PeriodIndex(fnl['ds'], freq='Q').quarter

    index_cut= fnl[(fnl['year'] == pib_lastValue[0]) & (fnl['quarter'] == pib_lastValue[1])].index[0]
    fnl['full'] = np.append(fnl.iloc[:index_cut+1, 1], 
                            fnl.iloc[index_cut+1:, 2])

    ## Ajustes en predicciones de Q2s por debajo de la media historica de Q2
    q2 = fnl[(fnl['y'].notna()) & (fnl['year'] > 2009)].groupby('quarter')['y'].mean()[2]
    fnl.loc[(fnl['y'].isna()) & (fnl['quarter'] == 2) & (fnl['yhat'] < q2), 'yhat'] = q2

    ####################################################################

    return fnl[['year', 'quarter', 'full']]