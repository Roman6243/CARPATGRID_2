import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp
from functions import *
import csv

exec(open("functions.py").read())

# turn off the ineractive mode for pop up plot windows
plt.ioff()
plt.isinteractive()

os.chdir('C:\\Users\\mukho\\Downloads\\CARPATGRID')
print(" ".join(['Current path is', os.getcwd()]))
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 1000)

MONTHS = [['January', 1], ['February', 2], ['March', 3], ['April', 4], ['May', 5], ['June', 6],
          ['July', 7], ['August', 8], ['September', 9], ['October', 10], ['November', 11], ['December', 12]]
MONTHS = pd.DataFrame(MONTHS, columns=['month', 'order'])

YEARS = [[1961, 1970], [1971, 1980], [1981, 1990], [1991, 2000], [2001, 2010]]
Y_FRAMES = pd.DataFrame(YEARS, columns=['start', 'end'])

# extract altitude
f_in = 'INPUT\\PredtandfilaGrid.dat'
f_out = 'INPUT\\altitude_grid.csv'
if os.path.isfile(f_out):
    print('File ' + f_out + ' exist')
else:
    with open(f_in) as fin, open(f_out, 'w') as fout:
        for line in fin:
            fout.write(";".join(line.split()) + "\n")
alt = pd.read_csv(f_out, delimiter=';')
alt["index"] = alt["index"].astype(str)

print("\nWould you like to recalculate parameters: 1)RG; 2)MRT; 3)WS10m; 4)RH; 5)TMax ?\n")
print("y/n ?")
answer = input()

if answer == 'y':
    # to write data in different year batch
    for i in range(5):
        start = Y_FRAMES['start'][i]
        end = Y_FRAMES['end'][i]

        rg = extract_d(f_input = "CARPATGRID_RG\\CARPATGRID_RG_D.ser", f_output = "CARPATGRID_RG\\CARPATGRID_RG_D.csv", param = 'rg12')
        rg = rg.merge(alt, how='left', on=['index'])
        rg.loc[rg['altitude'] >= 1200, 'rg12'] = -0.1945*(rg['rg12']**2) + 35.054*rg['rg12'] + 42.521
        rg.loc[rg['altitude'] < 1200, 'rg12'] = -0.3408*(rg['rg12']**2) + 39.654*rg['rg12'] + 23.76

        mrt = rg.copy()
        mrt.loc[mrt['altitude'] >= 1200, 'mrt'] = -0.00002*(mrt['rg12']**2) + 0.0699*mrt['rg12'] - 13.387
        mrt.loc[mrt['altitude'] < 1200, 'mrt'] = -0.00003* (mrt['rg12']**2) + 0.0806*mrt['rg12'] - 7.1488

        ws10m = extract_d(f_input = "CARPATGRID_WS10\\CARPATGRID_WS10_D.ser", f_output = "CARPATGRID_WS10\\CARPATGRID_WS10_D.csv", param = 'ws12')
        ws10m = ws10m.merge(alt, how='left', on=['index'])
        ws10m.loc[ws10m['altitude'] >= 1200, 'ws12'] = ws10m['ws12']*1.7
        ws10m.loc[ws10m['altitude'] < 1200, 'ws12'] = ws10m['ws12']*1.4
        ws10m.loc[ws10m['ws12'] < 0.5, 'ws12'] = 0.5
        ws10m.loc[ws10m['ws12'] > 27, 'ws12'] = 27

        rh = extract_d('CARPATGRID_RH\\CARPATGRID_RH_D.ser', "CARPATGRID_RH\\CARPATGRID_RH_D.csv", param = 'rh12')
        rh['rh12'] = rh['rh12'] * 0.86

        tmax = extract_d(f_input = "CARPATGRID_TMAX\\CARPATGRID_TMAX_D.ser", f_output = "CARPATGRID_TMAX\\CARPATGRID_TMAX_D.csv", param = 't12')

        print("Merge all parameters:")
        print("Merge rh and ws10m")
        df = rh.merge(ws10m, how='left', on=['index', 'day', 'month', 'year'])
        print("Merge df and rg")
        df = df.merge(rg[['year', 'month', 'day', 'index', 'rg12']], how='left', on=['index', 'day', 'month', 'year'])
        print("Merge df and mrt")
        df = df.merge(mrt[['year', 'month', 'day', 'index', 'mrt']], how='left', on=['index', 'day', 'month', 'year'])
        print("Merge df and tmax")
        df = df.merge(tmax[['year', 'month', 'day', 'index', 't12']], how='left', on=['index', 'day', 'month', 'year'])

        df['utci'] = 3.21 + 0.872 * df['t12'] + 0.2459 * df['mrt'] - 2.5078 * df['ws12'] - 0.0176 * df['rh12']
        df = df.round({'utci': 4, 't12': 4, 'mrt': 4, 'ws12': 4, 'rh12': 4, 'rg12': 4, 'lon': 1, 'lat': 1})
        # save calculated df on hdd
        df.to_pickle('results/df_' + str(start) + '_' + str(end) +'.pkl')
        print("\n***********************************************************************************************************************************************\n")
else:
    print("Skip this step of recalculation 1)RG; 2)MRT; 3)WS10m; 4)RH; 5)TMax")

# utci_1961_1970 = pd.read_pickle('results/df_1961_1970.pkl')
# utci_1971_1980 = pd.read_pickle('results/df_1971_1980.pkl')
# utci_1981_1990 = pd.read_pickle('results/df_1981_1990.pkl')
# utci_1991_2000 = pd.read_pickle('results/df_1991_2000.pkl')
# utci_2001_2010 = pd.read_pickle('results/df_2001_2010.pkl')
# 
# utci_list = [utci_1961_1970, utci_1971_1980, utci_1981_1990, utci_1991_2000, utci_2001_2010]
# utci = pd.concat(utci_list)
# utci.to_pickle('results/df_' + str(1961) + '_' + str(2010) +'.pkl')

utci = pd.read_pickle('results/df_1961_2010.pkl')
print("UTCI data read from pickle files successfully!")


# to know the min and max UTCI across all MONTHS
data = pd.DataFrame()
for month in MONTHS['order'].unique():
    temp = utci[utci['month'] == month].groupby(['lon', 'lat'])['utci'].mean().reset_index()
    temp = temp.assign(month = month)
    data = pd.concat([data, temp])
    print("Concat for month - " + str(month))
min = data['utci'].min()
max = data['utci'].max()

# to save maps on hdd
for month in MONTHS['order'].unique():
    plot_hm_utci(df=data, value='utci', vmax=max, vmin=min, month=month)

# there is no difference between average monthly and daily except February -> different number of days
m_mean_daily = utci.groupby(['month'])['utci'].mean().reset_index()
m_mean_monthly = utci.groupby(['year', 'month'])['utci'].mean().groupby(['month']).mean().reset_index()
m_d_comp = m_mean_daily.merge(m_mean_monthly, on=['month'], suffixes = ['_daily_avg', '_monthly_avg'])
m_d_comp['diff'] = round(m_d_comp['utci_daily_avg'] - m_d_comp['utci_monthly_avg'], 5)
print(m_d_comp)

# average daily temperature per year and calculate the trend (slope) of utci during 1961 - 2010 per each index
utci_slope = utci[['year', 'month', 'index', 'utci']]

print('\nWould you like to calculate slope (trend) per each month during whole time frame 1991-2010?\n')
answer = input()
if answer == 'y':
    # for month_n in MONTHS['order'].unique():
    for month_n in [2,3,4]:
        slope_df = pd.DataFrame()
        for index in utci_slope['index'].unique():
            m_mean_year_monthly = utci_slope[(utci_slope['month'] == month_n) & (utci_slope['index'] == index)].groupby(['year'])['utci'].mean().reset_index()
            x = m_mean_year_monthly['year'].to_numpy().reshape((-1,1))
            y = m_mean_year_monthly['utci'].to_numpy()
            model = LinearRegression(fit_intercept=True)
            model.fit(x, y)
            slope = round(model.coef_[0], 5)
            print('slope:', slope)

            temp = [[index, month_n, slope]]
            temp = pd.DataFrame(temp, columns=['index', 'month', 'slope'])
            slope_df = pd.concat([slope_df, temp])
            m_name = MONTHS[MONTHS['order'] == month_n]['month'].tolist()[0]
            print("calculate data for index - " + str(index))
        slope_df.to_pickle('results/utci_slope_df_month_' + str(m_name) + '.pkl')
else:
    print("Skip step of recalculation slope of UTCI per time frame 1961-2010\n")

# read the slope data and calculate the p-value
for m_number in [1,2,3]:
    utci_slope = pd.read_pickle('results/utci_slope_df_month_'+MONTHS[MONTHS['order']==m_number]['month'].tolist()[0]+'.pkl')
    utci_slope.reset_index(drop=True, inplace=True)
    pvalue_df = calculate_pvalue(df=utci_slope, var=str(m_number))
    plot_hm_utci(df=pvalue_df, value='pvalue', vmax=None, vmin=None, variable=m_number)
    plot_hm_utci(df=pvalue_df, value='slope', vmax=None, vmin=None, variable=m_number)

# *********************************** CATEGORY CALCULATION ***********************************
# calculate the number of days per each utci category
utci_ranges_n = [min, -40, -27, -13, 0, 9, 26, 32, 38, 46, max]
utci_ranges_v = ['EC', 'VSC', 'SC', 'MC', 'SLC', 'NT', 'MH', 'SH', 'VSH', 'EH']
for i in range(0, len(utci_ranges_n)-1):
    print("Define the category "+utci_ranges_v[i] )
    utci.loc[(utci['utci'] > utci_ranges_n[i]) & (utci['utci'] <= utci_ranges_n[i+1]), 'category'] = utci_ranges_v[i]

print('\nWould you like to calculate slope (trend) per decade during 1991-2010?\n')
answer = input()
if answer == 'y':
    for i in range(1,5):
        df = utci[(utci['year']>=Y_FRAMES['start']) & (utci['year'] <= Y_FRAMES['end'])]
        slope_days = pd.DataFrame()
        for index in df['index'].unique():
            df_dec_index = df[df['index'] == index].groupby(['year', 'category'])['utci'].\
                count().reset_index().rename(columns={'utci':'n_days'})
            temp_inter = pd.DataFrame()

            for cat in df_dec_index['category'].unique():
                df_annual_slope_dec = df_dec_index[df_dec_index['category'] == cat]

                x = df_annual_slope_dec['year'].to_numpy().reshape((-1,1))
                y = df_annual_slope_dec['n_days'].to_numpy()
                model = LinearRegression(fit_intercept=True)
                model.fit(x, y)
                slope = round(model.coef_[0], 5)

                temp = [[index, str(start)+"::" + str(end), slope, cat]]
                temp = pd.DataFrame(temp, columns=['index', 'decade', 'slope', 'category'])
                temp_inter = pd.concat([temp_inter, temp])

            slope_days = pd.concat([slope_days, temp_inter])
            print("calculate data for index - " + str(index))
        slope_days.to_pickle("results/days_slope_df_"+str(Y_FRAMES['start'])+".pkl")


days_slope = pd.read_pickle("results/days_slope_df.pkl")
for cat in days_slope['category'].unique():
    df_cat = days_slope[days_slope['category'] == cat]
    df_cat = calculate_pvalue(df=df_cat, var=cat)
    plot_hm_days(df=df_cat, value='pvalue', vmin=None, vmax=None, var=cat)
    plot_hm_days(df=df_cat, value='slope', vmin=None, vmax=None, var=cat)

# ************************************************************************************************************************
# Density Plot and Histogram of all arrival delays
# sns.distplot(utci_slope['slope'], hist=True, kde=True,
#              bins=200, color = 'darkblue',
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 2})
# plt.show()
# plt.close()
#
# sns.boxplot(x=utci_slope['slope'])
# plt.show()
# plt.close()