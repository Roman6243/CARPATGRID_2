import pandas
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_1samp
from functions import plot_hm
import geopandas as gpd

# turn off the ineractive mode for pop up plot windows
plt.ioff()
plt.isinteractive()

os.chdir('C:\\Users\\mukho\\Downloads\\CARPATGRID')
print(os.getcwd())
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 1000)

# extract altitude
f_in = 'PredtandfilaGrid.dat'
f_out = 'altitude_grid.csv'
if os.path.isfile(f_out):
    print('File ' + f_out + ' exist')
else:
    with open(f_in) as fin, open(f_out, 'w') as fout:
        for line in fin:
            fout.write(";".join(line.split()) + "\n")
alt = pd.read_csv(f_out, delimiter=';')
alt["index"] = alt["index"].astype(str)



# to write data in different year batch
years = [[1961, 1970], [1971, 1980], [1981, 1990], [1991, 2000], [2001, 2010]]
y_frames = pd.DataFrame(years, columns=['start', 'end'])
for i in range(1,5):
    start = y_frames['start'][i]
    end = y_frames['end'][i]

    def extract_d(f_input, f_output, param):
        if os.path.isfile(f_output):
            print("File " + f_input.split('\\')[1] +  " exist")
        else:
            print("File not exist")
            with open(f_input) as fin, open(f_output, 'w') as fout:
                for line in fin:
                    fout.write(";".join(line.split()) + "\n")
                print("File " + f_input.split('\\')[1] +  " saved successfully!")

        df = pd.read_csv(f_output, delimiter=';')
        print('File ' + f_input.split('\\')[1] +  ' read successfully!')
        cols = df.columns
        df.reset_index(inplace=True)

        df = pd.melt(df, id_vars=['level_0', 'level_1', 'level_2'], value_vars = cols, value_name = param, var_name='index')
        df = df.rename(columns={"level_0": "year", "level_1": "month", "level_2": "day"})
        df = df.astype( {'year': int, 'month': int, 'day': int})
        print('Function run successfully!')
        return df

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

utci_1961_1970 = pd.read_pickle('results/df1961_1970.pkl')
utci_1971_1980 = pd.read_pickle('results/df1971_1980.pkl')
utci_1981_1990 = pd.read_pickle('results/df1981_1990.pkl')
utci_1991_2000 = pd.read_pickle('results/df1991_2000.pkl')
utci_2001_2010 = pd.read_pickle('results/df2001_2010.pkl')

utci_list = [utci_1961_1970, utci_1971_1980, utci_1981_1990, utci_1991_2000, utci_2001_2010]
utci = pd.concat(utci_list)

# count number of days with different utci
# utci.loc[(utci['utci'] < -40), 'tsc'] = 'EC'
# utci.loc[(utci['utci'] > -40) & (utci['utci'] <= -27.1), 'tsc'] = 'VSC'
# utci.groupby(['tsc', 'year']).count()


# to know the min and max across all months
data = pd.DataFrame()
for i in range(1, 13):
    temp = utci[utci['month'] == i].groupby(['lon', 'lat'])['utci'].mean().reset_index()
    temp = temp.assign(month = i)
    data = pd.concat([data, temp])
    print("Concat for month - " + str(i))
min = data['utci'].min()
max = data['utci'].max()

# to save maps on hdd
months = [['January', 1], ['February', 2], ['March', 3], ['April', 4], ['May', 5], ['June', 6],
          ['July', 7], ['August', 8], ['September', 9], ['October', 10], ['November', 11], ['December', 12]]
months = pd.DataFrame(months, columns=['month', 'order'])
for month in months['order'].unique():
    plot_hm(df=data, value='utci', vmax=max, vmin=min, month=month)

# there is no difference between average monthly and daily except February -> different number of days
m_mean_daily = utci.groupby(['month'])['utci'].mean().reset_index()
m_mean_monthly = utci.groupby(['year', 'month'])['utci'].mean().groupby(['month']).mean().reset_index()
m_d_comp = m_mean_daily.merge(m_mean_monthly, on=['month'], suffixes = ['_daily_avg', '_monthly_avg'])
m_d_comp['diff'] = round(m_d_comp['utci_daily_avg'] - m_d_comp['utci_monthly_avg'], 5)
print(m_d_comp)

# average daily temperature per year and calculate the trend (slope) of utci during 1961 - 2010 per each index
utci_slope = utci[['year', 'month', 'index', 'utci']]

for month_n in months['order'].unique():
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
        m_name = months[months['order'] == month_n]['month'].tolist()[0]
        print("calculate data for index - " + str(index))
    slope_df.to_pickle('results/slope_df_month_' + str(m_name) + '.pkl')


# read the slope data and calculate the p-value
utci_slope = pd.read_pickle('results/slope_df_month_January.pkl')
utci_slope['index'] = utci_slope['index'].astype(int)
utci_slope.reset_index(drop=True, inplace=True)

pvalue_df = pd.DataFrame()
for index in utci_slope["index"].unique():
    slope_per_month = utci_slope[utci_slope['index'] == index]['slope']
    tset, pval = ttest_1samp(utci_slope['slope'], slope_per_month.tolist()[0])
    if pval < 0.05:
       print(" we are rejecting null hypothesis")
    else:
      print("we are accepting null hypothesis")
    temp = [[index, pval, slope_per_month.tolist()[0]]]
    temp = pd.DataFrame(temp, columns=['index', 'pvalue', 'slope'])
    pvalue_df = pd.concat([pvalue_df, temp])
slope_df_mean = pvalue_df['slope'].mean()
pvalue_df = pvalue_df.assign(slope_avg = slope_df_mean)
pvalue_df.loc[pvalue_df['pvalue'] > 0.05, 'significante'] = "insignificante"
pvalue_df.loc[pvalue_df['pvalue'] <= 0.05, 'significante'] = "significante"

alt['index'] = alt['index'].astype(int)
pvalue_df = pvalue_df.merge(alt, on=['index']).assign(month = 1)
plot_hm(df=pvalue_df, value='pvalue', vmax=None, vmin=None, month=1)
plot_hm(df=pvalue_df, value='slope', vmax=None, vmin=None, month=1)

temp.to_excel("utci_jan.xlsx")

# fg, ax = plt.subplots(figsize = (21, 11), dpi = 100)
# ax.plot(m_mean_year_monthly['year'], m_mean_year_monthly['utci'], color='c', marker='o')
# ax.set_xlabel('Time [Year]')
# ax.set_ylabel('Mean UTCI')
# ax.set_facecolor('#cccccc')
# ax.grid(True, linestyle = ':', alpha=0.8)
# ax.set_xticks(np.arange(1960, 2014, 4))
# plt.show()


fg, ax = plt.subplots(4, 1, sharey = True, sharex = True, figsize=(21, 11), dpi=100)
season=0
for month in range(1,4):
    m_year = utci[utci['month'] == month].groupby(['year'])['utci'].mean().reset_index()
    ax[season].plot(m_year['year'], m_year['utci'], marker='o', label = str(month))
ax[season].legend(loc='center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=3)
season=1
for month in range(4,7):
    m_year = utci[utci['month'] == month].groupby(['year'])['utci'].mean().reset_index()
    ax[season].plot(m_year['year'], m_year['utci'], marker='o', label = str(month))
ax[season].legend(loc='center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=3)
season=2
for month in range(7,10):
    m_year = utci[utci['month'] == month].groupby(['year'])['utci'].mean().reset_index()
    ax[season].plot(m_year['year'], m_year['utci'], marker='o', label = str(month))
ax[season].legend(loc='center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=3)
season=3
for month in range(10,13):
    m_year = utci[utci['month'] == month].groupby(['year'])['utci'].mean().reset_index()
    ax[season].plot(m_year['year'], m_year['utci'], marker='o', label = str(month))
ax[season].legend(loc='center', bbox_to_anchor=(0.5, -0.25), shadow=False, ncol=3)

plt.show()