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
print(" ".join(['\nCurrent path is', os.getcwd()]))
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 1000)

MONTHS = [['January', 1], ['February', 2], ['March', 3], ['April', 4], ['May', 5], ['June', 6],
          ['July', 7], ['August', 8], ['September', 9], ['October', 10], ['November', 11], ['December', 12]]
MONTHS = pd.DataFrame(MONTHS, columns=['month', 'order'])

YEARS = [[1961, 1970], [1971, 1980], [1981, 1990], [1991, 2000], [2001, 2010]]
Y_FRAMES = pd.DataFrame(YEARS, columns=['start', 'end'])

UTCI_RANGES_N = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
UTCI_RANGES_V = ['EC', 'VSC', 'SC', 'MC', 'SLC', 'NT', 'MH', 'SH', 'VSH', 'EH']

alt = read_altitude()

def calc_utci(alt):
    """
    calculate parameters: 1)RG; 2)MRT; 3)WS10m; 4)RH; 5)TMax
    """

    # to write data in different year batch
    for i in range(5):
        start = Y_FRAMES['start'][i]
        end = Y_FRAMES['end'][i]

        rg = extract_data(f_input = "CARPATGRID_RG\\CARPATGRID_RG_D.ser", f_output = "CARPATGRID_RG\\CARPATGRID_RG_D.csv", param = 'rg12')
        rg = rg.merge(alt, how='left', on=['index'])
        rg.loc[rg['altitude'] >= 1200, 'rg12'] = -0.1945*(rg['rg12']**2) + 35.054*rg['rg12'] + 42.521
        rg.loc[rg['altitude'] < 1200, 'rg12'] = -0.3408*(rg['rg12']**2) + 39.654*rg['rg12'] + 23.76

        mrt = rg.copy()
        mrt.loc[mrt['altitude'] >= 1200, 'mrt'] = -0.00002*(mrt['rg12']**2) + 0.0699*mrt['rg12'] - 13.387
        mrt.loc[mrt['altitude'] < 1200, 'mrt'] = -0.00003* (mrt['rg12']**2) + 0.0806*mrt['rg12'] - 7.1488

        ws10m = extract_data(f_input = "CARPATGRID_WS10\\CARPATGRID_WS10_D.ser", f_output = "CARPATGRID_WS10\\CARPATGRID_WS10_D.csv", param = 'ws12')
        ws10m = ws10m.merge(alt, how='left', on=['index'])
        ws10m.loc[ws10m['altitude'] >= 1200, 'ws12'] = ws10m['ws12']*1.7
        ws10m.loc[ws10m['altitude'] < 1200, 'ws12'] = ws10m['ws12']*1.4
        ws10m.loc[ws10m['ws12'] < 0.5, 'ws12'] = 0.5
        ws10m.loc[ws10m['ws12'] > 27, 'ws12'] = 27

        rh = extract_data('CARPATGRID_RH\\CARPATGRID_RH_D.ser', "CARPATGRID_RH\\CARPATGRID_RH_D.csv", param = 'rh12')
        rh['rh12'] = rh['rh12'] * 0.86

        tmax = extract_data(f_input = "CARPATGRID_TMAX\\CARPATGRID_TMAX_D.ser", f_output = "CARPATGRID_TMAX\\CARPATGRID_TMAX_D.csv", param = 't12')

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

    utci_1961_1970 = pd.read_pickle('results/df_1961_1970.pkl')
    utci_1971_1980 = pd.read_pickle('results/df_1971_1980.pkl')
    utci_1981_1990 = pd.read_pickle('results/df_1981_1990.pkl')
    utci_1991_2000 = pd.read_pickle('results/df_1991_2000.pkl')
    utci_2001_2010 = pd.read_pickle('results/df_2001_2010.pkl')

    utci_list = [utci_1961_1970, utci_1971_1980, utci_1981_1990, utci_1991_2000, utci_2001_2010]
    utci = pd.concat(utci_list)
    utci.to_pickle('results/df_' + str(1961) + '_' + str(2010) +'.pkl')

def read_utci():
    """
    read and calculate the number of days per each utci category
    :return:
    """
    utci = pd.read_pickle('results/df_1961_2010.pkl')
    for i in range(0, len(UTCI_RANGES_N)-1):
        print("Define the category "+UTCI_RANGES_V[i] )
        utci.loc[(utci['utci'] > UTCI_RANGES_N[i]) & (utci['utci'] <= UTCI_RANGES_N[i+1]), 'category'] = UTCI_RANGES_V[i]
    print("UTCI data read from pickle files successfully!\n")
    return utci

def save_utci_map(utci):
    """
    calculate the min and max UTCI across all MONTHS in the aim to normalize the colors on heatmaps
    """
    data = pd.DataFrame()
    for month in MONTHS['order'].unique():
        temp = utci[utci['month'] == month].groupby(['lon', 'lat'])['utci'].mean().reset_index()
        temp = temp.assign(month = month)
        data = pd.concat([data, temp])
        print(' '.join(["Append avg monthly utci data for month - ",month]))
    min = data['utci'].min()
    max = data['utci'].max()
    for month in MONTHS['order'].unique():
        data_m = data[data['month'] == month]
        plot_hm_utci(df=data_m, value='utci', vmax=max, vmin=min, variable=month)

def calc_slope_utci(utci):
    """
    average daily temperature per year and calculate the trend (slope) of utci during 1961 - 2010 per each index
    """
    # for month_n in MONTHS['order'].unique():
    # for that moment calculate only for three months
    for month_n in [1,2,3]:
        slope_df = pd.DataFrame()
        df_month = utci[utci['month'] == month_n]
        for index in utci['index'].unique():
            m_mean_year_monthly = df_month[df_month['index'] == index].groupby(['year'])['utci'].mean().reset_index()
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

def read_slope_utci():
    """
    read the slope data and calculate the p-value
    """
    # for that moment calculate only for three months
    for m_number in [1,2,3]:
        utci_slope = pd.read_pickle('results/utci_slope_df_month_'+MONTHS[MONTHS['order']==m_number]['month'].tolist()[0]+'.pkl')
        utci_slope.reset_index(drop=True, inplace=True)
        pvalue_df = calculate_pvalue(df=utci_slope, var=str(m_number))
        plot_hm_utci(df=pvalue_df, value='pvalue', vmax=None, vmin=None, variable=m_number)
        plot_hm_utci(df=pvalue_df, value='slope', vmax=None, vmin=None, variable=m_number)

# *********************************** CATEGORY CALCULATION ***********************************
# ********************************************************************************************
# Linear trend for the annual number of days of thermal stress categories
# ********************************************************************************************

def calc_slope_days_cat(utci):
    """
    to calculate slope (trend) of days number category during 1961-2010
    """
    slope_days = pd.DataFrame()
    # for cat in utci['category'].unique():
    # for that moment calculate only for current category
    for cat in ['SC', 'VSH', 'VSC', 'EC', 'EH']:
        df_cat = utci[utci['category'] == cat]
        temp_cat = pd.DataFrame()
        for index in sorted(df_cat['index'].unique()):
            df_cat_index = df_cat[df_cat['index'] == index].groupby(['year', 'category'])['utci']. \
                count().reset_index().rename(columns={'utci': 'n_days'})

            x = df_cat_index['year'].to_numpy().reshape((-1,1))
            y = df_cat_index['n_days'].to_numpy()
            model = LinearRegression(fit_intercept=True)
            model.fit(x, y)
            slope = round(model.coef_[0], 5)

            temp = [[index, slope, cat]]
            temp = pd.DataFrame(temp, columns=['index', 'slope', 'category'])
            temp_cat = pd.concat([temp_cat, temp])
            print(" ".join(["calculate data for category -", cat, "and for index", index]))
        slope_days = pd.concat([slope_days, temp_cat])
        slope_days.to_pickle("results/days_slope_df_1961_2010_"+cat+".pkl")

def read_slope_days_cat(utci):
    days_slope = pd.DataFrame()
    for cat in utci['category'].unique():
        temp = pd.read_pickle("results/days_slope_df_1961_2010_"+cat+".pkl")
        days_slope = pd.concat([days_slope, temp])
    return days_slope

def save_slope_pvalue_map(days_slope):
    for cat in days_slope['category'].unique():
        df_cat = days_slope[days_slope['category'] == cat]
        df_cat = calculate_pvalue(df=df_cat, var=cat)
        df_cat = df_cat.merge(alt[['lon', 'lat']], on=['lon', 'lat'], how='right')
        df_cat.to_excel("maps/n_days_slope/" + cat + '_n_days_slope.xlsx')
        plot_hm_cat(df=df_cat, value='pvalue', vmin=None, vmax=None, variable=cat)
        plot_hm_cat(df=df_cat, value='slope', vmin=None, vmax=None, variable=cat)

# ********************************************************************************************
# Mean annual number of days of thermal stress categories
# ********************************************************************************************
def save_days_cat_map(utci):
    for cat in utci['category'].unique():
        df_cat = utci[utci['category']==cat]
        df_cat = df_cat.groupby(['year', 'lon', 'lat']).count().reset_index().\
            groupby(['lon', 'lat'])['category'].mean().reset_index().\
            rename(columns={'category':'n_days'})
        df_cat = df_cat.merge(alt[['lon', 'lat']], on=['lon', 'lat'], how='right')
        df_cat.to_excel("maps/n_days_per_cat/"+cat+'_n_dasys.xlsx')
        plot_hm_cat(df=df_cat, value='n_days', vmin=None, vmax=None, variable=cat)

def main():
    utci = read_utci()
    save_days_cat_map(utci)
    # days_slope = read_slope_days_cat(utci)
    # save_slope_pvalue_map(days_slope)

if __name__ == "__main__":
    main()



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
# def main():