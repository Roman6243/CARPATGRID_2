def plot_hm_cat(df, value, vmin, vmax, variable):
    df = pd.pivot_table(df, values=value, index=['lat'], columns=['lon'], dropna=False)
    df = df[::-1]
    plt.figure(figsize=(21, 11), dpi=100, facecolor='w', edgecolor='k')
    chart = sns.heatmap(df, vmin=vmin, vmax=vmax)
    plt.savefig('maps/'+ variable + "_" + value + '.png')
    plt.close()
    print(" ".join(["Save", value, "chart for category -", variable]))

def plot_hm_utci(df, value, vmin, vmax, variable):
    df = pd.pivot_table(df, values=value, index=['lat'], columns=['lon'])
    df = df[::-1]
    plt.figure(figsize=(21, 11), dpi=100, facecolor='w', edgecolor='k')
    chart = sns.heatmap(df, vmin=vmin, vmax=vmax)
    m_name = MONTHS[MONTHS['order'] == variable]['month'].tolist()[0]
    plt.savefig('maps/'+ str(variable) + "_" + m_name + "_" + value + '.png')
    plt.close()
    print(" ".join(["Save", value, "chart for month -", m_name]))

def read_file(ser_read, csv_write):
    global line_count
    line_count = 0
    for row in ser_read:
        line_count += 1
        csv_write.write(";".join(row.split()) + "\n")
    print(f'Write {line_count} rows.')

def extract_data(f_input, f_output, param):
    if os.path.isfile(f_output):
        print("File " + f_output +  " exist")
    else:
        print("File not exist")
        with open(f_input, mode='r') as ser_read, open(f_output, mode='w') as csv_write:
            line_count = 0
            read_file(ser_read, csv_write)

    df = pd.read_csv(f_output, delimiter=';')
    print(" ".join(["File", f_output, "read successfully!"]))
    cols = df.columns
    df.reset_index(inplace=True)

    df = pd.melt(df, id_vars=['level_0', 'level_1', 'level_2'], value_vars = cols, value_name = param, var_name='index')
    df = df.rename(columns={"level_0": "year", "level_1": "month", "level_2": "day"})
    df = df.astype( {'year': int, 'month': int, 'day': int})
    df = df[(df['year'] >= start) & (df['year'] <= end)]
    print('Function run successfully!\n')
    print("****************************")
    return df

def read_altitude():
    """
    extract altitude
    """
    f_in = 'INPUT\\PredtandfilaGrid.dat'
    f_out = 'INPUT\\altitude_grid.csv'
    if os.path.isfile(f_out):
        print('File ' + f_out + ' exists')
    else:
        print("File not exist")
        with open(f_in, mode='r') as ser_read, open(f_out, mode='w') as csv_write:
            line_count = 0
            read_file(ser_read, csv_write)
    alt = pd.read_csv(f_out, delimiter=';')
    alt["index"] = alt["index"].astype(str)
    return alt

"""
Prob(F-Statistic): This tells the overall significance of the regression. This is to assess the significance level of 
all the variables together unlike the t-statistic that measures it for individual variables. 

The null hypothesis under this is “all the regression coefficients are equal to zero”. 
Prob(F-statistics) depicts the probability of null hypothesis being true. 

If your p-value is 0.1921 (greater than 0.05) means that there is no statistically significant evidence to reject the null hypothesis. 
Thus, there is no evidence of a relationship (of the kind posited in your model) between the set of explanatory variables and your response variable.

If your p-value is less than 0.05 we reject the null hypothesis, and can say that regression parameters are dependent.
"""