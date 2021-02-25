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

def calculate_pvalue(df, var):
    """
        calculate the p-value
        the null-hypothesis is: slope_per_month is the mean value in our dataset
        if p-value < 0.05 then we reject the null-hypothesis
        since this p-value is less than our significance level alpha = 0.05, we reject the null hypothesis,
        we have sufficient evidence to say that the mean value of slope is not equal to proposed one.
        the p-value of our test is greater than alpha = 0.05, we fail to reject the null hypothesis of the test,
        we do not have sufficient evidence to say that the mean value of slope is different from proposed one.
    """
    pvalue_df = pd.DataFrame()
    for index in df["index"].unique():
        slope_per_index = df[df['index'] == index]['slope'].tolist()[0]
        tset, pval = ttest_1samp(df['slope'], slope_per_index)
        temp = [[index, pval, slope_per_index]]
        temp = pd.DataFrame(temp, columns=['index', 'pvalue', 'slope'])
        pvalue_df = pd.concat([pvalue_df, temp])
    slope_df_mean = pvalue_df['slope'].mean()
    pvalue_df = pvalue_df.assign(slope_avg = slope_df_mean)
    pvalue_df.loc[pvalue_df['pvalue'] > 0.05, 'significance'] = "insignificance"
    pvalue_df.loc[pvalue_df['pvalue'] <= 0.05, 'significance'] = "significance"
    pvalue_df = pvalue_df.merge(alt, on=['index']).assign(variable = var)
    print("\nPvalue calculated successfully for category "+var+"\n")
    return pvalue_df

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