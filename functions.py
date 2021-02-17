def plot_hm(df, value, vmin, vmax, month):
    df = df[df['month'] == month]
    df = pd.pivot_table(df, values=value, index=['lat'], columns=['lon'])
    df = df[::-1]
    plt.figure(figsize=(21, 11), dpi=100, facecolor='w', edgecolor='k')
    chart = sns.heatmap(df, vmin=vmin, vmax=vmax)
    m_name = months[months['order'] == month]['month'].tolist()[0]
    plt.savefig('maps/'+ str(month) + "_" + m_name + "_" + value + '.png')
    print("save "+value+" chart for month - " + m_name)

def extract_d(f_input, f_output, param):
    if os.path.isfile(f_output):
        print("File " + f_output +  " exist")
    else:
        print("File not exist")
        with open(f_input) as fin, open(f_output, 'w') as fout:
            for line in fin:
                fout.write(";".join(line.split()) + "\n")
            print("File " + f_output +  " saved successfully!")

    df = pd.read_csv(f_output, delimiter=';')
    print('File ' + f_output +  ' read successfully!')
    cols = df.columns
    df.reset_index(inplace=True)

    df = pd.melt(df, id_vars=['level_0', 'level_1', 'level_2'], value_vars = cols, value_name = param, var_name='index')
    df = df.rename(columns={"level_0": "year", "level_1": "month", "level_2": "day"})
    df = df.astype( {'year': int, 'month': int, 'day': int})
    df = df[(df['year']>=start) & (df['year'] <= end)]
    print('Function run successfully!\n')
    print("****************************")
    return df