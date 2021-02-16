def plot_hm(df, value, vmin, vmax, month):
    df = df[df['month'] == month]
    df = pd.pivot_table(df, values=value, index=['lat'], columns=['lon'])
    df = df[::-1]
    plt.figure(figsize=(21, 11), dpi=100, facecolor='w', edgecolor='k')
    chart = sns.heatmap(df, vmin=vmin, vmax=vmax)
    m_name = months[months['order'] == month]['month'].tolist()[0]
    plt.savefig('maps/'+ str(month) + "_" + m_name + "_" + value + '.png')
    print("save chart for month - " + m_name)
