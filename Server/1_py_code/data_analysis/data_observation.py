import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def observation(data):
    sns.set_style("whitegrid")
    
    # Selectiong n rows for each value of y
    n = 3
    data_y_1 = data[data['y'] == 1][:n]
    data_y_2 = data[data['y'] == 2][:n]
    data_y_3 = data[data['y'] == 3][:n]
    data_y_4 = data[data['y'] == 4][:n]
    data_y_5 = data[data['y'] == 5][:n]
    samples_to_show = pd.concat([data_y_1, data_y_2, data_y_3, data_y_4, data_y_5], axis=0, ignore_index=True)

    # Creating a dataframe with one row for each value of X
    df_splitted_seizure_short = (samples_to_show
                    .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                    .reset_index()
                    .rename(columns={'index': 'id'})
                )

    # Getting time_index column from time_label
    df_splitted_seizure_short['time_label'] = (df_splitted_seizure_short['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))

    # Creating and showing the graph
    g = sns.relplot(
        palette=sns.color_palette(n_colors=15),
        data=df_splitted_seizure_short,
        kind='line',
        x='time_label',
        y='EEG',
        col='y'
    )
    g.fig.subplots_adjust(top=.9, left=.07)
    g.fig.suptitle("y differences")
    g.fig.set_size_inches(13, 5)
    plt.legend([], [], frameon=False)
    plt.savefig('Server\\outputImg\\y_inspection\\y_differences.png')
    plt.show()

    # Checking the number of rows for each value of y
    data_y_1 = data[data['y'] == 1]
    data_y_2 = data[data['y'] == 2]
    data_y_3 = data[data['y'] == 3]
    data_y_4 = data[data['y'] == 4]
    data_y_5 = data[data['y'] == 5]

    
    labels = 'y = 1', 'y = 2', 'y = 3', 'y = 4', 'y = 5'
    sizes = [len(data_y_1.index), len(data_y_2.index), len(data_y_3.index), len(data_y_4.index), len(data_y_5.index)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    fig.suptitle("Number of rows for each value of y")
    plt.savefig('Server\\outputImg\\y_inspection\\row_number.png')
    plt.show()