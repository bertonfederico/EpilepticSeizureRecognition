import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def observation(data: pd.DataFrame):
    """
    Observing data before ml
    :param data: input dataset
    """

    y_percentage(data)
    y_differences(data)
    create_heatmap(data)


def y_percentage(data: pd.DataFrame):
    """
    Calculating and showing y values percentage
    :param data: input dataset
    """

    """ Checking the number of rows for each value of y """
    data_y_1 = data[data['y'] == 1]
    data_y_2 = data[data['y'] == 2]
    data_y_3 = data[data['y'] == 3]
    data_y_4 = data[data['y'] == 4]
    data_y_5 = data[data['y'] == 5]
    labels = ('Epileptic area\nin seizure activity', 'Tumor area', 'Healthy area\nin tumor brain',
              'Healthy brain\n- eyes closed', 'Healthy brain\n- eyes open')
    colors = plt.cm.Blues(np.linspace(0.2, 0.7, len(labels)))
    sizes = [len(data_y_1.index), len(data_y_2.index), len(data_y_3.index), len(data_y_4.index), len(data_y_5.index)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    fig.set_facecolor('none')
    plt.savefig('..\\outputImg\\y_inspection\\row_number.png')
    plt.show()


def y_differences(data: pd.DataFrame):
    """
    Showing x differences for every y value
    :param data: input dataset
    """

    sns.set_style("whitegrid")

    """ Selecting n rows for each value of y """
    n = 10
    data_y_1 = data[data['y'] == 1][:n]
    data_y_2 = data[data['y'] == 2][:n]
    data_y_3 = data[data['y'] == 3][:n]
    data_y_4 = data[data['y'] == 4][:n]
    data_y_5 = data[data['y'] == 5][:n]
    samples_to_show = pd.concat([data_y_1, data_y_2, data_y_3, data_y_4, data_y_5], axis=0, ignore_index=True)

    """ Creating a dataframe with one row for each value of X """
    df_divided_seizure_short = (samples_to_show
                                .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                                .reset_index()
                                .rename(columns={'index': 'id'})
                                )

    """ Getting time_index column from time_label """
    df_divided_seizure_short['time_label'] = (
        df_divided_seizure_short['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))

    """ Creating and showing the graph """
    subplot_names_mapping = {
        1: 'Epileptic area in seizure activity',
        2: 'Tumor area',
        3: 'Healthy area in tumor brain',
        4: 'Healthy brain - eyes closed',
        5: 'Healthy brain - eyes open'
    }
    df_divided_seizure_short['subplot_name'] = df_divided_seizure_short['y'].map(subplot_names_mapping)
    g = sns.relplot(
        palette=sns.color_palette(n_colors=15),
        data=df_divided_seizure_short,
        kind='line',
        x='time_label',
        y='EEG',
        col='subplot_name'
    )
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=.9, left=.07)
    g.fig.set_size_inches(13, 5)
    g.fig.set_facecolor('none')
    plt.legend([], [], frameon=False)
    plt.savefig('..\\outputImg\\y_inspection\\y_differences.png')
    plt.show()


def create_heatmap(dataframe: pd.DataFrame):
    """
    Heatmap creation
    :param dataframe: EEG dataset
    """

    """ removing y values """
    heatmap_data = dataframe.iloc[:, 0:178]

    """ creating heatmap """
    sns.heatmap(heatmap_data.corr(), cmap=sns.color_palette("Blues_d", as_cmap=True))
    plt.gcf().set_facecolor('none')
    plt.savefig('..\\outputImg\\y_inspection\\eeg_heatmap.png')
    plt.show()
