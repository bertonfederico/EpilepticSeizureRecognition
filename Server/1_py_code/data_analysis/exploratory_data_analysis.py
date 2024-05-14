import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


def create_plots(input_data: pd.DataFrame):
    """
    Creates all plots
    :param input_data: EEG dataset
    """

    data_exploratory = input_data.copy()
    data_exploratory['y'] = np.where(data_exploratory['y'] == 1, 'Epileptic', 'Non-epileptic')

    """ heatmap """
    create_heatmap(data_exploratory)

    """ potential """
    create_potential_plot(data_exploratory)

    """ frequency """
    create_frequency_plot(data_exploratory)




""""""""""""""""""""""""""""""""""""
"""   Exploratory Data Analysis  """
""""""""""""""""""""""""""""""""""""
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
    print_save_plots("basic\\eeg_heatmap", False)


def create_potential_plot(dataframe: pd.DataFrame):
    """
    Potential plots creation
    :param dataframe: EEG dataset
    """

    potential_df = create_df_potential(dataframe)
    create_min_max_potential_plot(potential_df[['id', 'min', 'max', 'y']])
    create_std_potential_plot(potential_df[['id', 'std', 'y']])


def create_min_max_potential_plot(potential_df: pd.DataFrame):
    """
    Min/max potential plots
    :param potential_df: input dataframe
    """

    df_potential_min_max = (
        potential_df.melt(id_vars=['id', 'y'], var_name='Value type', value_name='value', ignore_index=True))
    folder_name = 'pot_min_max'
    create_cat_plot(df_potential_min_max, 'y', 'value', 'Potential values (μV)', 'Value type', folder_name)
    create_kde(df_potential_min_max, 'value', 'y', 'Potential values (μV)', 'Value type', folder_name)
    create_rel_plot(df_potential_min_max, 'id', 'value', 'y', 'Potential values (μV)', 'Value type', 'Value type', 2,
                    folder_name)


def create_std_potential_plot(potential_df: pd.DataFrame):
    """
    Std potential plots
    :param potential_df: input dataframe
    """

    folder_name = 'pot_std'
    create_cat_plot(potential_df, 'y', 'std', 'Potential values (μV)', None, folder_name)
    create_kde(potential_df, 'std', 'y', 'Potential values (μV)', None, folder_name)
    create_rel_plot(potential_df, 'id', 'std', 'y', 'Potential values (μV)', None, None, 0, folder_name)


def create_frequency_plot(dataframe: pd.DataFrame):
    """
    Frequency plots
    :param dataframe: input dataframe
    """

    frequency_df = create_df_frequency(dataframe)
    folder_name = 'freq'
    create_cat_plot(frequency_df, 'y', 'freq', 'Frequency values (Hz)', None, folder_name)
    create_kde(frequency_df, 'freq', 'y', 'Frequency values (Hz)', None, folder_name)
    create_rel_plot(frequency_df, 'id', 'freq', 'y', 'Frequency values (Hz)', None, None, 1, folder_name)




""""""""""""""""""""""""""""""""""""
"""        Prepare methods       """
""""""""""""""""""""""""""""""""""""
def create_df_potential(input_data: pd.DataFrame):
    df_potential_prepare = input_data.iloc[:, 0:178]

    """ std """
    df_potential_prepare = df_potential_prepare.std(axis=1).reset_index()
    df_potential_prepare.rename(columns={0: "std"}, inplace=True)

    """ extreme min/max """
    df_potential_prepare['min'] = input_data.min(axis=1, numeric_only=True)
    df_potential_prepare['max'] = input_data.max(axis=1, numeric_only=True)

    """ id & y """
    df_potential_prepare['id'] = df_potential_prepare.index + 1
    df_potential_prepare['y'] = input_data['y']
    return df_potential_prepare


def create_df_frequency(input_data: pd.DataFrame):
    df_frequency_prepare = input_data.iloc[:, 0:178]

    """ max """
    df_frequency_prepare['max_number'] = 0
    for count in range(2, 178):
        df_frequency_prepare.loc[(input_data["X" + str(count - 1)] < input_data["X" + str(count)]) &
                                 (input_data["X" + str(count)] > input_data["X" + str(count + 1)]),
        "max_number"] = df_frequency_prepare['max_number'] + 1
    df_frequency_prepare['freq'] = df_frequency_prepare['max_number'] / 1.02

    """ id & y """
    df_frequency_prepare['id'] = df_frequency_prepare.index + 1
    df_frequency_prepare['y'] = input_data['y']
    df_frequency_prepare = df_frequency_prepare[['id', 'freq', 'y']]
    return df_frequency_prepare




""""""""""""""""""""""""""""""""""""
"""          Plot methods        """
""""""""""""""""""""""""""""""""""""
def create_cat_plot(df: pd.DataFrame, x: str, y: str, title: str, col: str, folder_name: str):
    cat = sns.catplot(
        data=df,
        kind='boxen',
        x=x,
        y=y,
        col=col
    )
    cat.set_ylabels(title, clear_inner=False)
    cat.set_xlabels("")
    cat.fig.subplots_adjust(top=.9)
    cat.fig.suptitle("Categorical box plot - " + title)
    cat.fig.set_facecolor('none')
    print_save_plots(folder_name + "\\cat_plot", False)


def create_kde(df: pd.DataFrame, x: str, hue: str, title: str, col: str, folder_name: str):
    dist = sns.displot(
        data=df,
        kind='kde',
        x=x,
        hue=hue,
        col=col,
        fill=True
    )
    for ax in dist.axes.flat:
        for collect in [0, 1]:
            kde_data = ax.collections[collect].get_paths()[0].vertices
            kde_x, kde_y = kde_data[:, 0], kde_data[:, 1]
            max_density_index = np.argmax(kde_y)
            max_density_x = kde_x[max_density_index]
            ax.axvline(x=max_density_x, color='orange' if collect == 0 else 'blue')

    dist.fig.subplots_adjust(top=.9)
    dist.set_ylabels("Probability density")
    dist.fig.suptitle("Kernel Density Estimate - " + title)
    dist.fig.set_facecolor('none')
    dist.set_xlabels(title, clear_inner=False)
    print_save_plots(folder_name + "\\kde_plot", False)


def create_rel_plot(df: pd.DataFrame, x: str, y: str, col: str, title: str, hue: str, style: str, axline_numb: int, folder_name: str):
    ret = sns.relplot(
        data=df,
        x=x,
        y=y,
        col=col,
        hue=hue,
        style=style,
        edgecolor='#CCFFFF'
    )
    ret.set_xlabels('Neurological beats recordings', clear_inner=False)
    ret.set_ylabels(title, clear_inner=False)
    ret.fig.subplots_adjust(top=.9)
    ret.fig.suptitle("Relational plot - " + title)
    ret.set(xticklabels=[])
    ret.fig.set_facecolor('none')
    if axline_numb == 2:
        axes = ret.axes.flat[0]
        axes.axhline(df[(df[col] == 'Non-epileptic') & (df[hue] == 'min')][y].mean(), ls='--', linewidth=2, color='red')
        axes.axhline(df[(df[col] == 'Non-epileptic') & (df[hue] == 'max')][y].mean(), ls='--', linewidth=2, color='red')
        axes = ret.axes.flat[1]
        axes.axhline(df[(df[col] == 'Epileptic') & (df[hue] == 'min')][y].mean(), ls='--', linewidth=2, color='red')
        axes.axhline(df[(df[col] == 'Epileptic') & (df[hue] == 'max')][y].mean(), ls='--', linewidth=2, color='red')
    if axline_numb == 1:
        axes = ret.axes.flat[0]
        axes.axhline(df[(df[col] == 'Non-epileptic')][y].mean(), ls='--', linewidth=2, color='red')
        axes = ret.axes.flat[1]
        axes.axhline(df[(df[col] == 'Epileptic')][y].mean(), ls='--', linewidth=2, color='red')
    print_save_plots(folder_name + "\\rel_plot", False)


def print_save_plots(name: str, remove_legend: bool):
    if (remove_legend): plt.legend([], [], frameon=False)
    plt.savefig('..\\outputImg\\' + name + '.png')
    plt.show()
