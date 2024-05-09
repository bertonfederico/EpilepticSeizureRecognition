import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")




################################
############# Main #############
################################
def create_plots(input_data):

    data_exploratory = input_data.copy()
    data_exploratory['y'] = np.where(data_exploratory['y'] == 1, 'Epileptic', 'Non-epileptic')

    # heatmap
    create_heatmap(data_exploratory)

    # potential
    create_pontential_plot(data_exploratory)

    # frequence
    create_frequence_plot(data_exploratory)




#################################
### Exploratory Data Analysis ###
#################################

# Heatmap
def create_heatmap(dataframe):
    # removing y values
    heatmap_data = dataframe.iloc[:,0:178]

    # creating heatmap
    sns.heatmap(heatmap_data.corr())
    plt.title("Heatmap")
    print_save_plots("basic\\eeg_heatmap", False)


# Potential plots
def create_pontential_plot(input_data):
    potential_df = create_df_potential(input_data)
    create_min_max_pontential_plot(potential_df[['id', 'min', 'max', 'y']])
    create_std_pontential_plot(potential_df[['id', 'std', 'y']])


# Min/max potential plots
def create_min_max_pontential_plot(potential_df):
    df_potential_min_max = (potential_df.melt(id_vars=['id', 'y'], var_name='Value type', value_name='value', ignore_index=True))
    folder_name = 'pot_min_max'
    create_cat_plot(df_potential_min_max, 'y', 'value', 'Potential values (μV)', 'Value type', folder_name)
    create_kde(df_potential_min_max, 'value', 'y', 'Potential values (μV)', 'Value type', folder_name)
    create_rel_plot(df_potential_min_max, 'id', 'value', 'y', 'Potential values (μV)', 'Value type', 'Value type', 2, folder_name)


# Std potential plots
def create_std_pontential_plot(potential_df):
    folder_name = 'pot_std'
    create_cat_plot(potential_df, 'y', 'std', 'Potential values (μV)', None, folder_name)
    create_kde(potential_df, 'std', 'y', 'Potential values (μV)', None, folder_name)
    create_rel_plot(potential_df, 'id', 'std', 'y', 'Potential values (μV)', None, None, 0, folder_name)


# Frequence plots
def create_frequence_plot(input_data):
    frequence_df = create_df_frequence(input_data)
    folder_name = 'freq'
    create_cat_plot(frequence_df, 'y', 'freq', 'Frequence values (Hz)', None, folder_name)
    create_kde(frequence_df, 'freq', 'y', 'Frequence values (Hz)', None, folder_name)
    create_rel_plot(frequence_df, 'id', 'freq', 'y', 'Frequence values (Hz)', None, None, 1, folder_name)




################################
####### Prepare methods ########
################################
def create_df_potential(input_data):
    df_potential_prepare = input_data.iloc[:, 0:178]

    # std
    df_potential_prepare = df_potential_prepare.std(axis=1).reset_index()
    df_potential_prepare.rename(columns={0: "std"}, inplace = True)

    # extreme min/max
    df_potential_prepare['min'] = input_data.min(axis=1, numeric_only=True)
    df_potential_prepare['max'] = input_data.max(axis=1, numeric_only=True)

    # id & y
    df_potential_prepare['id'] = df_potential_prepare.index + 1
    df_potential_prepare['y'] = input_data['y']
    return df_potential_prepare


def create_df_frequence(input_data):
    df_frequence_prepare = input_data.iloc[:, 0:178]

    # max
    df_frequence_prepare['max_number'] = 0
    for count in range(2,178):
        df_frequence_prepare.loc[(input_data["X"+str(count-1)] < input_data["X"+str(count)]) &
                                (input_data["X"+str(count)] > input_data["X"+str(count+1)]),
                        "max_number"] = df_frequence_prepare['max_number'] + 1
    df_frequence_prepare['freq'] = df_frequence_prepare['max_number']/1.02

    # id & y
    df_frequence_prepare['id'] = df_frequence_prepare.index + 1
    df_frequence_prepare['y'] = input_data['y']
    df_frequence_prepare = df_frequence_prepare[['id', 'freq', 'y']]
    return df_frequence_prepare




################################
######### Plot methods #########
################################
def create_cat_plot(df, x, y, title, col, folder_name):
    cat = sns.catplot(
        data=df,
        kind='boxen',
        x=x,
        y=y,
        col=col
    )
    cat.set_ylabels(title, clear_inner=False)
    cat.fig.subplots_adjust(top=.9)
    cat.fig.suptitle("Categorical plot - " + title)
    print_save_plots(folder_name + "\\cat_plot", False)


def create_kde(df, x, hue, title, col, folder_name):
    dist = sns.displot(
        data=df,
        kind='kde',
        x=x,
        hue=hue,
        col=col
    )
    dist.fig.subplots_adjust(top=.9)
    dist.fig.suptitle("Kernel Density Estimate - " + title)
    dist.set_xlabels(title, clear_inner=False)
    print_save_plots(folder_name + "\\kde_plot", False)
    

def create_rel_plot(df, x, y, col, title, hue, style, axline_numb, folder_name):
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


def print_save_plots(name, remove_legend):
    if (remove_legend): plt.legend([], [], frameon=False)
    plt.savefig('..\\outputImg\\' + name + '.png')
    plt.show()