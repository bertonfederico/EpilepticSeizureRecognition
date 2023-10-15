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
    # creating splitted seizure
    dataframe = get_df_splitted_seizure(input_data)

    # heatmap
    create_heatmap(input_data)

    # box-plot
    create_box_plot(dataframe)

    # kernel density estimate
    create_kernel_density_estimate(dataframe)

    # altitude
    altitude_df = create_df_altitude(input_data)
    create_altitude_mean_median_plot(altitude_df[['id', 'median', 'y']], 'median', 'Median')
    create_altitude_mean_median_plot(altitude_df[['id', 'mean', 'y']], 'mean', 'Mean')
    create_altitude_min_max_plot(altitude_df[['id', 'min', 'max', 'y']], 'min_max', 'Min/max', '')
    create_altitude_min_max_plot(altitude_df[['id', 'extreme_min', 'extreme_max', 'y']], 'extreme_min_max', 'Extreme min/max', 'extreme_')

    # frequence
    frequence_df = create_df_frequence(input_data)
    create_frequence_mean_median_plot(frequence_df[['id', 'median', 'y']], 'median', 'Median')
    create_frequence_mean_median_plot(frequence_df[['id', 'mean', 'y']], 'mean', 'Mean')
    create_frequence_min_max_plot(frequence_df[['id', 'min', 'max', 'y']], 'min_max', 'Min/max')



#################################
### Exploratory Data Analysis ###
#################################
# Box plot
def create_box_plot(dataframe):
    sns.catplot(
        data=dataframe.groupby(["id", "y"]).std().reset_index(),
        kind='box',
        x='y',
        y='EEG',
    ).fig.suptitle("Standard deviation")
    print_save_plots("basic\\eeg_standard_deviation", False)



# Heatmap
def create_heatmap(dataframe):
    dataframe['y'] = np.where(dataframe['y'] == 'Epileptic', 1, 0)
    sns.heatmap(dataframe.corr())
    plt.title("Heatmap")
    print_save_plots("basic\\eeg_heatmap", False)



# Kernel Density Estimate
def create_kernel_density_estimate(dataframe):
    sns.displot(
        data=dataframe.groupby(["id", "y"]).std().reset_index(),
        kind='kde',
        x='EEG',
        hue='y'
    ).fig.suptitle("Kernel Density Estimate")
    print_save_plots("basic\\eeg_kde", False)



# Mean/median altitude value epileptic recordings
def create_altitude_mean_median_plot(dataframe, mean_median_name, value_type):
    g = create_relplot(dataframe, 'id', mean_median_name, 'y', None, value_type + ' altitude')
    add_plot_mean_line(g, 0, dataframe[dataframe['y'] == 'Non-epileptic'][mean_median_name].mean())
    add_plot_mean_line(g, 1, dataframe[dataframe['y'] == 'Epileptic'][mean_median_name].mean())
    print_save_plots("altitude\\" + mean_median_name + "_altitude", True)



# Min/max altitude value epileptic recordings
def create_altitude_min_max_plot(dataframe, min_max_name, value_type, to_concatenate):
    df_min_max = (dataframe.melt(id_vars=['id', 'y'], var_name='min_or_max', value_name='min_max_value', ignore_index=False))
    g = create_relplot(df_min_max, 'id', 'min_max_value', 'y', 'min_or_max', value_type + ' altitude')
    add_plot_mean_line(g, 0, df_min_max[(df_min_max['y'] == 'Non-epileptic') & (df_min_max['min_or_max'] == (to_concatenate + 'min'))]['min_max_value'].mean())
    add_plot_mean_line(g, 1, df_min_max[(df_min_max['y'] == 'Epileptic') & (df_min_max['min_or_max'] == (to_concatenate + 'min'))]['min_max_value'].mean())
    add_plot_mean_line(g, 0, df_min_max[(df_min_max['y'] == 'Non-epileptic') & (df_min_max['min_or_max'] == (to_concatenate + 'max'))]['min_max_value'].mean())
    add_plot_mean_line(g, 1, df_min_max[(df_min_max['y'] == 'Epileptic') & (df_min_max['min_or_max'] == (to_concatenate + 'max'))]['min_max_value'].mean())
    print_save_plots("altitude\\" + min_max_name + "_altitude", False)



# Mean/median frequence value epileptic recordings
def create_frequence_mean_median_plot(dataframe, mean_median_name, value_type):
    g = create_relplot(dataframe, 'id', mean_median_name, 'y', None, value_type + ' frequence')
    add_plot_mean_line(g, 0, dataframe[dataframe['y'] == 'Non-epileptic'][mean_median_name].mean())
    add_plot_mean_line(g, 1, dataframe[dataframe['y'] == 'Epileptic'][mean_median_name].mean())
    print_save_plots("frequence\\" + mean_median_name + "_frequence", True)



# Min/max frequence value epileptic recordings
def create_frequence_min_max_plot(dataframe, min_max_name, value_type):
    df_extreme_min_max = (dataframe.melt(id_vars=['id', 'y'], var_name='min_or_max', value_name='min_max_value', ignore_index=False))
    g = create_relplot(df_extreme_min_max, 'id', 'min_max_value', 'y', 'min_or_max', value_type + ' frequence')
    add_plot_mean_line(g, 0, df_extreme_min_max[(df_extreme_min_max['y'] == 'Non-epileptic') & (df_extreme_min_max['min_or_max'] == 'min')]['min_max_value'].mean())
    add_plot_mean_line(g, 1, df_extreme_min_max[(df_extreme_min_max['y'] == 'Epileptic') & (df_extreme_min_max['min_or_max'] == 'min')]['min_max_value'].mean())
    add_plot_mean_line(g, 0, df_extreme_min_max[(df_extreme_min_max['y'] == 'Non-epileptic') & (df_extreme_min_max['min_or_max'] == 'max')]['min_max_value'].mean())
    add_plot_mean_line(g, 1, df_extreme_min_max[(df_extreme_min_max['y'] == 'Epileptic') & (df_extreme_min_max['min_or_max'] == 'max')]['min_max_value'].mean())
    print_save_plots("frequence\\" + min_max_name + "_frequence", False)





################################
######### Help methods #########
################################
def get_df_splitted_seizure(input_data):
    input_data['y'] = np.where(input_data['y'] == 1, 'Epileptic', 'Non-epileptic')
    df_splitted_seizure = (input_data
                    .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                    .reset_index()
                    .rename(columns={'index': 'id'})
                )
    df_splitted_seizure['time_label'] = (df_splitted_seizure['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))
    return df_splitted_seizure



def create_df_altitude(input_data):
    temp = input_data.iloc[:, 0:177]
    # median
    temp['median'] = input_data.median(axis=1, numeric_only=True)
    # mean
    temp['mean'] = input_data.mean(axis=1, numeric_only=True)
    # extreme min/max
    temp['extreme_min'] = input_data.min(axis=1, numeric_only=True)
    temp['extreme_max'] = input_data.max(axis=1, numeric_only=True)
    # min/max
    dataframe_max = input_data.iloc[:, 0:178]
    dataframe_min = input_data.iloc[:, 0:178]
    for count in range(2,178):
        dataframe_max.loc[(input_data["X"+str(count-1)] <= input_data["X"+str(count)]) & (input_data["X"+str(count)] < input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
        dataframe_max.loc[(input_data["X"+str(count-1)] >= input_data["X"+str(count)]) & (input_data["X"+str(count)] >= input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
        dataframe_max.loc[(input_data["X"+str(count-1)] > input_data["X"+str(count)]) & (input_data["X"+str(count)] < input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
        dataframe_min.loc[(input_data["X"+str(count-1)] >= input_data["X"+str(count)]) & (input_data["X"+str(count)] > input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
        dataframe_min.loc[(input_data["X"+str(count-1)] <= input_data["X"+str(count)]) & (input_data["X"+str(count)] <= input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
        dataframe_min.loc[(input_data["X"+str(count-1)] < input_data["X"+str(count)]) & (input_data["X"+str(count)] > input_data["X"+str(count+1)]), "X"+str(count)] = np.NaN
    dataframe_max = dataframe_max.drop(columns=['X1'])
    dataframe_max = dataframe_max.drop(columns=['X178'])
    dataframe_min = dataframe_min.drop(columns=['X1'])
    dataframe_min = dataframe_min.drop(columns=['X178'])
    temp['min'] = dataframe_min.mean(axis=1, numeric_only=True)
    temp['max'] = dataframe_max.mean(axis=1, numeric_only=True)
    # id
    temp['id'] = temp.index + 1
    # y = 0 --> 'Non-epileptic', y = 1 --> 'Epileptic'
    temp['y'] = np.where(input_data['y'] == 1, 'Epileptic', 'Non-epileptic')
    return temp



def create_df_frequence(input_data):
    temp_1 = input_data.iloc[:, 0:178]
    temp_2 = input_data.copy()
    temp_1['actual_freq'] = 1
    for count in range(2,178):
        temp_1.loc[((input_data["X"+str(count-1)] <= input_data["X"+str(count)]) & (input_data["X"+str(count)] < input_data["X"+str(count+1)])) |
                        ((input_data["X"+str(count-1)] >= input_data["X"+str(count)]) & (input_data["X"+str(count)] > input_data["X"+str(count+1)])),
                        'actual_freq'] = temp_1['actual_freq'] + 1
        temp_1.loc[((input_data["X"+str(count-1)] <= input_data["X"+str(count)]) & (input_data["X"+str(count)] <= input_data["X"+str(count+1)])) |
                        ((input_data["X"+str(count-1)] >= input_data["X"+str(count)]) & (input_data["X"+str(count)] > input_data["X"+str(count+1)])),
                        "X"+str(count)] = np.NaN
        temp_1.loc[((input_data["X"+str(count-1)] < input_data["X"+str(count)]) & (input_data["X"+str(count)] >= input_data["X"+str(count+1)])) |
                        ((input_data["X"+str(count-1)] > input_data["X"+str(count)]) & (input_data["X"+str(count)] <= input_data["X"+str(count+1)])),
                        "X"+str(count)] = temp_1['actual_freq']
        temp_1.loc[((input_data["X"+str(count-1)] < input_data["X"+str(count)]) & (input_data["X"+str(count)] >= input_data["X"+str(count+1)])) |
                        ((input_data["X"+str(count-1)] > input_data["X"+str(count)]) & (input_data["X"+str(count)] <= input_data["X"+str(count+1)])),
                        'actual_freq'] = 1
    temp_1 = temp_1.drop(columns=['actual_freq'])
    temp_1 = temp_1.drop(columns=['X1'])
    temp_1 = temp_1.drop(columns=['X178'])
    temp_2['min'] = temp_1.min(axis=1, numeric_only=True)
    temp_2['max'] = temp_1.max(axis=1, numeric_only=True)
    temp_2['mean'] = temp_1.mean(axis=1, numeric_only=True)
    temp_2['median'] = temp_1.median(axis=1, numeric_only=True)
    temp_2['id'] = temp_2.index + 1
    temp_2['y'] = np.where(input_data['y'] == 1, 'Epileptic', 'Non-epileptic')
    return temp_2



def add_plot_mean_line(plot, count, median_value):
    axes = plot.axes.flat[count]
    axes.axhline(median_value, ls='--', linewidth=2, color='red')



def create_relplot(data, x, y, col, hue, value_type):
    ret = sns.relplot(
        data=data,
        x=x,
        y=y,
        col=col,
        hue=hue,
        style=hue,
        edgecolor='#CCFFFF'
    )
    ret.set_xlabels("Heartbeat recording", clear_inner=False)
    ret.set_ylabels(value_type + " value", clear_inner=False)
    ret.fig.subplots_adjust(top=.9)
    ret.fig.suptitle(value_type + " value for each heartbeat recording")
    ret.set(xticklabels=[])
    return ret



def print_save_plots(name, remove_legend):
    if (remove_legend): plt.legend([], [], frameon=False)
    plt.savefig('Server\\outputImg\\' + name + '.png')
    plt.show()