import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np


def show_pie_chart(df, title='', cut_off=1.5, column="JASC_cat", legend=False, save=None):
    sel_jasc_cat = {
        21: 'Air Conditioning',
        22: 'Auto Flight',
        23: 'Communicatinos',
        25: 'Equipment/Furnishings',
        26: 'Fire Protection',
        28: 'Fuel',
        29: 'Hydraulic Power',
        27: 'Flight Control',
        30: 'Ice and Rain Protection',
        32: 'Landing Gear',
        33: 'Lights',
        34: 'Navigation',
        52: 'Doors',
        53: 'Fuselage',
        54: 'Nacelles/Pylons',
        49: 'Airborne Auxiliary Power',
        55: 'Stabilizers',
        57: 'Wings',
        72: 'Turbine/Turboprop Engine',
        73: 'Engine Fuel and Control',

    }

    def assign_jasc_cat(jasc_int):
        return int(str(jasc_int)[:2])

    if column == "JASC_cat":
        df['JASC_cat'] = df['JASCCode'].apply(lambda x: assign_jasc_cat(x))


    cat_count = (df.groupby(column).size() / df.shape[0] * 100)
    cat_sel = cat_count[cat_count >= cut_off]
    cat_other = cat_count[cat_count < cut_off].sum()
    cat_pie = cat_sel.to_dict()
    if cat_other > 0:
        cat_pie['Other'] = cat_other
    cat_pie = dict(sorted(cat_pie.items(), key=lambda x: x[1], reverse=True))

    if column == "JASC_cat":
        cmap = plt.cm.get_cmap('tab20', len(sel_jasc_cat) + 1)
        colors = [cmap(len(sel_jasc_cat)) if not cat in sel_jasc_cat.keys() else cmap(list(sel_jasc_cat).index(cat)) for cat in cat_pie.keys()]

    else:
        vals = cat_pie.values()
        # cmap = plt.cm.get_cmap('tab20', len(vals))
        cmap = mpl.colormaps['Blues']
        colors = [cmap((1 - list(vals).index(v)/len(vals))) for v in vals]

    cat_pie = {sel_jasc_cat[cat] if cat in sel_jasc_cat.keys() else cat: val for cat, val in cat_pie.items()}
    # cat_pie = {'A' if cat in sel_jasc_cat.keys() else cat: val for cat, val in cat_pie.items()}


    if legend:
        fig, ax = plt.subplots(figsize=(8, 3))
        patches, texts = ax.pie(list(cat_pie.values()), colors=colors, counterclock=False, radius=1.6)
        labels = ['{0} - {1:1.2f}%'.format(i, j) for i, j in zip(list(cat_pie.keys()), list(cat_pie.values()))]
        patches, labels, dummy = zip(*sorted(zip(patches, labels, list(cat_pie.values())),
                                             key=lambda x: x[2],
                                             reverse=True))


        plt.legend(patches, labels, loc='right', bbox_to_anchor=(0, 0.55), fontsize=12, framealpha=0)
    else:
        fig, ax = plt.subplots(figsize=(5, 3))
        # ax.pie(list(cat_pie.values()), labels=cat_pie.keys(), autopct='%1.1f%%',
        #        pctdistance=0.85, labeldistance=1.02, counterclock=False, colors=colors, radius=1.2,
        #        wedgeprops={'edgecolor': 'grey', 'linewidth': 1, 'linestyle': 'solid', 'antialiased': True},
        #        )
        ax.pie(list(cat_pie.values()), labels=cat_pie.keys(),
               pctdistance=0.85, labeldistance=1.02, counterclock=False, colors=colors, radius=1.2,
               wedgeprops={'edgecolor': 'grey', 'linewidth': 1, 'linestyle': 'solid', 'antialiased': True},
               )
        ax.set(aspect=1)
        plt.subplots_adjust(top=0.95)
    plt.tight_layout()
    plt.title(title)
    if save:
        plt.savefig(save, transparent=True)
    plt.show()


def show_heatmap(df_h, time_factor ='AircraftTotalTime', time_step=750, title='General'):
    '''
    Visualise heatmap of occurence of reports per aircraft over time
    :param df_h: pandas Dataframe for heatmap
    :param time_factor: 'AircraftTotalTime' or 'AircraftTotalCycles' or 'AgeatDifficulty'
    :param time_step: number of units of time_factor to group
    :param title: title of figure
    :return:
    '''
    df_age = df_h[df_h['ManufactureYear'] >= 2010]
    df_age = df_age[df_age["DifficultyDate"].dt.year >= 2010]
    if time_factor == 'AgeatDifficulty':
        df_age['AgeatDifficulty'] = 12 * (df_age['DifficultyDate'].dt.year - df_age['ManufactureYear']) \
                                    + df_age['DifficultyDate'].dt.month
    pre_dropna = df_age.shape[0]
    df_age = df_age.dropna(subset=time_factor, axis=0)
    print(f"{pre_dropna - df_age.shape[0]}/{pre_dropna} entries had not entry for {time_factor}")
    if not df_age.empty:
        df_age[time_factor] = (df_age[time_factor] / time_step).astype(int) * time_step
        size_filtered_df = df_age.groupby('icao24').filter(lambda g: len(g) > 3)
        heatmap_df = size_filtered_df.groupby([time_factor, 'icao24']).size().unstack()
        # print(heatmap_df.head())
        heatmap_df.sort_values(by=time_factor, inplace=True)
        heatmap_df.dropna(axis=0, how='all', inplace=True)
        s = heatmap_df.sum()
        heatmap_df = heatmap_df[s.sort_values(ascending=True).index]
        heatmap_df.sort_index(ascending=True, inplace=True)
        sns.heatmap(heatmap_df.transpose(), cmap=sns.color_palette("rocket_r", as_cmap=True))
        plt.title(title)
        plt.tight_layout()
        plt.show()


def show_feature_density(df, column, save=None):
    bins = np.histogram(df[column], bins=20)[1]
    df[~df["Inspection"]][column].plot.hist(density=True, alpha=0.5, label="No Inspection", bins=bins)
    df[df["Inspection"]][column].plot.hist(density=True, alpha=0.5, label="Inspection", bins=bins)
    plt.legend()
    plt.title(column)
    if save:
        plt.savefig(save)
    plt.show()


def show_2d_hist(df, col1="LayoverLocalStartTime", col2="LayoverHours", bins=100, fontsize=20, save=None):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

    ax[0].hist2d(data=df[~df["Inspection"]], x=col1, y=col2, bins=bins, density=True,
                      norm='log', cmap=mpl.colormaps['Blues'])
    ax[1].hist2d(data=df[df["Inspection"]], x=col1, y=col2, bins=bins, density=True,
                 norm='log', cmap=mpl.colormaps['Reds'])

    ax[0].set_title("No Inspection", fontsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize)
    ax[0].tick_params(axis='y', labelsize=fontsize)

    ax[1].set_title("Inspection", fontsize=fontsize)
    ax[1].tick_params(axis='x', labelsize=fontsize)

    if col1 == "LayoverLocalStartTime" and col2 == "LayoverHours":
        ax[0].set_xticks(ticks=[0, 6, 12, 18], labels=['0:00', '6:00', '12:00', '18:00'])
        ax[0].set_xlabel("Local Start Time", fontsize=fontsize)
        ax[0].set_ylabel("Duration (hours)", fontsize=fontsize)
        ax[0].set_yticks(ticks=range(0, 175, 12))
        ax[1].set_xlabel("Local Start Time", fontsize=fontsize)
        ax[1].set_xticks(ticks=[0, 6, 12, 18], labels=['0:00', '6:00', '12:00', '18:00'])
    else:
        ax[0].set_xlabel(col1)
        ax[0].set_ylabel(col2)
        ax[1].set_ylabel(col2)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()


def show_svm_decision_function(clf, save=None):
    grid_axis = np.linspace(0, 1, 500)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    fontsize = 20
    y = []
    for x1 in grid_axis:
        y1 = []
        for x0 in grid_axis:
            y1.append(clf.predict([[x0, x1]])[0])
        y.append(y1)
    # %%
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdBu
    red_name = cmap(0.25)
    blue_name = cmap(0.75)
    cmap = mpl.colors.ListedColormap([blue_name, red_name])
    im = ax.pcolormesh(xx, yy, y, cmap=cmap)
    ax.set_xticks(ticks=list(np.array([0, 6, 12, 18]) / 24), labels=['0:00', '6:00', '12:00', '18:00'])
    ax.set_yticks(ticks=list(np.array(range(0, 175, 12)) / (7 * 24))[1:], labels=range(0, 175, 12)[1:])
    ax.set_xlabel("Local Start Time", fontsize=fontsize)
    ax.set_ylabel("Duration (hours)", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    legend_elements = [mpl.patches.Patch(edgecolor=blue_name, label='No Inspection',
                             facecolor=blue_name),
                       mpl.patches.Patch(edgecolor=red_name, label='Inspection',
                             facecolor=red_name)
                       ]
    plt.legend(handles=legend_elements, fontsize=fontsize)
    plt.tight_layout()
    plt.axis("tight")
    if save:
        plt.title(save, size="medium")
        plt.savefig(save)
    plt.show()