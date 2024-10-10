import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import graphviz_layout
from get_data import make_sdg

mpl.rcParams['font.family'] = 'Helvetica'


def make_network_figure(fig_path,
                        G, node_color, norm,
                        G_mills, node_color_mills,
                        G_dowd, node_color_dowd):
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((2, 10), (0, 0), rowspan=2, colspan=6)
    ax2 = plt.subplot2grid((2, 10), (0, 6), rowspan=1, colspan=4)
    ax3 = plt.subplot2grid((2, 10), (1, 6), rowspan=1, colspan=4)

    # Plot the networks
    pos = graphviz_layout(G, prog="neato")

    nx.draw(G, pos, node_size=50,
            node_color=node_color,
            with_labels=False, ax=ax1, alpha=1,
            edgecolors='k',  # Set node border color to black
            linewidths=1,  # Set node border (edge) width
            width=0.1)  # Set edge (lines between nodes) width

    nx.draw(G_mills, pos, node_size=50,
            node_color=node_color_mills,
            with_labels=False, ax=ax2, alpha=1,
            edgecolors='k',  # Set node border color to black
            linewidths=1,  # Set node border (edge) width
            width=0.01)  # Set edge (lines between nodes) width

    nx.draw(G_dowd, pos, node_size=50,
            node_color=node_color_dowd,
            with_labels=False, ax=ax3, alpha=1,
            edgecolors='k',  # Set node border color to black
            linewidths=1,  # Set node border (edge) width
            width=0.1)  # Set edge (lines between nodes) width

    # Add titles
    ax1.set_title('a.', loc='left', fontsize=19, y=0.975)
    ax2.set_title('b.', loc='left', fontsize=19, y=0.95)
    ax3.set_title('c.', loc='left', fontsize=19, y=0.95)

    # Adjust the layout to make space for the colorbar on the right
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # Add a colorbar to the right of the figures based on the shared normalization
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('Spectral_r'), norm=norm)
    sm.set_array([])  # Needed to set the array for the ScalarMappable

    # Create the colorbar on the right side of the figure
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Degree Centrality', rotation=90, fontsize=16)

    # Save the figure in different formats
    plt.savefig(os.path.join(fig_path, 'figure_3_LCDS_5yr.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'figure_3_LCDS_5yr.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(fig_path, 'figure_3_LCDS_5yr.png'), dpi=400, bbox_inches='tight')


def plot_concepts(df_level_1, df_level_2, df_level_3, fig_path):
    colors = ['#2b83ba', '#d7191c', '#abdda4', '#fdae61']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 6))

    df_level_1_sorted = df_level_1.sort_values(by='total_score_level_1',
                                               ascending=False)[0:25]
    df_level_1_sorted = df_level_1_sorted.sort_values(by='total_score_level_1',
                                                      ascending=True)
    ax1.hlines(df_level_1_sorted['display_name'],
               [0],
               df_level_1_sorted['total_score_level_1'],
               color='k',
               linestyles='dotted',
               lw=2)
    ax1.plot(df_level_1_sorted['total_score_level_1'],
             df_level_1_sorted['display_name'],
             markeredgecolor='k', markerfacecolor=colors[0], linewidth=0, markersize=8,
             marker='o')

    df_level_2_sorted = df_level_2.sort_values(by='total_score_level_2',
                                               ascending=False)[0:25]
    df_level_2_sorted = df_level_2_sorted.sort_values(by='total_score_level_2',
                                                      ascending=True)
    ax2.hlines(df_level_2_sorted['display_name'],
               [0],
               df_level_2_sorted['total_score_level_2'],
               color='k',
               linestyles='dotted',
               lw=2)
    ax2.plot(df_level_2_sorted['total_score_level_2'],
             df_level_2_sorted['display_name'],
             markeredgecolor='k', markerfacecolor=colors[1], linewidth=0, markersize=8,
             marker='o')

    df_level_3_sorted = df_level_3.sort_values(by='total_score_level_3',
                                               ascending=False)[0:25]
    df_level_3_sorted['display_name'] = df_level_3_sorted['display_name'].str.replace('2019-20 coronavirus outbreak',
                                                                                      'Covid-19')

    df_level_3_sorted['display_name'] = df_level_3_sorted['display_name'].str.replace(
        'Infectious disease (medical specialty)',
        'Infectious disease')

    df_level_3_sorted['display_name'] = df_level_3_sorted['display_name'].str.replace(
        'Social network (sociolinguistics)',
        'Social networks')
    df_level_3_sorted = df_level_3_sorted.sort_values(by='total_score_level_3',
                                                      ascending=True)
    ax3.hlines(df_level_3_sorted['display_name'],
               [0],
               df_level_3_sorted['total_score_level_3'],
               color='k',
               linestyles='dotted',
               lw=2)
    ax3.plot(df_level_3_sorted['total_score_level_3'],
             df_level_3_sorted['display_name'],
             markeredgecolor='k', markerfacecolor=colors[3], linewidth=0, markersize=8,
             marker='o')

    ax1.set_xlabel("Level 1 Concepts\n(Average Score)")
    ax1.set_ylabel("")
    ax2.set_xlabel("Level 2 Concepts\n(Average Score)")
    ax2.set_ylabel("")
    ax3.set_xlabel("Level 3 Concepts\n(Average Score)")
    ax3.set_ylabel("")

    ax1.set_title('a.', loc='left', fontsize=17)
    ax2.set_title('b.', loc='left', fontsize=17)
    ax3.set_title('c.', loc='left', fontsize=17)

    ax1.grid(linestyle='--', alpha=0.225)
    ax2.grid(linestyle='--', alpha=0.225)
    ax3.grid(linestyle='--', alpha=0.225)

    sns.despine()
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, 'figure_2_LCDS_5yr.pdf'))
    plt.savefig(os.path.join(fig_path, 'figure_2_LCDS_5yr.svg'))
    plt.savefig(os.path.join(fig_path, 'figure_2_LCDS_5yr.png'), dpi=500)


def plot_figure1(df, data_path, figure_path):
    colors = ['#2b83ba', '#d7191c', '#abdda4', '#fdae61']
    colors6 = ['#3288bd', '#99d594', '#fee08b', '#d53e4f', '#fc8d59', '#e6f598']

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%Y-%m-%d')
    for pub in df[df['publication_date'].notnull()].index:
        size = np.sqrt(df.at[pub, 'cited_by_count']) * 15  # Scale the size based on citation count
        if df.at[pub, 'primary_topic.domain.display_name'] == 'Social Sciences':
            ax.scatter(x=df.at[pub, 'publication_date'],
                       y=df.at[pub, 'cited_by_count'],
                       s=size,  # Set size here
                       color=colors[0], edgecolor='k')
        elif df.at[pub, 'primary_topic.domain.display_name'] == 'Health Sciences':
            ax.scatter(x=df.at[pub, 'publication_date'],
                       y=df.at[pub, 'cited_by_count'],
                       s=size,  # Set size here
                       color=colors[1], edgecolor='k')
        elif df.at[pub, 'primary_topic.domain.display_name'] == 'Physical Sciences':
            ax.scatter(x=df.at[pub, 'publication_date'],
                       y=df.at[pub, 'cited_by_count'],
                       s=size,  # Set size here
                       color=colors[2], edgecolor='k')
        elif df.at[pub, 'primary_topic.domain.display_name'] == 'Life Sciences':
            ax.scatter(x=df.at[pub, 'publication_date'],
                       y=df.at[pub, 'cited_by_count'],
                       s=size,  # Set size here
                       color=colors[3], edgecolor='k')
    df[['publication_date',
        'cited_by_count',
        'primary_topic.domain.display_name']].to_csv(os.path.join(data_path, 'scatter_data.csv'))
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_ylabel('Cited by Count (Log Scale)', fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(which="major", linestyle='--', alpha=0.225)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('a.', loc='left', fontsize=19)
    ax2.set_axisbelow(True)
    ax2.grid(linestyle='--', alpha=0.225)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_title('b.', loc='left', fontsize=19)

    df_sdg = make_sdg(df, data_path)
    df_sdg.plot(ax=ax2, kind='bar',
                edgecolor='k', legend=False,
                color='#fee08b')
    for container in ax2.containers:
        ax2.bar_label(container, label_type='edge', fontsize=8)
    ax2.tick_params(axis='x', labelsize=8, labelrotation=45)

    df['primary_topic.field.display_name'] = df['primary_topic.field.display_name'].replace(
        {'Economics, Econometrics and Finance': 'EEF'})
    df['primary_topic.field.display_name'] = df['primary_topic.field.display_name'].replace(
        {'Biochemistry, Genetics and Molecular Biology': 'Biochem, Genetics and Molecular Biology'})

    df1 = pd.DataFrame(df['primary_topic.field.display_name'].value_counts()[0:10].sort_values(ascending=True))
    df1.to_csv(os.path.join(data_path, 'pie_chart_data.csv'))

    labels = df1.index.tolist()
    sizes = df1['count'].tolist()
    explode = [0.1] * len(labels)
    ax4.pie(
        sizes,
        labels=labels,
        startangle=15,
        wedgeprops={'width': 0.4, 'edgecolor': 'k'},
        explode=explode,
        colors=['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd',
                '#5e4fa2'],
        textprops={'fontsize': 8},
        autopct='%1.1f%%',
        pctdistance=0.85
    )
    ax2.set_xlabel('')
    ax2.set_ylabel('Aggregate Scores', fontsize=13)
    ax3.set_title('c.', loc='left', fontsize=19)
    df['first_display_name'] = df['first_display_name'].str.replace('Proceedings of the National Academy of Sciences',
                                                                    'PNAS')
    df['first_display_name'] = df['first_display_name'].str.replace('medRxiv (Cold Spring Harbor Laboratory)',
                                                                    'medRxiv')
    df['first_display_name'] = df['first_display_name'].str.replace('International Journal of Epidemiology', 'IJE')
    df['first_display_name'] = df['first_display_name'].str.replace('bioRxiv (Cold Spring Harbor Laboratory)',
                                                                    'bioRxiv')
    df['first_display_name'] = df['first_display_name'].str.replace('UCL Discovery (University College London)',
                                                                    'UCL Discovery')
    df['first_display_name'] = df['first_display_name'].str.replace('Population and Development Review', 'PDR')
    df['first_display_name'] = df['first_display_name'].str.replace('The Journals of Gerontology Series B',
                                                                    'Gerontology B')
    df['first_display_name'] = df['first_display_name'].str.replace('European Sociological Review', 'ESR')
    df['first_display_name'] = df['first_display_name'].str.replace('Nature Human Behaviour', 'ESR')
    df['first_display_name'] = df['first_display_name'].str.replace('SSM - Population Health', 'SSM - Pop Health')
    df['first_display_name'] = df['first_display_name'].str.replace('Population Studies', 'Pop Studies')
    df['first_display_name'] = df['first_display_name'].str.replace('Demographic Research', 'Dem Research')
    ax4.set_title('d.', loc='left', fontsize=19, x=-.065)
    df['first_display_name'].value_counts()[0:15].sort_values(ascending=True).plot(ax=ax3, kind='barh', edgecolor='k',
                                                                                   color=colors[0])
    df['first_display_name'].value_counts()[0:15].sort_values(ascending=True).to_csv(
        os.path.join(data_path, 'display_name.csv'))
    ax3.set_ylabel('')
    legend_elements2 = [
        Line2D([0], [0], color='k',
               lw=0, marker='o',
               markerfacecolor=colors[0], markersize=9,
               label=r'Social Sciences', alpha=1),
        Line2D([0], [0], color='k', lw=0,
               marker='o',
               markerfacecolor=colors[1], markersize=9,
               label=r'Health Sciences', alpha=1),
        Line2D([0], [0], color='k', lw=0,
               marker='o',
               markerfacecolor=colors[2], markersize=9,
               label=r'Physical Sciences', alpha=1),
        Line2D([0], [0], color='k', lw=0,
               marker='o',
               markerfacecolor=colors[3], markersize=9,
               label=r'Life Sciences', alpha=1),
    ]
    ax.legend(handles=legend_elements2, loc='upper right',
              frameon=True,
              fontsize=8, framealpha=1, facecolor='w',
              edgecolor=(0, 0, 0, 1), ncols=1)
    ax3.set_xlabel('Count', fontsize=13)

    ax2.grid(linestyle='--', alpha=0.225)
    ax3.grid(linestyle='--', alpha=0.225)
    sns.despine()
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(os.path.join(figure_path, 'figure_1_LCDS_5yr.pdf'))
    plt.savefig(os.path.join(figure_path, 'figure_1_LCDS_5yr.svg'))
    plt.savefig(os.path.join(figure_path, 'figure_1_LCDS_5yr.png'), bbox_inches='tight', dpi=600)
