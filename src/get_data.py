import requests
from tqdm import tqdm
import os
import pandas as pd
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import sparse as sp

mpl.rcParams['font.family'] = 'Helvetica'



def make_sdg(df, path_to_scientometrics):
    sdg_holder = {}
    for sdg in df['sustainable_development_goals']:
        # Check if the cell_data is iterable (i.e., not NaN or float)
        if isinstance(sdg, list):
            for ind_sdg in sdg:
                display_name = ind_sdg['display_name']
                score = ind_sdg['score']
                if display_name in sdg_holder:
                    sdg_holder[display_name] += score
                else:
                     sdg_holder[display_name] = score
    df_sdg = pd.DataFrame(list(sdg_holder.items()),
                          columns=['display_name', 'total_score'])
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Good health and well-being',
                                                                'Good health\nand well-being')
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Sustainable cities and communities',
                                                                'Sustainable cities\nand communities')
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Partnerships for the goals',
                                                                'Partnerships for\nthe goals')
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Quality education',
                                                                'Quality\neducation')
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Peace, justice, and strong institutions',
                                                                'Peace, justice, and\nstrong institutions	')
    df_sdg['display_name'] = df_sdg['display_name'].str.replace('Decent work and economic growth',
                                                                'Decent work and\neconomic growth')
    df_sdg = df_sdg.set_index(df_sdg['display_name'])
    df_sdg = df_sdg.sort_values(ascending=True, by='total_score')
    df_sdg.to_csv(os.path.join(path_to_scientometrics, 'sdg_scores.csv'))
    return df_sdg


def make_network(auth_df, norm=None):
    temp_auth = auth_df.drop_duplicates(subset=['doi', 'authorid'])
    temp_auth = temp_auth[temp_auth['doi'].notnull()]
    author_papers = temp_auth[temp_auth['authorid'].notnull()]
    int_p_id = dict(enumerate(list(author_papers['doi'].unique())))
    int_a_id = dict(enumerate(list(author_papers['authorid'].unique())))
    a_int_id = {authorId: intVal for intVal, authorId in int_a_id.items()}
    p_int_id = {paperId: intVal for intVal, paperId in int_p_id.items()}  # Fixed variable name
    author_paper_tuples = list(zip(author_papers['authorid'], author_papers['doi']))
    author_paper_tuples = [(a_int_id[t[0]], p_int_id[t[1]]) for t in author_paper_tuples]

    n_rows = len(a_int_id)  # Number of unique authors
    n_cols = len(p_int_id)  # Number of unique papers
    rows, cols = zip(*author_paper_tuples)  # Unpack the row (author) and column (paper) indices
    AP = sp.csc_matrix((np.ones(len(author_paper_tuples)), (rows, cols)), shape=(n_rows, n_cols))

    AA = AP.dot(AP.T)
    AA = np.array(AA - np.diag(AA.diagonal()))
    G = nx.from_numpy_array(AA, parallel_edges=True)
    centrality = nx.degree_centrality(G)
    norm = plt.Normalize(vmin=0, vmax=0.35)
    cmap = mpl.colormaps.get_cmap('Spectral_r')
    node_color = [cmap(norm(centrality[n])) for n in G.nodes()]
    return G, node_color, norm



def make_auth_df(df, path_to_scientometrics):
    auth_df = pd.DataFrame(index=[], columns=['doi', 'authorname'])
    counter = 0
    row_counter = 0
    for authorship in df['authorships']:
        if (authorship is not np.nan):
            if (len (authorship) > 0):
                for author in authorship:
                    auth_df.at[counter, 'doi'] = df.at[row_counter, 'doi']
                    auth_df.at[counter, 'authorname'] = author['author']['display_name']
                    counter +=1
        row_counter +=1
    auth_df['authorname'].value_counts().sort_values(ascending=False).to_csv(os.path.join(path_to_scientometrics,
                                                                                          'authorship_counts.csv'))
    auth_df['authorid'] = auth_df['authorname'].astype('category').cat.codes
    auth_df['authorname'] = auth_df['authorname'].str.normalize('NFKD').str.encode('ascii',
                                                                                   errors='ignore').str.decode('utf-8')
    print(auth_df['authorname'].value_counts().sort_values(ascending=False)[0:10])
    return auth_df



def make_concepts(df, data_path):
    # Create dictionaries to store the cumulative scores for each level
    concept_scores_level_1 = {}
    concept_scores_level_2 = {}
    concept_scores_level_3 = {}

    # Process each "cell" of data
    for cell_data in df['concepts']:
        # Check if the cell_data is iterable (i.e., not NaN or float)
        if isinstance(cell_data, list):
            for concept in cell_data:
                level = concept.get('level')
                display_name = concept['display_name']
                score = concept['score']

                if level == 1:
                    # Add score to the existing display_name for level 1
                    if display_name in concept_scores_level_1:
                        concept_scores_level_1[display_name] += score
                    else:
                        concept_scores_level_1[display_name] = score
                elif level == 2:
                    # Add score to the existing display_name for level 2
                    if display_name in concept_scores_level_2:
                        concept_scores_level_2[display_name] += score
                    else:
                        concept_scores_level_2[display_name] = score
                elif level == 3:
                    # Add score to the existing display_name for level 3
                    if display_name in concept_scores_level_3:
                        concept_scores_level_3[display_name] += score
                    else:
                        concept_scores_level_3[display_name] = score

    # Convert each dictionary into a DataFrame
    df_level_1 = pd.DataFrame(list(concept_scores_level_1.items()),
                              columns=['display_name', 'total_score_level_1'])
    df_level_2 = pd.DataFrame(list(concept_scores_level_2.items()),
                              columns=['display_name', 'total_score_level_2'])
    df_level_3 = pd.DataFrame(list(concept_scores_level_3.items()),
                              columns=['display_name', 'total_score_level_3'])

    df_level_1.to_csv(os.path.join(data_path, 'level_one_concepts.csv'))
    df_level_2.to_csv(os.path.join(data_path, 'level_two_concepts.csv'))
    df_level_3.to_csv(os.path.join(data_path, 'level_three_concepts.csv'))
    return df_level_1, df_level_2, df_level_3


def load_data(filename):
    return pd.read_excel(filename)


def get_first_display_name(locations):
    if not isinstance(locations, list):
        return None  # If the value is not a list, return None
    for location in locations:
        if location.get('source') and location['source'].get('display_name'):
            return location['source']['display_name']
    return None  # Return None if no display_name is found



def sanitize_string(input_str):
    return input_str.encode('ascii', 'ignore').decode('ascii')  # Ignores problematic characters


def get_all_openalex_dois(dois_to_query, fpath):
    print(f'We have {len(dois_to_query)} DOIs to query from OpenAlex')
    base_url = 'https://api.openalex.org/works/'
    records = []

    for doi in tqdm(dois_to_query):
        url = base_url + doi + \
              '?mailto=charles dot rahal at demography dot ox dot ac dot uk'
        response = requests.get(url)
        if response.status_code == 200:
            json_response = response.json()
            json_response['doi'] = doi
            records.append(json_response)
        elif response.status_code == 429:
            print("Woah! you're out of API keys there!")
            break
        else:
            records.append({'doi': doi, 'OpenAlex_Response': np.nan})
    df_openalex = pd.json_normalize(records)  # Flatten the JSON structure into columns
    df_openalex.to_csv(fpath)
    return df_openalex


def get_openalex_data():
    print('Getting OpenAlex Data!')
    df = load_data('publications.xlsx')
    df['DOI'] = df['DOI'].str.lower().str.strip()
    df = df[df['DOI'].notnull()]
    df_openalex = get_all_openalex_dois(df['DOI'].to_list())
    return df_openalex
