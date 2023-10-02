import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional

def get_dataset(df: pd.DataFrame, group) -> pd.DataFrame:
    filtered_df = df[df['dataset'].isin(group)]
    return filtered_df

def get_groups():
    return {'opml_reg_purnum_group': ['336-361072',
                                      '336-361073',
                                      '336-361074',
                                      '336-361076',
                                      '336-361077',
                                      '336-361078',
                                      '336-361079',
                                      '336-361080',
                                      '336-361081',
                                      '336-361082',
                                      '336-361083',
                                      '336-361084',
                                      '336-361085',
                                      '336-361086',
                                      '336-361087',
                                      '336-361088',
                                      '336-361279',
                                      '336-361280',
                                      '336-361281'],
            'opml_class_purnum_group': ['337-361055',
                                        '337-361060',
                                        '337-361061',
                                        '337-361062',
                                        '337-361063',
                                        '337-361065',
                                        '337-361066',
                                        '337-361068',
                                        '337-361069',
                                        '337-361070',
                                        '337-361273',
                                        '337-361274',
                                        '337-361275',
                                        '337-361276',
                                        '337-361277',
                                        '337-361278'],
            'opml_reg_numcat_group': ['335-361093',
                                      '335-361094',
                                      '335-361096',
                                      '335-361097',
                                      '335-361098',
                                      '335-361099',
                                      '335-361101',
                                      '335-361102',
                                      '335-361103',
                                      '335-361104',
                                      '335-361287',
                                      '335-361288',
                                      '335-361289',
                                      '335-361291',
                                      '335-361292',
                                      '335-361293',
                                      '335-361294'],
            'opml_class_numcat_group': ['334-361110',
                                        '334-361111',
                                        '334-361113',
                                        '334-361282',
                                        '334-361283',
                                        '334-361285',
                                        '334-361286']}


def flatten_results(df: pd.DataFrame, flatten: Optional[List[str]] = None) -> pd.DataFrame:
    if flatten is None:
        return df

    all_cols = df.columns.tolist()
    cols_to_include = list(set(all_cols) - set(flatten))

    data_return = df[cols_to_include].copy()

    for column in flatten:
        # Convert string representation of dictionary back to dictionary
        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Extract the list of dictionaries for the current column, filtering out non-dict types
        data_list = [x for x in df[column].tolist() if isinstance(x, dict)]

        # Flatten the list of dictionaries into a DataFrame
        if data_list:
            flattened_df = pd.json_normalize(data_list, max_level=1)

            # Ensure the index matches before concatenating
            flattened_df.index = data_return.index[:len(flattened_df)]

            # Concatenate the original DataFrame and the flattened DataFrame column-wise
            data_return = pd.concat([data_return, flattened_df], axis=1)

    return data_return



def get_top_models(df: pd.DataFrame, target_column: str, top: int = 1, highest: bool = True) -> pd.DataFrame:
    # Sort the DataFrame based on the target_column and the 'highest' parameter
    sort_order = target_column if highest else [target_column, 'dataset', 'model']
    df = df.sort_values(by=sort_order, ascending=not highest)

    # Group by 'dataset' and 'model', then select the top rows
    top_df = df.groupby(['dataset', 'model']).head(top)

    return top_df


def plot_per_model_metrics(df, metric, jitter=False):
    # Create the plot
    sns.stripplot(data=df, x='model', y=metric, hue='model', jitter=jitter, dodge=True)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.title(f'{metric} by Model')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])

    # Add a legend
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()




def clean_model_names(df: pd.DataFrame) -> pd.DataFrame:
    df['model'] = df['model'].str.replace(r'(Mlp_\w+)(_cat|_cont)$', r'\1', regex=True)
    df['model'] = df['model'].str.replace(r'(_cls|_reg|_class)$', '', regex=True)
    return df



def pivot_per_dataset(df: pd.DataFrame, results_column: str) -> pd.DataFrame:
    # Create a pivot table with lists
    pivot_df = pd.pivot_table(df,
                              values=results_column,
                              index='model',
                              columns='dataset',
                              aggfunc=list,
                              fill_value=None)

    # Flatten the columns and reset index
    pivot_df.columns = [col if not isinstance(col, tuple) else '_'.join(col) for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # Explode lists into separate rows
    for col in pivot_df.columns[1:]:
        pivot_df = pivot_df.explode(col)

    # Reset the index to keep things tidy
    pivot_df.reset_index(drop=True, inplace=True)

    return pivot_df



def plot_per_datasets(df: pd.DataFrame, datasets: List[str]) -> None:
    # Melt the DataFrame for seaborn
    melted_df = df.melt(id_vars='model', value_vars=datasets)

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.stripplot(x='model', y='value', hue='variable', data=melted_df, jitter=False, dodge=False)

    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.legend(title='Dataset')

    # Scale Y-axis
    plt.ylim(0, 1)

    # Rotate X-axis labels for better visibility
    plt.xticks(rotation=90)

    plt.show()