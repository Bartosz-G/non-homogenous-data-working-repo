import numpy as np
import pandas as pd
import ast
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
