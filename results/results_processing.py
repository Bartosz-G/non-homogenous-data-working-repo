import numpy as np
import pandas as pd
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


def flatten_results(df: pd.DataFrame, omit: Optional[List[str]] = None) -> pd.DataFrame:
    if omit is None:
        omit = []

    # Flatten the DataFrame
    flat_df = pd.json_normalize(df.to_dict(orient='records'))

    # Rename the columns to remove prefixes
    flat_df.columns = flat_df.columns.str.replace('hyperparameters\.', '', regex=False)
    flat_df.columns = flat_df.columns.str.replace('metrics\.', '', regex=False)

    # Keep the original columns that are in 'omit'
    for col in omit:
        flat_df[col] = df[col]

    return flat_df