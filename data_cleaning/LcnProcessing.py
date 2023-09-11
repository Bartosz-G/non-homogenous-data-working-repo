import numpy as np
import pandas as pd

class editArgs():
    def __init__(self):
        self.special = True
        self.parent = None

    def apply(self, **kwargs):
        args_obj = kwargs.get('args', None)
        if args_obj is not None:
            X, y, categorical_indicator, attribute_names = self.parent.train
            num_columns_X = X.shape[1]
            num_columns_y = y.shape[1] if isinstance(y, pd.DataFrame) else 1
            args_obj.set_input_dims(num_columns_X, num_columns_y)