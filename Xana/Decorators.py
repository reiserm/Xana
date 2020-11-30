#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from functools import wraps


class Decorators:
    """
    Decorators class: Decorators are defined here.
    """

    @staticmethod
    def input2list(func):
        @wraps(func)
        def wrapper(a, x, *args, **kwargs):
            if np.issubdtype(type(x), np.integer):
                if x == -1:
                    x = a.meta.index.values
                else:
                    x = [
                        x,
                    ]
            return func(a, x, *args, **kwargs)

        return wrapper

    @staticmethod
    def init_figure(x=(1, 1), y=(6, 5)):
        def init_figure_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if "ax" in kwargs:
                    ax = kwargs["ax"]
                    kwargs_new = dict(kwargs)
                    del kwargs_new["ax"]
                else:
                    fig, ax = plt.subplots(*x, figsize=y)
                    if type(ax) == np.ndarray:
                        ax = ax.flatten()
                    kwargs_new = dict(kwargs)
                return func(*args, ax=ax, **kwargs_new)

            return wrapper

        return init_figure_decorator
