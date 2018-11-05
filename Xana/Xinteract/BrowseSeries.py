import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
import time

class BrowseSeries:
    def __init__(self, first, last, ):
        
        play = widgets.Play(
        #     interval=10,
            value=0,
            min=first,
            max=last,
            step=1,
            description="Press play",
            disabled=False,
            continuous_updata=1
        )
        slider = widgets.IntSlider()
        widgets.jslink((play, 'value'), (slider, 'value'))
        hbox = widgets.HBox([play, slider])
        display(hbox)
        fig, ax = plt.subplots(1,1,figsize=(9,6))
        while True:
            print(play.value)
            time.sleep(2)
            ax.plot(play.value)
            fig.canvas.draw_idle()

