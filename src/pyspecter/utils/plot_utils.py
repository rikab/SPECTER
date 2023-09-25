from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl


# ######################
# ##### Color Maps #####
# ######################

def initialize():
    fig, axes = plt.subplots(figsize=(8,8))
    plt.close()



# Initialize a new plot
def newplot(width = 8, height = 8, fontsize = 20, style = 'sans-serif', fontset = "cm", auto_layout = True, stamp = True):


    fig, axes = plt.subplots(figsize=(width,height))

    plt.rcParams['figure.figsize'] = (width,height)
    plt.rcParams['font.family'] = style
    plt.rcParams['mathtext.fontset']= fontset
    plt.rcParams['figure.autolayout'] = auto_layout
    plt.rcParams['font.size'] = str(fontsize)

    # Default colors and plot style
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "b", "g", "y", "c", "m"])
    plt.rcParams['lines.linewidth'] = 3

    if stamp:
        # Text in the top right corner, right aligned:
        plt.text(1, 1, 'SPECTER', horizontalalignment='right', verticalalignment='bottom', transform=axes.transAxes, fontsize=fontsize-2, weight='bold')

    return fig, axes