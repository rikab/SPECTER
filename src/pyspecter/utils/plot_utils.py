from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import numpy as np

# ######################
# ##### Color Maps #####
# ######################

def initialize():
    fig, axes = plt.subplots(figsize=(8,8))
    plt.close()



# Initialize a new plot
def newplot(width = 8, height = 8, fontsize = 20, style = 'serif', fontset = "cm", auto_layout = True, stamp = True):


    fig, axes = plt.subplots(figsize=(width,height))

    plt.rcParams['figure.figsize'] = (width,height)
    plt.rcParams['font.family'] = style
    plt.rcParams['mathtext.fontset']= fontset
    plt.rcParams['figure.autolayout'] = auto_layout
    plt.rcParams['font.size'] = str(fontsize)

    # Default colors and plot style
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "b", "g", "y", "c", "m"])


    if stamp:
        # Text in the top right corner, right aligned:
        plt.text(1, 1, 'SPECTER', horizontalalignment='right', verticalalignment='bottom', transform=axes.transAxes, fontsize=fontsize-2, weight='bold')

    return fig, axes



# ######################


def plot_event(ax, event, R, filename=None, color="red", title="", show=True, label = "Event"):

    # plot the two events
    plt.rcParams.update({'font.size': 18})
    if ax is None:
        newplot()

    pts, ys, phis =event[:,0], event[:, 1], event[:, 2]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=10, label=label)

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [150]

    # plot settings
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.title(title)
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    # ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.show()
        plt.close()

    


    # Function to take a list of points and create a histogram of points with sqrt(N) errors, normalized to unit area
def hist_with_errors(ax, points, bins, range, weights = None, **kwargs):

    if weights is None:
        weights = np.ones_like(points)

    hist, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errs2 = np.histogram(points, bins = bins, range = range, weights = weights**2)[0] + 1

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    hist_tot = np.sum(hist * bin_widths)
    hist = hist / hist_tot
    errs2 = errs2 / (hist_tot**2)

    ax.errorbar(bin_centers, hist, np.sqrt(errs2), xerr = bin_widths, fmt = "o", **kwargs)


def hist_with_outline(ax, points, bins, range, weights = None, color = "purple", **kwargs):
    
    if weights is None:
        weights = np.ones_like(points)

    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = 0.25, histtype='stepfilled', density = True, **kwargs)
    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = 0.75, histtype='step', density = True)



# function to add a stamp to figures
def stamp(left_x, top_y,
          ax=None,
          delta_y=0.05,
          textops_update=None,
          boldfirst = True,
          **kwargs):
    
     # handle defualt axis
    if ax is None:
        ax = plt.gca()
    
    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'fontsize': 18,
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)
    
    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))


        if t is not None:
            if boldfirst and i == 0:
                ax.text(left_x, y, t, weight='bold', **textops)
            else:
                ax.text(left_x, y, t, **textops)
