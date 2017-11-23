import numpy as np
import matplotlib

mpl = matplotlib
mpl.use('Agg') # Must be before importing matplotlib.pyplot
mpl.rcParams['font.size'] = 13.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'x-small'
mpl.rcParams['figure.dpi'] = 200.0
mpl.rcParams['lines.markersize'] = 3.0
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 5.0
mpl.rcParams['xtick.minor.size'] = 3.0
mpl.rcParams['ytick.major.size'] = 5.0
mpl.rcParams['ytick.minor.size'] = 3.0

import matplotlib.pyplot
plt = matplotlib.pyplot

__all__ = ['matplotlib', 'mpl', 'plt', 'SimpleComparisonPlot']

_colors = ('#009292', '#ff6db6', '#490092', '#6db6ff', '#924900', '#24ff24')
_linestyles = ('-', '--', '-.', ':')


class SimpleComparisonPlot(object):
    def __init__(self, savefig_path=None, save_pdf=False, logx=True, logy=True):
        self.savefig_path = savefig_path
        self.save_pdf = save_pdf
        self.logx = logx
        self.logy = logy
        self.fig = None
        self.ax = None
        self.ax_lower = None


    def __enter__(self):
        self.fig, (self.ax, self.ax_lower) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': (1, 0.3), 'hspace':0})
        self.ax.set_xscale('log' if self.logx else 'linear')
        self.ax_lower.set_xscale('log' if self.logx else 'linear')
        self.ax.set_yscale('log' if self.logy else 'linear')
        self.ax_lower.set_yscale('linear')
        return self


    def __exit__(self, *exc_args):
        self.ax_lower.axhline(0.0, c='k', lw=0.5)
        self.ax_lower.minorticks_on()
        for t in self.ax_lower.yaxis.get_major_ticks()[-1:]:
            t.label1.set_visible(False)
        self.fig.tight_layout()
        if self.savefig_path:
            self.fig.savefig(self.savefig_path)
            if self.save_pdf:
                self.fig.savefig(self.savefig_path+'.pdf')
        plt.close(self.fig)


    def plot_data(self, ref_data, ref_label, other_data, other_labels, ref_as_line=False, interp=False):
        if isinstance(other_labels, str):
            ref_color = 'C1'
            other_format = [('-', 'C0')]
            other_data = [other_data]
            other_labels = [other_labels]
        else:
            ref_color = 'k'
            other_format = itertools.cycle(itertools.product(_linestyles, _colors))
            #other_colors = mpl.cm.get_cmap('viridis')(np.linspace(0, 1, len(other_data)))
            #other_linestyles = ['--', '-']*((len(other_data)+1)//2)

        for data, label, (ls, color) in zip(other_data, other_labels, other_format):
            self.add_line(self.mask_data(data), label, color, ls)
            self.add_line(self.compare_data(ref_data, data, interp), label, color, ls, lower=True)

        add_ref = self.add_line if ref_as_line else self.add_points
        add_ref(self.mask_data(ref_data), ref_label, ref_color)
        add_ref(self.compare_data(ref_data, ref_data), ref_label, ref_color, lower=True)


    def compare_data(self, ref_data, this_data, interp=False):
        d = dict()
        d['x'] = this_data['x']
        ref_y = ref_data['y']
        if interp:
            s = ref_data['x'].argsort()
            ref_x = ref_data['x'][s]
            this_x = d['x']
            if self.logx:
                ref_x = np.log(ref_x)
                this_x = np.log(this_x)
            ref_y = ref_data['y'][s]
            if self.logy:
                ref_y = np.log(ref_y)
            ref_y = np.interp(this_x, ref_x, ref_y)
            if self.logy:
                ref_y = np.exp(ref_y)
        for k in ('y', 'y+', 'y-'):
            if k in this_data:
                d[k] = this_data[k]/ref_y if self.logy else (this_data[k]-ref_y)
        d = self.mask_data(d)
        if self.logy:
            for k in ('y', 'y+', 'y-'):
                if k in this_data:
                    d[k] = np.log(d[k])
        return d


    def mask_data(self, data):
        if self.logy:
            mask = np.isfinite(data['y']) & (data['y'] > 0)
            d = {k: v[mask] for k, v in data.items()}
            if 'y-' in d:
                d['y-'][d['y-'] <= 0] = 1.0e-100
            return d
        return data


    def add_line(self, data, label, color, linestyle='-', lower=False):
        ax_this = self.ax_lower if lower else self.ax
        ax_this.plot(data['x'], data['y'], label=label, color=color, ls=linestyle)
        if 'y-' in data and 'y+' in data:
            ax_this.fill_between(data['x'], data['y-'], data['y+'], alpha=0.15, color=color, lw=0)


    def add_points(self, data, label, color, lower=False):
        ax_this = self.ax_lower if lower else self.ax
        if 'y-' in data and 'y+' in data:
            ax_this.errorbar(data['x'], data['y'], [data['y']-data['y-'], data['y+']-data['y']], label=label, color=color, marker='s', ls='')
        else:
            ax_this.plot(data['x'], data['y'], label=label, color=color, marker='s', ls='')


    def add_vband(self, x0, x1):
        for ax_this in (self.ax, self.ax_lower):
            xlim_lo, xlim_hi = ax_this.get_xlim()
            if self.logx:
                xlim_lo /= 1000.0
                xlim_hi *= 1000.0
            else:
                xlim_lo -= 1000.0
                xlim_hi += 1000.0
            ax_this.axvspan(xlim_lo, x0, alpha=0.1, color='k', lw=0)
            ax_this.axvspan(x1, xlim_hi, alpha=0.1, color='k', lw=0)


    def add_legend(self, **kwargs):
        d = dict(ncol=2)
        d.update(kwargs)
        self.ax.legend(**d)


    def set_lims(self, xlim=None, ylim=None, ylim_lower=None):
        if xlim:
            self.ax.set_xlim(xlim)
            self.ax_lower.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        if ylim_lower is None:
            ylim_lower = (-0.7, 0.7)
        self.ax_lower.set_ylim(ylim_lower)


    def set_labels(self, xlabel=None, ylabel=None, ylabel_lower=None, title=None):
        if xlabel:
            self.ax_lower.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if ylabel_lower is None:
            ylabel_lower = 'ln(ratio)' if self.logy else 'diff.'
        self.ax_lower.set_ylabel(ylabel_lower)
        if title:
            self.ax.set_title(title)
