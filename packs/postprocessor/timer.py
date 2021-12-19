import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
from matplotlib import ticker
from scipy import optimize as op
from scipy import stats as st

class Timer:
    def __init__(self, var):
        font = {'size'   :40}
        matplotlib.rc('font', **font)
        self.lw=3
        self.var=var
        self.get_cases()

    def get_cases(self):
        var=self.var
        cases=[]
        for key in var.keys():
            cases.append(key.split('_')[1])
        self.cases=np.unique(np.array(cases))

    def export_processor_cumulative(self):
        var=self.var
        all_ords=[]
        for case in self.cases:
            fs_times=var['fs_'+case]
            ms_prep=var['prep_'+case][:-1]
            ms_proc=var['proc_'+case][:-1]
            vpi_proc=np.linspace(0,0.5,len(ms_proc))*100
            cum_ms=ms_prep.sum()+np.cumsum(ms_proc.sum(axis=1))
            cum_fs=np.cumsum(fs_times)
            all_ords.append(cum_ms)
            all_ords.append(cum_fs)
            plt.plot(vpi_proc,cum_ms,label='NU-ADM '+case,lw=self.lw)
            try:
                plt.plot(vpi_proc,cum_fs,label='fs '+case, lw=self.lw)
            except:
                pass
            plt.yscale('log')

        self.format_plot(np.unique(np.concatenate(all_ords)))
        plt.legend()
        plt.xlabel('PVI [%]', fontsize=60)
        plt.ylabel('time [s]',fontsize=60)
        plt.gcf().set_size_inches(20,20)
        plt.savefig('results/cumulative.svg', bbox_inches='tight', transparent=True)

    def export_final_time(self):
        plt.close('all')
        var=self.var
        all_ords=[]
        times_ms=[]
        times_fs=[]
        for case in self.cases:
            fs_times=var['fs_'+case]
            ms_prep=var['prep_'+case][:-1]
            ms_proc=var['proc_'+case][:-1]
            times_ms.append(ms_prep.sum()+ms_proc.sum())
            times_fs.append(fs_times.sum())
        cases=self.cases.astype(int)
        times_ms, times_fs=np.array(times_ms), np.array(times_fs)
        inds=np.argsort(cases)
        cases=cases[inds]
        times_ms, times_fs=times_ms[inds], times_fs[inds]
        pos=times_fs>0

        plt.scatter(cases[pos],times_fs[pos],label='fs ', lw=3*self.lw)
        plt.scatter(cases,times_ms,label='NU-ADM ',lw=3*self.lw)


        slope, intercept, r, p, se = st.linregress(np.log(cases[pos]), np.log(times_fs[pos]))
        a=np.float(np.exp(intercept))
        b=slope
        plt.plot(cases[pos],a*cases[pos]**b,label='$t^{fs}={'+"%.7f" % a+'}n^{'+"%.2f" % b+'}$',lw=6)

        slope, intercept, r, p, se = st.linregress(np.log(cases[cases>10000]), np.log(times_ms[cases>10000]))
        a=np.float(np.exp(intercept))
        b=slope
        # import pdb; pdb.set_trace()
        plt.plot(cases,a*cases**b,label='$t^{NU-ADM}={'+"%.7f" % a+'}n^{'+"%.2f" % b+'}$',lw=6)

        self.format_plot(np.unique(np.concatenate([times_ms, times_fs])),'lin_lin')
        plt.legend()
        plt.xlabel('$n []$', fontsize=60)
        plt.ylabel('time [s]',fontsize=60)
        plt.gcf().set_size_inches(20,20)
        plt.savefig('results/final.svg', bbox_inches='tight', transparent=True)
        # import pdb; pdb.set_trace()
        # vals=np.array(self.prep_time)
        # lines=
        # import pdb; pdb.set_trace()
    def curve_fit(x,y):
        import pdb; pdb.set_trace()


    def format_plot(self, ordenadas, scales='lin_log'):
        x_scale, y_scale = scales.split('_')
        if x_scale=='lin':
            x_scale='linear'
        if y_scale=='lin':
            y_scale='linear'
        else:
            yscale='log'
        plt.xscale(x_scale)
        plt.yscale(y_scale)

        plt.grid(which='major', lw=2, color='black')
        plt.grid(which='minor', lw=1, color='gray')
        if y_scale=='logs':
            major_ticks=np.log10(ordenadas).astype('int')
            if major_ticks.min()!=major_ticks.max():
                major_ticks=10**np.arange(major_ticks.min(),major_ticks.max()*10).astype(float)
                plt.gca().yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
                plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in np.concatenate([major_ticks])])
        major_ticks=10**np.unique((np.log10(sorted(ordenadas))).astype(int)).astype(float)
        major_ticks=np.append(major_ticks,major_ticks.max()*10)
        mantissa=     np.array([2, 3, 4, 5, 6, 7, 8, 9])
        mantissa_plot=np.array([1, 0, 0, 1, 0, 0, 0, 0])
        if y_scale=='log' and major_ticks.min()!=major_ticks.max():
            plt.gca().yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
            plt.gca().set_yticklabels([self.form(x) for x in np.concatenate([major_ticks])])

            minor_ticks=np.unique(np.array(ordenadas))#.astype('int'))
            minor_ticks=np.concatenate([major_ticks[i]*mantissa for i in range(len(major_ticks)-1)])
            mantissa_flags=np.concatenate([mantissa_plot for i in range(len(major_ticks)-1)])

            fmt=np.array([self.form(x) for x in np.round(minor_ticks,5)])
            fmt[mantissa_flags==0]= ''
            # import pdb; pdb.set_trace()
            plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            plt.gca().yaxis.set_minor_formatter(ticker.FixedFormatter(fmt))

            plt.ylim(major_ticks.min(),major_ticks.max()*1.1)

        plt.gcf().set_size_inches(15,15)
        pos=['left', 'right', 'bottom', 'top']
        for p in pos:
            plt.gca().spines[p].set_color('black')
            plt.gca().spines[p].set_linewidth(3)

    def form(self,x):
        if x>=1:
            return str(int(x))
        else: return(str(x))
def organize_results():
    cases_ms=[]
    for root, dirs, files in os.walk('results/'):
        for dir in dirs:
            variables={}
            for r, ds, fs in os.walk(os.path.join(root,dir)):
                for file in fs:
                    var_name=os.path.splitext(file)[0]
                    var_extention=os.path.splitext(file)[1]
                    if var_extention=='.npy':
                        variables[var_name]=np.load(os.path.join(os.path.join(root,dir),file))
    return variables

def fit_func(x,a,b):
    return np.log(a) + b * x
