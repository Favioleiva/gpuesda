import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from workflow import OUT, COLORS, ACTIVE_CLASSES

def save_lines(panel,diag):
    dates=pd.to_datetime(diag.month)
    for tag,field in [('raw','raw'),('bh','fdr')]:
        prefix='RAW' if tag=='raw' else 'BH'
        for kind,labels in [('active_clusters',ACTIVE_CLASSES[:4]),('islands',ACTIVE_CLASSES[4:])]:
            fig,axs=plt.subplots(len(labels),1,figsize=(11,2.1*len(labels)),sharex=True,layout='constrained')
            for ax,c in zip(np.atleast_1d(axs),labels):
                ax.plot(dates,diag[prefix+'_'+c.replace(' ','_')],color=COLORS[c],lw=1.6,label='Significant '+c)
                if kind=='islands':
                    total=panel.descriptive_class.eq(c).groupby(panel.month).sum()
                    ax.plot(total.index,total.to_numpy(),color=COLORS[c],alpha=.3,ls='--',label='Descriptive total')
                ax.set_ylabel('Districts');ax.set_ylim(bottom=0);ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.15)
            fig.suptitle(f'{prefix} significant '+('active-interior classes' if kind=='active_clusters' else 'islands'))
            fig.savefig(OUT/f'figures/{tag}_{kind}.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for tag,color in [('RAW','#8B5E3C'),('BH','#236B8E')]:
        ax.plot(dates,diag[tag+'_discoveries']/diag.N_active,label=tag,color=color)
    ax.set(ylabel='Significant / all active',title='Monthly significant shares of active support',ylim=(0,None));ax.legend()
    fig.savefig(OUT/'figures/significance_shares.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    a=panel.loc[panel.E.eq(0)]
    for ax,col,label in [(axs[0],'p_raw','RAW pseudo-p'),(axs[1],'q_bh','Monthly BH q')]:
        ax.hist(a[col],bins=40,color='#547C91');ax.axvline(.05,color='#C51B2D',ls='--');ax.set(title=label,xlabel=label,ylabel='Active district-months')
    fig.savefig(OUT/'figures/p_q_diagnostics.png',dpi=130);plt.close(fig)
