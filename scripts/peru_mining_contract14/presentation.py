"""New-result rendering with the approved C11/12 architecture."""
from pathlib import Path
import time, json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from art import OriginalArt, flagship, SIZES, COLORS
from workflow import OUT, MONTHS, sha, write_json, ACTIVE_CLASSES
from partial_lisa import P_FLOOR


def display_fields(p,save=True):
    p=p.copy();interior=p.E.eq(0)&p.N.eq(0)
    p['descriptive_display_class']=p.descriptive_class
    for layer,sig in [('raw','raw_sig'),('bh','fdr_sig')]:
        p[layer+'_display_class']=np.where(interior&~p[sig].fillna(False),'NS',p.descriptive_class)
        np.testing.assert_array_equal(p.loc[~interior,layer+'_display_class'],p.loc[~interior,'descriptive_class'])
        counts=pd.crosstab(p.month,p[layer+'_display_class']).reindex(columns=list(COLORS),fill_value=0)
        if save:counts.to_csv(OUT/f'canonical_contract14_{layer}_display_counts.csv')
    if save:p[['month','ubigeo_1993','descriptive_display_class','raw_display_class','bh_display_class']].to_parquet(OUT/'data/contract14_display_classes.parquet',index=False)
    return p
def extra_figures(p):
    from temporal import save_lines
    diag=pd.read_csv(OUT/'canonical_contract14_monthly_diagnostics.csv',parse_dates=['month'])
    save_lines(p,diag)
    counts=pd.crosstab(p.month,p.descriptive_class)
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    for ax,labels,title in zip(axs,[ACTIVE_CLASSES[:4],ACTIVE_CLASSES[4:],['Empty']],['Descriptive active interior','Descriptive islands','Structural Empty']):
        for c in labels:ax.plot(counts.index,counts[c],label=c,color=COLORS[c])
        ax.set(title=title,ylabel='Districts');ax.legend(ncol=len(labels));ax.grid(alpha=.15)
    fig.savefig(OUT/'figures/descriptive_counts.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for layer,color in [('raw','#8B5E3C'),('bh','#236B8E')]:
        per=pd.read_parquet(OUT/f'canonical_contract14_{layer}_persistence.parquet');per=per.loc[per.class_scope.eq('ALL')]
        for ax,col in zip(axs,['total_significant_months','longest_consecutive_run']):
            ax.hist(per.loc[per.total_significant_months.gt(0),col],bins=25,alpha=.5,label=layer.upper(),color=color)
            ax.set(xlabel=col.replace('_',' '),ylabel='Districts');ax.legend()
    fig.suptitle('Significant persistence among districts with discoveries')
    fig.savefig(OUT/'figures/persistence.png',dpi=130);plt.close(fig)
    floor=pd.read_csv(OUT/'canonical_contract14_monthly_bh.csv',parse_dates=['month'])
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    axs[0].plot(floor.month,floor.first_BH_critical,label='First BH critical value');axs[0].axhline(P_FLOOR,color='#C51B2D',label='Permutation floor');axs[0].legend()
    axs[1].plot(floor.month,floor.p_floor_count,label='At floor');axs[1].plot(floor.month,floor.floor_hits_not_rejected,label='At floor, not rejected');axs[1].legend();axs[1].set_ylabel('Active observations')
    fig.suptitle('Permutation resolution: BH can reject through higher ranks')
    fig.savefig(OUT/'figures/permutation_floor.png',dpi=130);plt.close(fig)
    models=[pd.read_csv(OUT/f'canonical_contract14_monthly_{k}_models.csv',parse_dates=['month']) for k in ['naive','full','active']]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for k,m,style in zip(['naive','full','active'],models,['-','-','--']):ax.plot(m.month,m['beta_'+k],style,label=k)
    ax.legend();ax.set(title='Monthly spatial association',ylabel='Slope');ax.axhline(0,color='black',lw=.6)
    fig.savefig(OUT/'figures/monthly_models.png',dpi=130);plt.close(fig)
def render(p,geometry,indices=None,worker=False):
    start=time.perf_counter();p=display_fields(p,save=not worker);art=OriginalArt(p,geometry)
    keys=set(pd.read_csv(OUT/'canonical_contract14_key_months.csv',parse_dates=['month']).month)
    (OUT/'figures/selected').mkdir(parents=True,exist_ok=True)
    for layer in ['descriptive','raw','bh']:
        for kind in SIZES:(OUT/f'frames/{layer}_{kind}').mkdir(parents=True,exist_ok=True)
    records=[];positions=[]
    for i,(month,g) in enumerate(p.groupby('month',sort=True)):
        if indices is not None and i not in indices:continue
        maps=[];pos_by_layer=[]
        for layer in ['descriptive','raw','bh']:
            mp,mp_pos,_=art.map(g,layer,OUT/f'frames/{layer}_map/{i:04d}.png')
            sc,sc_pos,_=art.scatter(g,layer,OUT/f'frames/{layer}_scatter/{i:04d}.png')
            flagship(mp,sc,g,layer,OUT/f'frames/{layer}_flagship/{i:04d}.png');maps.append(mp)
            pos_by_layer.append((mp_pos,sc_pos))
            for kind in SIZES:
                path=OUT/f'frames/{layer}_{kind}/{i:04d}.png'
                with Image.open(path) as im:im.load();assert im.size==SIZES[kind]
                records.append({'stream':f'{layer}_{kind}','frame':i,'month':month.strftime('%Y-%m'),'path':str(path.relative_to(OUT)),'sha256':sha(path)})
                if month in keys or (indices is not None and not worker):shutil.copy2(path,OUT/f'figures/selected/{month:%Y-%m}_{layer}_{kind}.png')
        assert pos_by_layer[0]==pos_by_layer[1]==pos_by_layer[2]
        positions.append({'month':str(month),'identical_layer_axes':True})
        if month in keys or (indices is not None and not worker):
            im=Image.new('RGB',(3*SIZES['map'][0],SIZES['map'][1]),'white')
            for j,mp in enumerate(maps):im.paste(mp,(j*SIZES['map'][0],0))
            im.save(OUT/f'figures/selected/{month:%Y-%m}_comparison.png')
        if i%20==0:print(f'Rendered {i+1}/288 months',flush=True)
    if indices is None:
        assert len(records)==288*9
        pd.DataFrame(records).to_csv(OUT/'metadata/frame_manifest.csv',index=False)
        write_json(OUT/'render_validation.json',{'status':'PASS','months_rendered':288,'frames':len(records),'seconds':time.perf_counter()-start,'sizes':SIZES,'x_limits':art.env['X_LIMIT'],'y_limits':art.env['Y_LIMIT'],'palette':COLORS,'map_edgecolor':'none','map_linewidth':0,'geometry_sha256':art.geometry_hash,'axes':positions,'display_rule':'NS masks active interior only; island colors are contextual; canonical inference includes islands'})
    if not worker:extra_figures(p)
    return records,positions,art.env['X_LIMIT'],art.env['Y_LIMIT'],art.geometry_hash

def render_worker(args):
    geometry,indices=args
    p=pd.read_parquet(OUT/'canonical_contract14_local_partial_lisa_19999.parquet')
    return render(p,geometry,indices,worker=True)

def render_parallel(p,geometry,workers=4):
    from concurrent.futures import ProcessPoolExecutor
    started=time.perf_counter();display_fields(p)
    records=[];positions=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(render_worker,[(geometry,list(range(k,288,workers))) for k in range(workers)]))
    for rows,pos,xlim,ylim,gh in results:records.extend(rows);positions.extend(pos)
    assert len(records)==288*9
    pd.DataFrame(records).sort_values(['stream','frame']).to_csv(OUT/'metadata/frame_manifest.csv',index=False)
    write_json(OUT/'render_validation.json',{'status':'PASS','months_rendered':288,'frames':len(records),'seconds':time.perf_counter()-started,'sizes':SIZES,'x_limits':xlim,'y_limits':ylim,'palette':COLORS,'map_edgecolor':'none','map_linewidth':0,'geometry_sha256':gh,'axes':positions,'display_rule':'NS masks active interior only; island colors are contextual; canonical inference includes islands'})
    extra_figures(p)
if __name__=='__main__':
    import sys
    p=pd.read_parquet(OUT/'canonical_contract14_local_partial_lisa_19999.parquet')
    if '--sample' in sys.argv:render(p,OUT/'data/inputs/peru1793_districts_1993_redatam.parquet',[0,144,287])
    else:render_parallel(p,OUT/'data/inputs/peru1793_districts_1993_redatam.parquet')
