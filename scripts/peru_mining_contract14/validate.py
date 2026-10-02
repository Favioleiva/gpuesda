"""Independent saved-result validation and overlap diagnostics; never re-estimates production inference."""
import json, shutil, ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, linalg
from statsmodels.stats.multitest import multipletests
from workflow import OUT, MONTHS, ACTIVE_CLASSES, COLORS, sha, write_json
from partial_lisa import R_PERM, P_FLOOR

def comparison(p):
    old=pd.read_parquet(OUT/'data/inputs/canonical_contract12_local_partial_lisa_9999.parquet')
    old=old.loc[old.month.isin(MONTHS)]
    keys=['month','ubigeo_1993'];cols=['beta_active','E','descriptive_class','raw_class','fdr_class']
    both=p[keys+cols].merge(old[keys+cols],on=keys,how='outer',validate='one_to_one',suffixes=('_new','_old'),indicator=True)
    assert len(both)==516384 and both._merge.eq('both').all()
    out=OUT/'comparison_old_vs_new';out.mkdir(exist_ok=True)
    monthly=[]
    for month,g in both.groupby('month',sort=True):
        r={'month':month,'beta_old':g.beta_active_old.iloc[0],'beta_new':g.beta_active_new.iloc[0],
           'Empty_old':int(g.E_old.sum()),'Empty_new':int(g.E_new.sum()),
           'active_old':int(g.E_old.eq(0).sum()),'active_new':int(g.E_new.eq(0).sum())}
        for col in ['descriptive_class','raw_class','fdr_class']:
            equal=g[col+'_new'].eq(g[col+'_old']);r[col+'_agreement']=float(equal.mean());r[col+'_changed']=int((~equal).sum())
        monthly.append(r)
    m=pd.DataFrame(monthly);m['absolute_beta_change']=(m.beta_new-m.beta_old).abs()
    m.to_csv(out/'monthly_comparison.csv',index=False,float_format='%.17g')
    districts=pd.DataFrame({'ubigeo_1993':sorted(p.ubigeo_1993.unique())}).set_index('ubigeo_1993')
    summary={'overlap_months':288,'overlap_rows':len(both),'monthly_beta_correlation':float(m.beta_old.corr(m.beta_new)),
             'Empty_count_correlation':float(m.Empty_old.corr(m.Empty_new)),
             'active_count_correlation':float(m.active_old.corr(m.active_new)),
             'interpretation':'Diagnostic changes due to the new B2 and re-estimated pooled scale, support and models; no target matching.'}
    for col in ['descriptive_class','raw_class','fdr_class']:
        changed=both[col+'_new'].ne(both[col+'_old'])
        districts[col+'_changed']=changed.groupby(both.ubigeo_1993).sum()
        summary[col]={'agreement':float((~changed).mean()),'changed_rows':int(changed.sum())}
    districts.sum(axis=1).rename('total_layer_changes').sort_values(ascending=False).to_csv(out/'district_change_ranking.csv')
    districts.to_csv(out/'district_comparison.csv')
    m.sort_values('absolute_beta_change',ascending=False).to_csv(out/'months_largest_beta_changes.csv',index=False)
    m.assign(total_layer_changes=m[[c+'_changed' for c in ['descriptive_class','raw_class','fdr_class']]].sum(axis=1)).sort_values('total_layer_changes',ascending=False).to_csv(out/'months_largest_class_changes.csv',index=False)
    write_json(out/'comparison_summary.json',summary);return summary

def independent_validation():
    p=pd.read_parquet(OUT/'canonical_contract14_local_partial_lisa_19999.parquet')
    desc=pd.read_parquet(OUT/'canonical_contract14_descriptive_panel.parquet')
    for c in desc.columns:pd.testing.assert_series_equal(p[c],desc[c],check_exact=True)
    assert len(p)==516384 and not p.duplicated(['month','ubigeo_1993']).any()
    assert pd.DatetimeIndex(p.month.unique()).equals(MONTHS)
    W=sparse.load_npz(OUT/'data/inputs/peru1793_queen_knn1_row_standardized.npz')
    order=pd.read_parquet(OUT/'data/inputs/peru1793_queen_knn1_id_order.parquet').DI93.to_numpy()
    scale=json.loads((OUT/'metadata/pooled_scale.json').read_text())
    x=p.B2.to_numpy().reshape(288,1793).T
    E=np.isclose(x,x.min(),atol=1e-12,rtol=1e-10);N=np.asarray((W>0)@(~E).astype(int))==0
    np.testing.assert_array_equal(p.E,E.T.ravel());np.testing.assert_array_equal(p.N,N.T.ravel());np.testing.assert_array_equal(p.EN,p.E*p.N)
    np.testing.assert_array_equal(p.z_pool,(p.B2-scale['MU_POOL'])/scale['SD_POOL'])
    np.testing.assert_allclose(p.Wz_full,(W@p.z_pool.to_numpy().reshape(288,1793).T).T.ravel(),atol=1e-12,rtol=1e-12)
    models={k:pd.read_csv(OUT/f'canonical_contract14_monthly_{k}_models.csv',parse_dates=['month'],float_precision='round_trip') for k in ['naive','full','active']}
    errors=[];family=[]
    for t,(month,g) in enumerate(p.groupby('month',sort=True)):
        np.testing.assert_array_equal(g.ubigeo_1993,order)
        active=g.E.eq(0).to_numpy();island=active & g.N.eq(1).to_numpy();interior=active & ~island
        # Independent QR HC1 estimates, including every saved nuisance coefficient.
        for k,cols in [('naive',['z_pool']),('full',['z_pool','E','N','EN']),('active',['z_pool','N'])]:
            gg=g.loc[active] if k=='active' else g
            X=np.column_stack([np.ones(len(gg)),gg[cols].to_numpy(float)]);y=gg.Wz_full.to_numpy()
            Q,R=linalg.qr(X,mode='economic');coef=linalg.solve_triangular(R,Q.T@y)
            residual=y-X@coef; bread=linalg.solve_triangular(R,np.eye(len(cols)+1))
            # X^+ = R^-1 Q^T; HC1 sandwich computed without statsmodels.
            influence=(bread@Q.T)*residual
            cov=(influence@influence.T)*(len(gg)/(len(gg)-X.shape[1]));se=np.sqrt(np.diag(cov))
            row=models[k].iloc[t];expected=[row['alpha_'+k],row['beta_'+k]]
            if k=='full':expected += [row.delta_E,row.delta_N,row.delta_EN]
            elif k=='active':expected += [row.gamma_active]
            np.testing.assert_allclose(coef,expected,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(se[1],row['SE_'+k],atol=1e-10,rtol=1e-10)
            errors.append({'month':month,'family':k,'max_coefficient_error':float(abs(coef-expected).max()),'SE_error':float(abs(se[1]-row['SE_'+k]))})
        xx=g.z_pool-g.loc[active,'z_pool'].mean();adj=g.Wz_full-g.gamma_active*g.N;yy=adj-adj[active].mean()
        np.testing.assert_allclose(g.x_partial,xx,atol=1e-12,rtol=0);np.testing.assert_allclose(g.y_partial,yy,atol=1e-12,rtol=0)
        expected=np.full(1793,'Empty',dtype=object)
        expected[island & (xx>0)]='High Island';expected[island & (xx<0)]='Low Island'
        for xs,ys,label in [(1,1,'HH'),(-1,-1,'LL'),(1,-1,'HL'),(-1,1,'LH')]:expected[interior & (np.sign(xx)==xs)&(np.sign(yy)==ys)]=label
        np.testing.assert_array_equal(expected,g.descriptive_class)
        local=(active.sum()-1)*xx[active]*yy[active]/np.dot(xx[active],xx[active])
        np.testing.assert_allclose(local,g.loc[active,'partial_local_I'],atol=1e-12,rtol=1e-12)
        check=np.load(OUT/f'checkpoints/{month:%Y-%m}.npz',allow_pickle=False)
        upper=check['upper'];raw=(np.minimum(upper,R_PERM-upper)+1)/(R_PERM+1)
        np.testing.assert_array_equal(raw,g.loc[active,'p_raw'])
        np.testing.assert_array_equal(upper,g.loc[active,'upper_tail_count'])
        reject,q,_,_=multipletests(raw,alpha=.05,method='fdr_bh')
        np.testing.assert_array_equal(q,g.loc[active,'q_bh'])
        for sig,cl,flags in [('raw_sig','raw_class',raw<.05),('fdr_sig','fdr_class',reject)]:
            np.testing.assert_array_equal(flags,g.loc[active,sig].to_numpy(bool))
            labels=np.full(1793,'Empty',dtype=object);labels[active]=np.where(flags,expected[active],'NS')
            np.testing.assert_array_equal(labels,g[cl])
        assert g.loc[~active,['partial_local_I','p_raw','q_bh','raw_sig','fdr_sig']].isna().all().all()
        assert g.loc[active,['p_raw','q_bh']].notna().all().all()
        assert not (reject & ~(raw<.05)).any()
        assert not (g.raw_class.eq('NS') & ~g.fdr_class.eq('NS')).any()
        assert np.all((raw>=P_FLOOR)&(raw<=.5))
        family.append({'month':month,'active':int(active.sum()),'tested_islands':int(island.sum()),'RAW':int((raw<.05).sum()),'BH':int(reject.sum()),'BH_exact':True,'masking_only':True})
    # Independently reconcile all count tables with the saved panel.
    for label,field in [('descriptive','descriptive_class'),('raw','raw_class'),('bh','fdr_class')]:
        saved=pd.read_csv(OUT/f'canonical_contract14_{label}_counts.csv',parse_dates=['month']).set_index('month')
        actual=pd.crosstab(p.month,p[field]).reindex(columns=saved.columns,fill_value=0)
        np.testing.assert_array_equal(actual,saved);assert saved.sum(axis=1).eq(1793).all()
    # Reconstruct persistence through run boundaries, independently of the production loop.
    persistence_checked=0
    for layer,sig in [('raw','raw_sig'),('bh','fdr_sig')]:
        saved=pd.read_parquet(OUT/f'canonical_contract14_{layer}_persistence.parquet').set_index(['ubigeo_1993','class_scope'])
        for uid,g in p.groupby('ubigeo_1993',sort=False):
            assert pd.DatetimeIndex(g.month).equals(MONTHS)
            for scope in ['ALL',*ACTIVE_CLASSES]:
                mask=g[sig].fillna(False).to_numpy(bool)
                if scope!='ALL':mask &= g.descriptive_class.eq(scope).to_numpy()
                transitions=np.diff(np.r_[False,mask,False].astype(int))
                starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
                longest=int((ends-starts).max()) if len(starts) else 0
                r=saved.loc[(uid,scope)]
                assert int(r.total_significant_months)==int(mask.sum()) and int(r.longest_consecutive_run)==longest
                if mask.any():
                    dates=g.loc[mask,'month'];assert r.first_significant_month==dates.iloc[0] and r.last_significant_month==dates.iloc[-1]
                    counts=g.loc[mask,'descriptive_class'].value_counts();dominant=next(c for c in ACTIVE_CLASSES if counts.get(c,0)==counts.max())
                    assert r.dominant_significant_class==dominant
                else:assert pd.isna(r.first_significant_month) and pd.isna(r.last_significant_month) and pd.isna(r.dominant_significant_class)
                persistence_checked+=1
    pd.DataFrame(errors).to_csv(OUT/'audits/independent_model_validation.csv',index=False)
    pd.DataFrame(family).to_csv(OUT/'audits/independent_inference_validation.csv',index=False)
    summary={'status':'PASS','rows':len(p),'months':288,'models_independently_checked':len(errors),
             'maximum_coefficient_error':max(r['max_coefficient_error'] for r in errors),
             'maximum_SE_error':max(r['SE_error'] for r in errors),'classifications_checked':len(p)*3,
             'BH_exact_months':288,'all_active_tested':True,'Empty_untested':True,'islands_tested':int((p.E.eq(0)&p.N.eq(1)).sum()),'persistence_rows_checked':persistence_checked}
    write_json(OUT/'audits/independent_validation.json',summary)
    return summary

def export():
    for src in OUT.glob('canonical_contract14_*'):
        if src.suffix=='.parquet':dest=OUT/'data'/src.name
        elif src.suffix=='.csv':dest=OUT/'tables'/src.name
        else:continue
        shutil.copy2(src,dest)
    env=json.loads((OUT/'metadata/environment.json').read_text());scale=json.loads((OUT/'metadata/pooled_scale.json').read_text())
    meta={'contract':14,'status':'AWAITING USER REVIEW','panel':{'districts':1793,'months':288,'rows':516384,'start':'2001-01','end':'2024-12'},
          'scale':scale,'environment':env,'palette':COLORS,'inference':{'statistic':'(m-1)*x_partial*y_partial/sum_active(x_partial**2)','null':'Joint active z permutations, fixed W/E/N/Empty/gamma','upper':'simulated >= observed','lower':'strict complement','p':f'(min(upper,lower)+1)/{R_PERM+1}','RAW':'p<.05','BH':'q<=.05; one monthly E=0 family including islands'},
          'display':'NS masks nonsignificant active interior only, following final C12 amendment; island colors are structural context, not significance. Canonical raw_class/fdr_class mask all active observations.',
          'limitations':['Monthly BH does not provide joint control across time','HC1 uncertainty is pointwise','Spatial association is not causal','Fresh hosted Colab execution not performed here; local GPU execution and pinned remote retrieval tested']}
    write_json(OUT/'metadata/canonical_contract14_metadata.json',meta)
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(OUT)
        if any(s in rel.parts for s in ['__pycache__','runtime_cache','remote_verification','.cache','vendor']):continue
        if rel.parts[0]=='manifests' or p.suffix=='.log':continue
        files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(files).to_csv(OUT/'manifests/contract14_manifest.csv',index=False)
    return meta
if __name__=='__main__':
    print(independent_validation())
    print(comparison(pd.read_parquet(OUT/'canonical_contract14_local_partial_lisa_19999.parquet')))
