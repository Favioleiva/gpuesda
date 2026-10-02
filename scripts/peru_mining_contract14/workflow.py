"""Contract 14: final harmonized B2 on the approved fixed spatial system."""
from pathlib import Path
import os, json, hashlib, platform, time, importlib.metadata, sys
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import statsmodels.api as sm
OUT=Path(os.environ.get('CONTRACT14_OUTPUT',str(Path(__file__).resolve().parents[1]))).resolve()
INPUT=OUT/'data/inputs'
COLORS={'HH':'#C51B2D','LL':'#2166AC','HL':'#F4A582','LH':'#92C5DE','High Island':'#984EA3','Low Island':'#1B9E77','NS':'#BDBDBD','Empty':'#333333'}
CLASSES=list(COLORS); ACTIVE_CLASSES=CLASSES[:6]
MONTHS=pd.date_range('2001-01-01','2024-12-01',freq='MS')
COMMIT='b7ea60fab197473329a6caa50975c01bdd58c71f'
from partial_lisa import R_PERM, P_FLOOR
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def table(df,name): df.to_csv(OUT/f'canonical_contract14_{name}.csv',index=False,float_format='%.17g')
def environment():
    for d in ['audits','checkpoints','figures','frames','videos','data','tables','metadata','manifests','comparison_old_vs_new']:(OUT/d).mkdir(parents=True,exist_ok=True)
    import cupy as cp, gpu_esda
    assert cp.cuda.runtime.getDeviceCount()>0,'A CUDA GPU is mandatory; no production CPU fallback'
    assert int(cp.asnumpy(cp.arange(5).sum()))==10
    # The notebook extracts the approved package archive; inspect that source, not a mutable installation.
    record=json.loads((INPUT/'remote_inputs.json').read_text())
    package=Path(gpu_esda.__file__).parent
    for r in record['gpuesda_source_files']:
        p=package/Path(r['path'].replace('\\','/')).relative_to('src/gpu_esda')
        assert sha(p)==r['sha256'],f'GPUESDA source differs from pinned commit: {p.name}'
    env={'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,
         'python':platform.python_version(),'gpuesda_version':'0.2.0','gpuesda_commit':COMMIT,
         'gpuesda_source_files_verified':len(record['gpuesda_source_files']),
         'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','geopandas','matplotlib','pyarrow','psutil','Pillow','imageio-ffmpeg']},
         'seed':'12345 + zero-based month index','R':R_PERM,'batch_size':256,'dtype':'float64',
         'custom_extension_sha256':sha(Path(__file__).with_name('partial_lisa.py'))}
    write_json(OUT/'metadata/environment.json',env)
    return env
def load_inputs():
    rec=json.loads((INPUT/'remote_inputs.json').read_text()); paths={}
    for k,r in rec['inputs'].items():
        p=INPUT/r['filename']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],k
        paths[k]=p
    p=pd.read_parquet(paths['B2'],columns=['district_id','month','B2']).rename(columns={'district_id':'ubigeo_1993'})
    p.month=pd.to_datetime(p.month)
    assert len(p)==1793*288==516384 and not p.duplicated(['month','ubigeo_1993']).any()
    assert pd.DatetimeIndex(sorted(p.month.unique())).equals(MONTHS) and np.isfinite(p.B2).all()
    order_frame=pd.read_parquet(paths['order']); assert np.array_equal(order_frame.matrix_position,np.arange(1793))
    order=order_frame.DI93.astype(str).to_numpy()
    np.testing.assert_array_equal(pd.read_parquet(paths['registry']).sort_values('matrix_position').district_id,order)
    assert set(p.ubigeo_1993)==set(order) and len(set(order))==1793
    idx=pd.MultiIndex.from_product([MONTHS,order],names=['month','ubigeo_1993'])
    p=p.set_index(['month','ubigeo_1993']).reindex(idx).reset_index(); assert p.B2.notna().all()
    W=sparse.load_npz(paths['W'])
    assert W.format=='csr' and W.shape==(1793,1793) and W.nnz==10418 and W.has_canonical_format
    assert (W.data>0).all() and (W.diagonal()==0).all()
    np.testing.assert_allclose(np.asarray(W.sum(1)).ravel(),1,atol=1e-12,rtol=0)
    for a in [W.data,W.indices,W.indptr]:a.setflags(write=False)
    import geopandas as gpd
    geo=gpd.read_parquet(paths['geometry']); assert set(geo.DI93)==set(order) and geo.geometry.is_valid.all()
    write_json(OUT/'audits/input_validation.json',{'status':'PASS','rows':len(p),'districts':1793,'months':288,'start':'2001-01','end':'2024-12','inputs':rec['inputs'],'W_nnz':W.nnz,'registry_order_matches':True})
    return p,W,order,paths
def structural_scale(p,W):
    x=p.B2.to_numpy().reshape(288,1793).T
    b0=float(x.min()); E=np.isclose(x,b0,atol=1e-12,rtol=1e-10)
    A=(W>0).astype(np.int64); N=A@(~E).astype(np.int64)==0
    reference=~E & ~N; initial=reference.copy(); rounds=0
    while True:
        nxt=reference & ((A@reference.astype(np.int64))>0)
        if np.array_equal(nxt,reference): break
        reference=nxt; rounds+=1
    pool=x[reference]; mu=float(pool.mean()); sd=float(pool.std(ddof=0))
    assert len(pool)>0 and np.isfinite([mu,sd]).all() and sd>0
    p['E']=E.T.ravel().astype(np.int8);p['N']=N.T.ravel().astype(np.int8);p['EN']=p.E*p.N
    p['pooled_reference_member']=reference.T.ravel()
    p['structural_state']=np.select([p.E.eq(1)&p.N.eq(1),p.E.eq(1)&p.N.eq(0),p.E.eq(0)&p.N.eq(1)],['Minimum-Minimum','Empty-Focal','Active-Island'],default='Interior')
    p['z_pool']=(p.B2-mu)/sd
    meta={'MU_POOL':mu,'SD_POOL':sd,'B2_MIN':b0,'ddof':0,'atol':1e-12,'rtol':1e-10,
          'reference_count':int(reference.sum()),'reference_districts':int(reference.any(axis=1).sum()),
          'reference_members_removed_after_initial_selection':int(initial.sum()-reference.sum()),'reference_membership_iterations':rounds,
          'reference_definition':'Inherited Contract5 retained sample: E=0,N=0 followed by zero-out-degree membership removal until stable; equal district-month weighting. This is a scaling-reference mask only. Full W, geography and active inference family are unchanged.',
          'monthly_SD_division':False,'structural_definition':'E=isclose(B2,global minimum); N=all positive-weight outgoing neighbors E; EN=E*N'}
    write_json(OUT/'metadata/pooled_scale.json',meta)
    counts=p.groupby(['month','structural_state']).size().unstack(fill_value=0);table(counts.reset_index(),'structural_counts')
    return p,meta,counts
def lag(p,W):
    from gpu_esda.lag import spatial_lag
    z=p.z_pool.to_numpy().reshape(288,1793).T
    result=spatial_lag(W.copy(),z,backend='gpu')
    np.testing.assert_allclose(result,W@z,atol=1e-12,rtol=1e-12)
    p['Wz_full']=result.T.ravel(); return p
def regress(p):
    import approved_definitions as ad
    ad.sm=sm;ad.np=np;ad.DESIGN_AUDIT=[]
    frames={}
    for family,predictors in [('full',['z_pool','E','N','EN']),('active',['z_pool','N']),('naive',['z_pool'])]:
        rows=[]; subset=p.loc[p.E.eq(0)] if family=='active' else p
        for month,g in subset.groupby('month',sort=True):
            row,fit=ad.fit_month(g,predictors,family)
            if family=='full':row.update(delta_E=float(fit.params.E),delta_N=float(fit.params.N),delta_EN=float(fit.params.EN))
            if family=='active':row.update(gamma_active=float(fit.params.N),active_island_count=int(g.N.sum()))
            rows.append(row)
        frames[family]=pd.DataFrame(rows);table(frames[family],f'monthly_{family}_models')
    full,active=frames['full'],frames['active']
    err=full.beta_full.to_numpy()-active.beta_active.to_numpy()
    np.testing.assert_allclose(full.beta_full,active.beta_active,atol=1e-10,rtol=1e-10)
    audit={'status':'PASS','months':288,'max_absolute_slope_difference':float(abs(err).max()),'rmse':float(np.sqrt(np.mean(err**2))),'covariance':'HC1, normal reference use_t=False'}
    write_json(OUT/'audits/slope_equivalence.json',audit);table(pd.DataFrame(ad.DESIGN_AUDIT),'design_audit')
    return frames,audit
def coordinates(p,models):
    a=models['active'];p=p.merge(a[['month','alpha_active','beta_active','gamma_active']],on='month',validate='many_to_one',sort=False)
    p['Wz_adj']=p.Wz_full-p.gamma_active*p.N
    ref=p.loc[p.E.eq(0)].groupby('month').agg(focal_reference=('z_pool','mean'),lag_reference=('Wz_adj','mean')).reset_index()
    p=p.merge(ref,on='month',validate='many_to_one',sort=False)
    p['x_partial']=p.z_pool-p.focal_reference;p['y_partial']=p.Wz_adj-p.lag_reference
    np.testing.assert_allclose(p.loc[p.E.eq(0)].groupby('month')[['x_partial','y_partial']].mean(),0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(p.lag_reference,p.alpha_active+p.beta_active*p.focal_reference,atol=1e-12,rtol=0)
    table(ref,'monthly_references');return p
def classify(p):
    interior=p.E.eq(0)&p.N.eq(0);island=p.E.eq(0)&p.N.eq(1);empty=p.E.eq(1);x=p.x_partial;y=p.y_partial
    p['axis_tie']=((interior|island)&x.eq(0))|(interior&y.eq(0))
    assert not p.axis_tie.any(),'Exact-zero classification ties require user review'
    p['descriptive_class']=np.select([empty,island&x.gt(0),island&x.lt(0),interior&x.gt(0)&y.gt(0),interior&x.lt(0)&y.lt(0),interior&x.gt(0)&y.lt(0),interior&x.lt(0)&y.gt(0)],['Empty','High Island','Low Island','HH','LL','HL','LH'],default='UNASSIGNED')
    assert set(p.descriptive_class)<=set(ACTIVE_CLASSES+['Empty'])
    counts=pd.crosstab(p.month,p.descriptive_class).reindex(columns=ACTIVE_CLASSES+['Empty'],fill_value=0)
    assert counts.sum(1).eq(1793).all();table(counts.reset_index(),'descriptive_counts')
    p.to_parquet(OUT/'canonical_contract14_descriptive_panel.parquet',index=False)
    return p,counts
def observed_audit(p,W,order):
    from partial_lisa import cpu_reference
    obs=np.full(len(p),np.nan);rows=[]
    for month,g in p.groupby('month',sort=True):
        np.testing.assert_array_equal(g.ubigeo_1993,order)
        active=g.E.eq(0).to_numpy();x=g.loc[active,'x_partial'].to_numpy();y=g.loc[active,'y_partial'].to_numpy();m=active.sum()
        local=(m-1)*x*y/np.dot(x,x);beta=np.dot(x,y)/np.dot(x,x)
        np.testing.assert_allclose([beta,local.sum()/(m-1)],g.beta_active.iloc[0],atol=1e-10,rtol=1e-10)
        independent=cpu_reference(g.z_pool.to_numpy(),W,active,g.N.to_numpy(),g.gamma_active.iloc[0],np.arange(m)[None,:])[0]
        np.testing.assert_allclose(local,independent,atol=1e-12,rtol=1e-12)
        obs[g.index[active]]=local;rows.append({'month':month,'beta':beta,'local_decomposition':local.sum()/(m-1),'error':abs(beta-g.beta_active.iloc[0])})
    table(pd.DataFrame(rows),'observed_decomposition');return obs,pd.DataFrame(rows)
