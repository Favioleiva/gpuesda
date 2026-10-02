"""Frozen input loading, mandatory scientific audits and production checkpoints."""
from pathlib import Path
import json
import hashlib
import platform
import time
import importlib.metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import psutil
from statsmodels.stats.multitest import multipletests
from partial_lisa import simulate, cpu_reference, draw_orders, bh, pseudo_p, w_identity, R_PERM, P_FLOOR


from workflow import OUT, sha, write_json

def engine_validation(panel,W,observed):
    evidence = []
    for month_index in (0,144,287):
        g = panel.iloc[month_index*1793:(month_index+1)*1793]
        a = g.E.eq(0).to_numpy()
        args = (g.z_pool.to_numpy(),W,a,g.N.to_numpy(),float(g.gamma_active.iloc[0]),observed[g.index[a]])
        runs = [simulate(*args,seed=12345+month_index,R=67,batch_size=b,keep=True,audit=True) for b in (1,17,64)]
        again = simulate(*args,seed=12345+month_index,R=67,batch_size=17,keep=True)
        np.testing.assert_array_equal(runs[1]['simulations'],again['simulations'])
        for r in runs[1:]:
            np.testing.assert_allclose(r['simulations'],runs[0]['simulations'],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(r['upper'],runs[0]['upper'])
            assert r['audit']['draw_sha256']==runs[0]['audit']['draw_sha256']
        ref = cpu_reference(*args[:5],draw_orders(int(a.sum()),67,12345+month_index))
        np.testing.assert_array_equal(np.count_nonzero(ref>=args[5],axis=0),runs[0]['upper'])
        evidence.append({'month':str(g.month.iloc[0]),'pass':True,'tests':'CPU/GPU identical draws; observed NumPy; exact seeded repeat; batching counts; fixed W/E/N/Empty; donor multiset',
                         'audits':[r['audit'] for r in runs]})
    write_json(OUT/'audits/engine_validation.json',evidence)
    return evidence

def production(panel,W,observed):
    import cupy as cp
    (OUT/'checkpoints').mkdir(exist_ok=True)
    source_hash = sha(Path(__file__).with_name('partial_lisa.py'))
    panel_hash = sha(OUT/'canonical_contract14_descriptive_panel.parquet')
    all_rows, audits = [],[]
    for t,(month,g) in enumerate(panel.groupby('month',sort=True)):
        a = g.E.eq(0).to_numpy()
        checkpoint = OUT/f'checkpoints/{month:%Y-%m}.npz'
        identity = hashlib.sha256((source_hash+panel_hash+str(w_identity(W))+str(t)).encode()).hexdigest()
        if checkpoint.exists():
            saved = np.load(checkpoint,allow_pickle=False)
            assert str(saved['identity']) == identity
            upper = saved['upper']
            audit = json.loads(str(saved['audit']))
        else:
            r = simulate(g.z_pool.to_numpy(),W,a,g.N.to_numpy(),float(g.gamma_active.iloc[0]),observed[g.index[a]],seed=12345+t,R=R_PERM)
            upper,audit = r['upper'],r['audit']
            audit['cpu_rss_bytes'] = psutil.Process().memory_info().rss
            audit['cpu_peak_wset_bytes'] = getattr(psutil.Process().memory_info(),'peak_wset',None)
            np.savez_compressed(checkpoint,upper=upper,identity=identity,audit=json.dumps(audit))
        p = pseudo_p(upper,R_PERM)
        q = bh(p)
        reject,q_ref,_,_ = multipletests(p,alpha=.05,method='fdr_bh')
        np.testing.assert_array_equal(q,q_ref)
        np.testing.assert_array_equal(q<=.05,reject)
        out = g.copy()
        out['partial_local_I'] = observed[g.index]
        for col in ('p_raw','q_bh','upper_tail_count','lower_tail_count'):
            out[col] = np.nan
        lower = R_PERM - upper
        out.loc[a,'p_raw'],out.loc[a,'q_bh'] = p,q
        out.loc[a,'upper_tail_count'],out.loc[a,'lower_tail_count'] = upper,lower
        out['raw_sig'] = pd.array([pd.NA]*len(g),dtype='boolean')
        out['fdr_sig'] = pd.array([pd.NA]*len(g),dtype='boolean')
        out.loc[a,'raw_sig'],out.loc[a,'fdr_sig'] = p<.05,q<=.05
        for sig,cl in [('raw_sig','raw_class'),('fdr_sig','fdr_class')]:
            out[cl] = np.where(~a,'Empty',np.where(out[sig].fillna(False),out.descriptive_class,'NS'))
        all_rows.append(out)
        audits.append({'month':str(month),**audit,'checkpoint_sha256':sha(checkpoint),'bh_exact_match':True})
        if t%25==0 or t==287:
            print(f'{t+1}/288 months | {month:%Y-%m} | active={a.sum()} RAW={(p<.05).sum()} BH={(q<=.05).sum()}',flush=True)
    result = pd.concat(all_rows,ignore_index=True)
    result.to_parquet(OUT/'canonical_contract14_local_partial_lisa_19999.parquet',index=False)
    write_json(OUT/'audits/production_runtime.json',{'months':audits,'inference_seconds':sum(a['seconds'] for a in audits),
        'mean_seconds_per_month':np.mean([a['seconds'] for a in audits]),
        'peak_cupy_pool_bytes':max(a['sampled_peak_cupy_pool_bytes'] for a in audits),
        'peak_cpu_rss_bytes':max(a['cpu_rss_bytes'] for a in audits),
        'memory_scope':'CuPy reserved pool sampled each batch; CPU RSS each month plus Windows peak working set',
        'environment':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','statsmodels','matplotlib','pyarrow']},
        'python':platform.python_version(),'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),'cuda_driver':cp.cuda.runtime.driverGetVersion()})
    return result
