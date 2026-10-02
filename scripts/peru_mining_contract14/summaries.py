import numpy as np
import pandas as pd
from workflow import OUT, ACTIVE_CLASSES, CLASSES, write_json
from partial_lisa import P_FLOOR


def persistence(panel, layer):
    rows = []
    for uid,g in panel.groupby('ubigeo_1993',sort=False):
        for label in ['ALL',*ACTIVE_CLASSES]:
            sig = g[layer+'_sig'].fillna(False).to_numpy(dtype=bool)
            if label != 'ALL':
                sig &= g.descriptive_class.eq(label).to_numpy()
            idx = np.flatnonzero(sig)
            longest = current = 0
            for v in sig:
                current = current+1 if v else 0
                longest = max(longest,current)
            counts = g.loc[sig,'descriptive_class'].value_counts()
            # Deterministic ties follow canonical class order.
            dominant = next((c for c in ACTIVE_CLASSES if counts.get(c,0)==counts.max()),None) if len(counts) else None
            rows.append({'ubigeo_1993':uid,'class_scope':label,'total_significant_months':len(idx),
                         'longest_consecutive_run':longest,'first_significant_month':g.month.iloc[idx[0]] if len(idx) else pd.NaT,
                         'last_significant_month':g.month.iloc[idx[-1]] if len(idx) else pd.NaT,
                         'dominant_significant_class':dominant})
    return pd.DataFrame(rows)


def summarize(panel):
    diagnostics, shares, floor, summary = [],[],[],[]
    for month,g in panel.groupby('month',sort=True):
        a = g.loc[g.E.eq(0)]
        m = len(a)
        p_floor = P_FLOOR
        floor_count = int(a.p_raw.eq(p_floor).sum())
        d = {'month':month,'N_active':m,'N_interior':int(a.N.eq(0).sum()),'N_islands':int(a.N.eq(1).sum()),
             'min_p':a.p_raw.min(),'min_q':a.q_bh.min(),'p_floor_count':floor_count,'beta_t':g.beta_active.iloc[0]}
        min_rank = int(np.ceil(p_floor*m/.05-1e-12))
        floor.append({'month':month,'m_t':m,'first_BH_critical':.05/m,'p_floor':p_floor,
                      'p_floor_count':floor_count,'minimum_rank_for_floor_rejection':min_rank,
                      'floor_exceeds_first_critical':p_floor>.05/m,
                      'floor_hits_not_rejected':int((a.p_raw.eq(p_floor)&a.q_bh.gt(.05)).sum()),
                      'potentially_resolution_limited':bool(p_floor>.05/m and floor_count>0),
                      'observed_floor_blocked':bool((a.p_raw.eq(p_floor)&a.q_bh.gt(.05)).any())})
        for tag,field in [('RAW','raw'),('BH','fdr')]:
            sig = a[field+'_sig'].astype(bool)
            d[tag+'_discoveries'] = int(sig.sum())
            for c in ACTIVE_CLASSES:
                d[tag+'_'+c.replace(' ','_')] = int((sig & a.descriptive_class.eq(c)).sum())
            for c in ['ALL',*ACTIVE_CLASSES]:
                subset = a if c=='ALL' else a.loc[a.descriptive_class.eq(c)]
                num = int(subset[field+'_sig'].sum())
                shares.append({'month':month,'layer':tag,'descriptive_class':c,'eligible':len(subset),
                               'significant':num,'significant_share':num/len(subset) if len(subset) else np.nan})
        diagnostics.append(d)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(OUT/'canonical_contract14_monthly_diagnostics.csv',index=False)
    floor = pd.DataFrame(floor)
    floor.to_csv(OUT/'canonical_contract14_monthly_bh.csv',index=False)
    pd.DataFrame(shares).to_csv(OUT/'canonical_contract14_significance_shares.csv',index=False)
    for tag,field in [('raw','raw'),('bh','fdr')]:
        counts = pd.crosstab(panel.month,panel[field+'_class']).reindex(columns=CLASSES,fill_value=0)
        assert counts.sum(1).eq(1793).all()
        counts.to_csv(OUT/f'canonical_contract14_{tag}_counts.csv')
        persistence(panel,field).to_parquet(OUT/f'canonical_contract14_{tag}_persistence.parquet',index=False)
        for c in ['ALL',*ACTIVE_CLASSES]:
            a = panel.loc[panel.E.eq(0)]
            if c!='ALL':
                a = a.loc[a.descriptive_class.eq(c)]
            s = a.loc[a[field+'_sig'].astype(bool)]
            summary.append({'layer':tag.upper(),'descriptive_class':c,'active_district_month_tests':len(a),
                            'significant_observations':len(s),'share_significant':len(s)/len(a) if len(a) else np.nan,
                            'months_with_discovery':s.month.nunique(),'districts_ever_significant':s.ubigeo_1993.nunique()})
    pd.DataFrame(summary).to_csv(OUT/'canonical_contract14_discovery_summary.csv',index=False)
    candidates = []
    for c in ['RAW_discoveries','BH_discoveries',*[f'BH_{c}' for c in ['HH','LL','HL','LH']],
              'RAW_High_Island','RAW_Low_Island','BH_High_Island','BH_Low_Island']:
        mx = diag[c].max()
        tied = diag.loc[diag[c].eq(mx),'month']
        candidates.append({'reason':'highest '+c,'month':tied.iloc[0],'value':int(mx),'tied_months':len(tied)})
    for label,i in [('earliest',0),('middle',144),('latest',287)]:
        candidates.append({'reason':label,'month':diag.month.iloc[i],'value':np.nan,'tied_months':1})
    pd.DataFrame(candidates).to_csv(OUT/'canonical_contract14_key_months.csv',index=False)
    write_json(OUT/'audits/floor_interpretation.json',{
        'months_floor_above_first_BH_threshold':int(floor.floor_exceeds_first_critical.sum()),
        'months_potentially_resolution_limited_with_floor_hits':int(floor.potentially_resolution_limited.sum()),
        'months_observed_floor_hits_blocked':int(floor.observed_floor_blocked.sum()),
        'interpretation':'A floor above alpha/m does not by itself preclude BH rejection: BH is step-up. Potential limitation flags floor hits where rank 1 cannot pass; observed blockage counts floor hits that actually fail BH. True undiscovered sub-floor p-values cannot be inferred from finite draws.'})
    return diag,pd.DataFrame(summary),floor
