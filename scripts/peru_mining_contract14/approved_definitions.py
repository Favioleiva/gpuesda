def fit_month(g, predictors, family):
    design = sm.add_constant(g[predictors].astype(float), has_constant='add')
    rank = int(np.linalg.matrix_rank(design.to_numpy()))
    assert rank == design.shape[1], f'Rank deficiency: {family} {g.month.iloc[0]}'
    result = sm.OLS(g.Wz_full.to_numpy(), design).fit(cov_type='HC1', use_t=False)
    assert np.isfinite(result.params).all() and np.isfinite(result.bse).all()
    ci = result.conf_int().loc['z_pool']
    suffix = family
    row = {'month':g.month.iloc[0], 'N_'+suffix:len(g), 'alpha_'+suffix:float(result.params['const']),
        'beta_'+suffix:float(result.params.z_pool), 'SE_'+suffix:float(result.bse.z_pool),
        't_'+suffix:float(result.tvalues.z_pool), 'p_'+suffix:float(result.pvalues.z_pool),
        'CI95_low_'+suffix:float(ci.iloc[0]), 'CI95_high_'+suffix:float(ci.iloc[1]),
        'R2_'+suffix:float(result.rsquared)}
    DESIGN_AUDIT.append({'month':g.month.iloc[0], 'family':family, 'rows':len(g),
        'rank':rank, 'columns':design.shape[1], 'condition_number':float(result.condition_number)})
    return row, result

def partial_scatter(g):
    month=g.month.iloc[0];beta=float(g.beta_active.iloc[0])
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained',sharex=True,sharey=True)
    for ax,labels,title in [(axs[0],CLASSES[:4],'Active interior'),(axs[1],['Empty','High Island','Low Island'],'Structural context')]:
        for label in labels:
            q=g.loc[g.descriptive_class.eq(label)]
            marker='^' if label in ['High Island','Low Island'] else ('s' if label=='Empty' else 'o')
            ax.scatter(q.x_partial,q.y_partial,s=17 if label!='Empty' else 9,c=COLORS[label],
                marker=marker,alpha=.75 if label!='Empty' else .3,
                edgecolors='white' if ' Island' in label else 'none',linewidths=.45,
                rasterized=True,label=f'{label} (n={len(q):,})')
        ax.axhline(0,color='gray',lw=.65);ax.axvline(0,color='gray',lw=.65)
        ax.set(xlim=X_LIMIT,ylim=Y_LIMIT,xlabel='Centered focal activity (pooled units)',title=title)
        ax.legend(fontsize=8,loc='upper left')
    xx=np.array(X_LIMIT)
    axs[0].plot(xx,beta*xx,color='black',lw=1.3,label='Monthly adjusted slope')
    axs[0].set_ylabel('Adjusted centered lag (pooled units)')
    fig.suptitle(f'Descriptive partial spatial-association scatter · {month:%Y-%m} · β = {beta:.4f}')
    fig.supxlabel('Descriptive classes; no significance filter. Common monthly active reference; no SD division.',fontsize=9)
    return fig

def map_base():
    fig,ax=plt.subplots(figsize=(8.8,8),layout='constrained')
    collection=PatchCollection(patches,match_original=False,rasterized=True,edgecolor='none',linewidth=0,antialiased=False)
    ax.add_collection(collection)
    padx=(xmax-xmin)*.025;pady=(ymax-ymin)*.025
    ax.set_xlim(xmin-padx,xmax+padx);ax.set_ylim(ymin-pady,ymax+pady)
    ax.set_aspect(1/np.cos(np.deg2rad((ymin+ymax)/2)) if geo.crs.is_geographic else 'equal')
    ax.set_axis_off()
    handles=[Patch(facecolor=COLORS[c],edgecolor='none',linewidth=0,label=c) for c in CLASSES]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.01,1),frameon=True,title='Descriptive states')
    fig.supxlabel('Full district universe: 1,793 districts · fixed geography · descriptive, no significance filter',fontsize=9)
    return fig,ax,collection

def color_map(collection,g):
    assert np.array_equal(g.ubigeo_1993.astype(str),district_order)
    labels=g.descriptive_class.to_numpy()[owners]
    collection.set_facecolors([COLORS[c] for c in labels])
    collection.set_edgecolors('none')
    collection.set_linewidths(0)
    collection.set_antialiased(False)
    assert collection.get_edgecolors().size==0 and np.all(collection.get_linewidths()==0)
