from pathlib import Path
import sys, ast, io, json, time, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
import geopandas as gpd
from shapely.geometry.polygon import orient


from workflow import OUT, sha, write_json, MONTHS
BASE_COLORS={'HH':'#C51B2D','LL':'#2166AC','HL':'#F4A582','LH':'#92C5DE','High Island':'#984EA3','Low Island':'#1B9E77','Empty':'#333333'}
COLORS={**BASE_COLORS,'NS':'#BDBDBD'}
CLASSES=list(COLORS)
SIZES={'map':(912,1056),'scatter':(1446,640),'flagship':(2382,1056)}
RAW_SIZES={'map':(912,1055),'scatter':(1445,639)}

class OriginalArt:
    def __init__(self,p,geometry):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        geo=gpd.read_parquet(geometry)
        order=p.ubigeo_1993.iloc[:1793].to_numpy()
        geo.index=geo.DI93.astype(str).str.zfill(6);geo=geo.loc[order].copy()
        patches,owners=[],[]
        for i,geom in enumerate(geo.geometry):
            for poly in ([geom] if geom.geom_type=='Polygon' else geom.geoms):
                poly=orient(poly,sign=1.)
                vertices,codes=[],[]
                for ring in [poly.exterior,*poly.interiors]:
                    coords=np.asarray(ring.coords);code=np.full(len(coords),MplPath.LINETO)
                    code[0]=MplPath.MOVETO;code[-1]=MplPath.CLOSEPOLY
                    vertices.extend(coords);codes.extend(code)
                patches.append(PathPatch(MplPath(np.asarray(vertices),np.asarray(codes))));owners.append(i)
        xmin,ymin,xmax,ymax=geo.total_bounds
        self.env=dict(np=np,plt=plt,Patch=Patch,PatchCollection=PatchCollection,patches=patches,owners=np.asarray(owners),
                      xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,geo=geo,district_order=order,
                      X_LIMIT=(float(p.x_partial.min())-.2,float(p.x_partial.max())+.2),
                      Y_LIMIT=(float(p.y_partial.min())-.2,float(p.y_partial.max())+.2),
                      CLASSES=list(BASE_COLORS),COLORS=BASE_COLORS)
        source=Path(__file__).with_name('approved_definitions.py').read_text(encoding='utf-8')
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ['partial_scatter','map_base','color_map']]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'approved_C11_renderer','exec'),self.env)
        self.geometry_hash=sha(geometry)
        self.audit=[]

    def establish(self,fig,kind):
        # Run the original save operation first, then lock its layout and crop.
        buf=io.BytesIO();fig.savefig(buf,dpi=130,bbox_inches='tight')
        im=Image.open(buf).convert('RGB')
        assert im.size==RAW_SIZES[kind],(kind,im.size)
        fig.set_layout_engine('none');fig.set_dpi(130);fig.canvas.draw()
        bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
        positions=[a.get_position().bounds for a in fig.axes]
        return bbox,positions,im

    def save(self,fig,bbox,positions,kind,path):
        for ax,pos in zip(fig.axes,positions):np.testing.assert_allclose(ax.get_position().bounds,pos,atol=0,rtol=0)
        buf=io.BytesIO();fig.savefig(buf,dpi=130,bbox_inches=bbox)
        im=Image.open(buf).convert('RGB')
        # Same native C11 image dimensions, followed only by its original even padding.
        assert im.size==RAW_SIZES[kind],(kind,im.size)
        canvas=Image.new('RGB',SIZES[kind],'white');canvas.paste(im,(0,0));canvas.save(path)
        return canvas

    def map(self,g,layer,path):
        fig,ax,coll=self.env['map_base']()
        self.env['color_map'](coll,g)
        ax.set_title(f'Descriptive spatial mining states · {g.month.iloc[0]:%Y-%m}')
        bbox,pos,reference=self.establish(fig,'map')
        owners=self.env['owners'];labels=g[layer+'_display_class'].to_numpy()[owners]
        coll.set_facecolors([COLORS[c] for c in labels])
        assert coll.get_edgecolors().size==0 and np.all(coll.get_linewidths()==0)
        old=ax.get_legend()
        ax.legend(handles=[Patch(facecolor=COLORS[c],edgecolor='none',linewidth=0,label=c) for c in CLASSES],
                  loc='upper left',bbox_to_anchor=(1.01,1),frameon=True,title='Spatial states')
        title={'raw':'RAW p < .05','bh':'BH-FDR q ≤ .05','descriptive':'Descriptive'}[layer]
        ax.set_title(f'{title} spatial mining states · {g.month.iloc[0]:%Y-%m}')
        fig._supxlabel.set_text('Full district universe: 1,793 districts · fixed geography · '+('descriptive, no significance filter' if layer=='descriptive' else 'NS masks active interior only'))
        out=self.save(fig,bbox,pos,'map',path)
        plt.close(fig)
        return out,pos,reference

    def scatter(self,g,layer,path):
        fig=self.env['partial_scatter'](g)
        bbox,pos,reference=self.establish(fig,'scatter')
        before=[(c.get_offsets().copy(),c.get_sizes().copy()) for ax in fig.axes for c in ax.collections]
        for c,label in zip(fig.axes[0].collections,list(BASE_COLORS)[:4]):
            q=g.loc[g.descriptive_class.eq(label)]
            colors=[COLORS[v] for v in q[layer+'_display_class']]
            c.set_facecolors(colors)
        counts=g[layer+'_display_class'].value_counts()
        handles=[Line2D([],[],linestyle='',marker='o',markersize=4,color=COLORS[c],alpha=.75,label=f'{c} (n={counts.get(c,0):,})') for c in ['HH','LL','HL','LH','NS']]
        fig.axes[0].legend(handles=handles,fontsize=8,loc='upper left')
        title={'raw':'RAW p < .05','bh':'BH-FDR q ≤ .05','descriptive':'Descriptive'}[layer]
        fig._suptitle.set_text(f'{title} partial spatial-association scatter · {g.month.iloc[0]:%Y-%m} · β = {g.beta_active.iloc[0]:.4f}')
        fig._supxlabel.set_text('Common active reference; no SD division. '+('Descriptive; no significance filter.' if layer=='descriptive' else 'NS masks active interior only; islands retain context.'))
        after=[(c.get_offsets(),c.get_sizes()) for ax in fig.axes for c in ax.collections]
        for (x,s),(xx,ss) in zip(before,after):
            np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(s,ss)
        for ax in fig.axes:
            assert tuple(ax.get_xlim())==self.env['X_LIMIT'] and tuple(ax.get_ylim())==self.env['Y_LIMIT']
        result=self.save(fig,bbox,pos,'scatter',path)
        plt.close(fig)
        return result,pos,reference

FONT=font_manager.findfont('DejaVu Sans')
FT=ImageFont.truetype(FONT,28);FL=ImageFont.truetype(FONT,23)
FC=ImageFont.truetype(FONT,32);FF=ImageFont.truetype(FONT,19)


def flagship(mp,sc,g,layer,path):
    canvas=Image.new('RGB',SIZES['flagship'],'white');right=912+24
    canvas.paste(mp,(0,0));canvas.paste(sc,(right,0))
    draw=ImageDraw.Draw(canvas);counts=g[layer+'_display_class'].value_counts()
    draw.text((right+30,675),f'{layer.upper()} display counts | {g.month.iloc[0]:%Y-%m}',font=FT,fill='#222222')
    draw.text((right+30,714),f'Active: {int(g.E.eq(0).sum()):,} | RAW discoveries: {int(g.raw_sig.sum()):,} | BH discoveries: {int(g.fdr_sig.sum()):,}',font=FF,fill='#222222')
    for i,c in enumerate(CLASSES):
        x=right+30+(i%4)*350;y=740+(i//4)*120
        draw.rectangle((x,y+5,x+22,y+27),fill=COLORS[c])
        draw.text((x+36,y),c,font=FL,fill='#222222')
        draw.text((x+36,y+38),f'{counts.get(c,0):,}',font=FC,fill='#222222')
    draw.text((right+30,1000),'NS: active interior only | Islands retain context colors | Fixed geography | 1,793 districts',font=FF,fill='#333333')
    canvas.save(path)

