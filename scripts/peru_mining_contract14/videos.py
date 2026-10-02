"""Six compatible final videos, complete decode and frame/source verification."""
import sys, json, re, struct, subprocess, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import imageio_ffmpeg
from PIL import Image
from art import OUT, SIZES, sha, write_json
AMEND=OUT


def run(cmd,path):
    r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    path.write_text(r.stdout+r.stderr,encoding='utf-8')
    assert r.returncode==0,str(path)
    return r.stdout+r.stderr


def encode():
    review=json.loads((AMEND/'static_visual_review.json').read_text())
    assert review['status']=='PASS'
    assert all(sha(AMEND/p)==h for p,h in review['reviewed_files'].items())
    render=json.loads((AMEND/'render_validation.json').read_text())
    assert render['months_rendered']==288
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    (AMEND/'video_logs').mkdir(exist_ok=True);(AMEND/'decoded_keyframes').mkdir(exist_ok=True)
    results={};rows=[]
    for layer in ['descriptive','raw','bh']:
        for kind in SIZES:
            name=f'{layer}_{kind}';target=OUT/'videos'/f'{name}.mp4'
            files=sorted((AMEND/'frames'/name).glob('*.png'));assert len(files)==288
            started=time.perf_counter()
            cmd=[ff,'-hide_banner','-nostdin','-n','-framerate','8','-start_number','0','-i',str(AMEND/f'frames/{name}/%04d.png'),
                 '-frames:v','288','-an','-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-profile:v','high']
            if kind!='flagship':cmd+=['-level:v','4.1']
            cmd+=['-r','8','-fps_mode','cfr','-movflags','+faststart',str(target)]
            if not target.exists():run(cmd,AMEND/f'video_logs/{name}_encode.log')
            strict=run([ff,'-hide_banner','-nostdin','-v','error','-xerror','-err_detect','explode','-i',str(target),
                        '-map','0:v:0','-progress','pipe:1','-f','null','-'],AMEND/f'video_logs/{name}_decode.log')
            assert re.findall(r'^frame=(\d+)$',strict,re.M)[-1]=='288'
            header=run([ff,'-hide_banner','-i',str(target),'-map','0:v:0','-c:v','copy','-bsf:v','trace_headers','-frames:v','1','-f','null','-'],AMEND/f'video_logs/{name}_headers.log')
            assert 'Video: h264 (High)' in header and set(re.findall(r'profile_idc\s+\S+\s+=\s+(\d+)',header))=={'100'}
            reader=imageio_ffmpeg.read_frames(str(target),pix_fmt='rgb24');meta=next(reader)
            assert tuple(meta['size'])==SIZES[kind] and meta['fps']==8
            assert meta['pix_fmt'].split('(')[0]=='yuv420p' and abs(meta['duration']-36.0)<.01
            for i in range(288):
                data=next(reader);w,h=SIZES[kind];actual=np.frombuffer(data,dtype=np.uint8).reshape(h,w,3)
                ref=np.asarray(Image.open(files[i]).convert('RGB'))
                assert actual.shape==ref.shape
                mae=float(np.abs(actual.astype(np.int16)-ref.astype(np.int16)).mean());assert mae<8
                rows.append({'video':name,'frame':i,'month':str(pd.Timestamp('2001-01-01')+pd.offsets.MonthBegin(i)),
                             'source_png_sha256':sha(files[i]),'decoded_rgb_sha256':hashlib.sha256(data).hexdigest(),'rgb_MAE':mae})
                if i in (0,144,287):Image.fromarray(actual).save(AMEND/f'decoded_keyframes/{name}_{i:04d}.png')
            assert next(reader,None) is None;reader.close()
            atoms=[]
            with target.open('rb') as f:
                while f.tell()<target.stat().st_size:
                    start=f.tell();size,atom=struct.unpack('>I4s',f.read(8))
                    if size==1:size=struct.unpack('>Q',f.read(8))[0]
                    if size==0:size=target.stat().st_size-start
                    assert size>=8
                    atoms.append(atom.decode('ascii'));f.seek(start+size)
            assert atoms.index('moov')<atoms.index('mdat')
            results[name]={'path':str(target.relative_to(OUT)),'sha256':sha(target),'bytes':target.stat().st_size,
                           'size':list(SIZES[kind]),'frames':288,'fps':8,'duration_seconds':36.0,
                           'codec':'H.264 libx264','profile':'High','pixel_format':'yuv420p','faststart':True,
                           'every_frame_decoded_and_compared':True,'C11_size_and_cadence_match':True,
                           'command':cmd,'elapsed_seconds':time.perf_counter()-started}
            print('PASS',name,flush=True)
    pd.DataFrame(rows).to_csv(AMEND/'decoded_frames.csv',index=False)
    write_json(AMEND/'final_video_manifest.json',{'status':'PASS','videos':results,'decoded_frames':len(rows),
                'pixel_note':'Expected small lossy H.264/YUV420 differences; all decoded frames compared with exact PNG sources.',
                'streams':'Descriptive, RAW and BH each preserve approved map/scatter/flagship architecture.'})



if __name__=='__main__':encode()
