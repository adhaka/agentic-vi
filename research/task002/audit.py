"""Reconcile each experiment against the frozen design and independently reconstruct endpoints."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from study import design
from benchmark import load_instance


def read_array(root,ref):
    data = gzip.decompress((root/ref['path']).read_bytes())
    assert hashlib.sha256(data).hexdigest()==ref['sha256_npy']
    arr = np.load(io.BytesIO(data),allow_pickle=False)
    assert list(arr.shape)==ref['shape'] and str(arr.dtype)==ref['dtype']
    return arr


def audit(root, smoke=False):
    cells = json.loads((root/'index.json').read_text())['cells']
    expected = list(design(smoke))
    assert len(cells)==len(expected)
    for cell,(arm,kind,d,tid,c,B) in zip(cells,expected,strict=True):
        folder=root/cell['name']
        p=json.loads((folder/'profile.json').read_text())
        e=json.loads((folder/'evaluator.json').read_text())
        assert (p['target_instance_id'],p['dimension'],p['configuration']['steps'])==(tid,d,c.steps)
        mean,cov=load_instance(kind,d)
        assert hashlib.sha256(cov.tobytes()).hexdigest()==e['covariance_sha256']
        prec=np.linalg.inv(cov)
        budget=p['budget']
        steps=5 if arm=='short' else c.steps
        assert len(p['runs'])==4
        assert budget['B']==B and budget['used']+budget['remaining']==B
        for key,value in [('f',4*(steps*c.batch+c.G+c.S+c.J*c.R)),
                          ('g',4*(steps*c.batch+c.G+c.J*c.R)),('h',4*c.J*c.R)]:
            assert budget[key]==value==sum(row[key] for row in budget['ledger'].values())
        assert budget['used']==budget['f']+budget['g']+budget['h']==cell['used']
        assert budget['failed']==0 and budget['denied']==(4 if arm=='short' else 0)
        for row in budget['ledger'].values():
            assert row['attempts']==row['completed']+row['denied']+row['failed']
        for run,truth in zip(p['runs'],e['runs'],strict=True):
            assert run['steps']==steps
            assert run['status']==('insufficient_budget' if arm=='short' else 'ok')
            tr=run['trajectory']
            rows=np.concatenate([read_array(root,ref) for ref in tr['chunks']])
            assert rows.shape==(steps,4*d+1) and np.isfinite(rows).all()
            mu=np.asarray(run['end']['mu'])
            la=np.asarray(run['end']['effective_lambda'])
            var=np.exp(2*la)
            assert_allclose(rows[-1,:d],mu)
            assert_allclose(rows[-1,d:2*d],run['end']['raw_lambda'])
            # Integrate normalized log-density ratio at symmetric cubature points.
            shifts=np.diag(np.sqrt(d*var))
            x=np.concatenate((mu+shifts,mu-shifts))-mean
            kl=.5*(np.mean(np.sum((x@prec)*x,axis=1))-d+np.linalg.slogdet(cov)[1]-sum(np.log(var)))
            assert_allclose(kl,truth['reverse_kl'],atol=1e-10)
            assert_allclose(kl,truth['floor']+truth['gap'],atol=1e-10)
            imp=run['importance']['value']
            lw=read_array(root,imp['smoothed_log_weights'])
            assert_allclose(np.exp(lw).sum(),1)
            assert_allclose(1/np.exp(2*lw).sum(),imp['psis_ess'])
            curv=run['curvature']['value']
            assert abs(curv['geometry'])<1e-12
            assert np.isfinite(curv['dependence'])
            if kind=='independent':
                assert curv['dependence']<1e-12
            for key in ('directional','estimated_diagonals'):
                read_array(root,curv[key])
        assert len(p['profile']['optimization']['agreement']['pairs'])==6
    return sum(c['used'] for c in cells)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    print('Audited TEU',audit(args.root,args.smoke))
