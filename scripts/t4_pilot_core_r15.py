"""Pure scheduling and trajectory-weighted calibration; no model imports."""
import math
import random

class Schedule:
    def __init__(self, arm, tau=None, seed=470048):
        if arm not in ('calibration','fixed8','adaptive48','random15','fixed4'):
            raise ValueError(arm)
        if arm == 'adaptive48' and (tau is None or not math.isfinite(tau)):
            raise ValueError('adaptive48 requires frozen finite tau')
        self.arm, self.tau, self.rng = arm, tau, random.Random(seed)
    def early(self, error):
        if not math.isfinite(error):
            raise ValueError('Nonfinite monitor error')
        if self.arm == 'adaptive48': return error > self.tau
        if self.arm == 'random15': return self.rng.random() < .15
        return self.arm == 'fixed4'

def calibrate(trajectories, quantile=.75):
    if len(trajectories) != 2 or any(len(t)<16 for t in trajectories) or sum(map(len,trajectories))<64:
        raise ValueError('Insufficient independent calibration windows')
    weighted=[]
    for t in trajectories:
        for value in t:
            if not math.isfinite(value): raise ValueError('Nonfinite calibration error')
            weighted.append((value, .5/len(t)))
    cumulative=0.
    for value,weight in sorted(weighted):
        cumulative+=weight
        if cumulative+1e-12 >= quantile: return value
    raise AssertionError('Quantile unavailable')

def selftest():
    assert calibrate([list(range(16)),list(range(100,148))])==123
    assert not Schedule('adaptive48',1).early(1)
    assert Schedule('adaptive48',1).early(1.01)
    assert not Schedule('fixed8').early(100)
    assert Schedule('fixed4').early(0)
    a,b=Schedule('random15'),Schedule('random15')
    assert [a.early(0) for _ in range(100)] == [b.early(100) for _ in range(100)]
    for bad in (float('nan'),float('inf')):
        try: Schedule('fixed8').early(bad)
        except ValueError: pass
        else: raise AssertionError('Nonfinite accepted')
    try: calibrate([[0]*16,[1]*16])
    except ValueError: pass
    else: raise AssertionError('Insufficient calibration accepted')
    return dict(passed=True,checks=['strict_tau','trajectory_equal_weights','min_samples','independent_random_rng','nonfinite_stop'])
if __name__=='__main__': print(selftest())
