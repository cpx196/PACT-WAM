"""Pure 4-step scheduling and offline calibration for max32 evaluation."""
import math
import random

def weighted_tau(trajectories, quantile=.75):
    if len(trajectories) < 2 or any(len(t) < 16 for t in trajectories) or sum(map(len, trajectories)) < 64:
        raise ValueError('Insufficient calibration: need >=2 independent trajectories, >=16 windows each, >=64 total')
    rows=[]
    for errors in trajectories:
        for error in errors:
            if not math.isfinite(error):raise ValueError('Nonfinite calibration error')
            rows.append((error, 1 / len(trajectories) / len(errors)))
    cumulative=0.
    for value, weight in sorted(rows):
        cumulative+=weight
        if cumulative+1e-12>=quantile:return value
    raise AssertionError('No quantile')

class Schedule:
    def __init__(self, arm, tau=None, probability=None, random_seed=0):
        if arm not in ('fixed32','adaptive32','random32'):raise ValueError(arm)
        if arm=='adaptive32' and (tau is None or not math.isfinite(tau)):raise ValueError('Frozen tau required')
        if arm=='random32' and (probability is None or not 0<=probability<=1):raise ValueError('Frozen probability required')
        self.arm,self.tau,self.probability=arm,tau,probability
        self.rng=random.Random(random_seed)
    def decision(self, age, error=None):
        if age not in range(4,33,4):raise ValueError('Invalid plan age')
        if error is not None and not math.isfinite(error):raise ValueError('Nonfinite error')
        if age==32:return 'max32'
        if self.arm=='adaptive32':
            if error is None:raise ValueError('Adaptive decision requires actual feedback')
            return 'error_replan' if error>self.tau else 'continue'
        if self.arm=='random32':return 'random_replan' if self.rng.random()<self.probability else 'continue'
        return 'continue'

def replay_calls(errors,tau):
    """Calibration proxy: reset max32 deadline on each error trigger, without counterfactual environment simulation."""
    calls=1;age=0
    for i,error in enumerate(errors):
        age+=4
        if i<len(errors)-1 and (age==32 or error>tau):calls+=1;age=0
    return calls

def random_expected_calls(blocks,probability):
    if blocks<=0:raise ValueError('No blocks')
    states={0:1.};calls=1.
    for _ in range(blocks-1):
        nxt={}
        for age,weight in states.items():
            hit=1. if age==7 else probability
            calls+=weight*hit
            nxt[0]=nxt.get(0,0)+weight*hit
            if hit<1:nxt[age+1]=nxt.get(age+1,0)+weight*(1-hit)
        states=nxt
    return calls

def fit_random_probability(trajectories,tau):
    lengths=list(map(len,trajectories))
    target=sum(replay_calls(t,tau)/len(t) for t in trajectories)/len(trajectories)
    def rate(p):return sum(random_expected_calls(n,p)/n for n in lengths)/len(lengths)
    low,high=0.,1.
    for _ in range(64):
        middle=(low+high)/2
        if rate(middle)<target:low=middle
        else:high=middle
    p=(low+high)/2
    return dict(probability=p,adaptive_proxy_calls_per100_steps=25*target,random_expected_calls_per100_steps=25*rate(p),absolute_rate_gap=25*abs(rate(p)-target),method='Replay offline errors with reset max32 deadline; fit Bernoulli probability by exact finite-horizon expectation. Calibration proxy only; online budget equality is not guaranteed.')

def selftest():
    assert weighted_tau([list(range(16)),list(range(100,148))])==123
    assert Schedule('adaptive32',tau=1).decision(4,1)=='continue'
    assert Schedule('adaptive32',tau=1).decision(28,1.01)=='error_replan'
    assert Schedule('adaptive32',tau=1).decision(32,1.01)=='max32'
    assert Schedule('random32',probability=1).decision(4)=='random_replan'
    assert Schedule('random32',probability=0).decision(28)=='continue'
    assert Schedule('random32',probability=1).decision(32)=='max32'
    assert random_expected_calls(175,0)==22
    assert random_expected_calls(175,1)==175
    assert replay_calls([0]*175,1)==22
    assert replay_calls([2]*175,1)==175
    for errors in ([[0]*175,[0]*160],[[2]*175,[2]*160],[[i%7 for i in range(175)],[i%5 for i in range(160)]]):
        result=fit_random_probability(errors,1)
        assert result['absolute_rate_gap']<1e-10
    a=Schedule('random32',probability=.4,random_seed=7);b=Schedule('random32',probability=.4,random_seed=7)
    assert [a.decision(4) for _ in range(100)]==[b.decision(4) for _ in range(100)]
    try:Schedule('adaptive32',tau=1).decision(4,float('nan'))
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    return dict(passed=True,checks=['strict_threshold','4step_grid','max32_deadline','random_rng_independent','exact_budget_expectation','equal_trajectory_weights','nonfinite_stop'])
if __name__=='__main__':print(selftest())
