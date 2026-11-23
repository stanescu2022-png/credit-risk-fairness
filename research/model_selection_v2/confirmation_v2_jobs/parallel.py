"""Bounded independent subprocesses, process-tree RSS sampling and job budgets."""
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from .artifacts import write_json

DEFAULT_PROCESSES=3
THREADS_PER_PROCESS=2


def budget_decision(batch_seconds,processes=DEFAULT_PROCESSES,repeats=50):
    if batch_seconds<0 or processes<1 or repeats<1:raise ValueError('Invalid projection input')
    batches=math.ceil(repeats/processes)
    hours=batch_seconds*batches/3600
    limit=12 if hours<=12 else 24 if hours<=24 else None
    return dict(measured_batch_seconds=float(batch_seconds),batch_processes=processes,repeats=repeats,
        projected_batches=batches,projected_hours=hours,limit_hours=limit,
        limit_extended_before_continuing=limit==24,stop_for_new_version=limit is None,
        projection='ceil(50/processes) full batches; conservative final partial batch')


def memory_bytes():
    own=psutil.Process(os.getpid())
    processes=[own]+own.children(recursive=True)
    total=0
    for process in processes:
        try:total+=process.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return total


def terminate_owned(active):
    for process,log,repeat in active:
        if process.poll() is not None:continue
        try:
            root=psutil.Process(process.pid)
            owned=root.children(recursive=True)+[root]
            for child in reversed(owned):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            _,alive=psutil.wait_procs(owned,timeout=3)
            for child in alive:
                try:child.kill()
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass


def run_batch(specifications,folder,*,processes=DEFAULT_PROCESSES,limit_hours=12,progress=False,
              worker_module='research.model_selection_v2.confirmation_v2_jobs.jobs'):
    """Workers use fresh -S interpreters; all numerical libraries capped at 2."""
    if not getattr(sys,'_credit_risk_jobs_offline',False):raise RuntimeError('Use the guarded launcher')
    if not 1<=processes<=DEFAULT_PROCESSES:raise ValueError('Only 1..3 independent worker slots authorised')
    folder=Path(folder);folder.mkdir(exist_ok=False)
    pending=list(enumerate(specifications));active=[];completed=[]
    peak=memory_bytes();started=time.perf_counter();low_memory_since=None
    try:
        while pending or active:
            while pending and len(active)<processes:
                index,spec=pending.pop(0)
                specpath=folder/('spec_'+str(index)+'.json');write_json(specpath,spec)
                log=(folder/('worker_'+str(index)+'.log')).open('xb')
                environment=os.environ.copy()
                from .offline import TELEMETRY_OPTOUTS
                environment.update(TELEMETRY_OPTOUTS)
                for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS',
                             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):environment[name]='2'
                command=[sys.executable,'-S','-m','research.model_selection_v2.confirmation_v2_jobs.offline',
                    worker_module,str(specpath.resolve())]
                process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[3],env=environment,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((process,log,index))
            peak=max(peak,memory_bytes())
            now=time.perf_counter()
            if now-started>limit_hours*3600:raise TimeoutError('Job wall-clock limit reached')
            if psutil.virtual_memory().available<128*2**20:
                low_memory_since=now if low_memory_since is None else low_memory_since
                if now-low_memory_since>=5:raise MemoryError('Less than 128 MiB available for five seconds; stopping owned workers')
            else:low_memory_since=None
            remaining=[]
            for process,log,index in active:
                code=process.poll()
                if code is None:remaining.append((process,log,index));continue
                log.close()
                if code!=0:raise RuntimeError('Worker '+str(index)+' failed; inspect its private log')
                completed.append(index)
                if progress:print({'completed_workers':len(completed),'elapsed_seconds':time.perf_counter()-started},flush=True)
            active=remaining
            if pending or active:time.sleep(.1)
    except BaseException:
        terminate_owned(active)
        for process,log,index in active:
            if not log.closed:log.close()
        raise
    result=dict(wall_seconds=time.perf_counter()-started,peak_aggregate_RSS_bytes=peak,
        peak_aggregate_RSS_GiB=peak/2**30,process_slots=processes,threads_per_numerical_library=2,
        completed_workers=len(completed),memory_sampling_seconds=.1,
        memory_definition='Maximum sampled sum of RSS/working sets for controller and its process tree; shared pages may be counted more than once')
    write_json(folder/'timing_memory.json',result)
    return result
