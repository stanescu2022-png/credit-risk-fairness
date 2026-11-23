"""Install network guard and two-thread caps before numerical imports."""
import os
import sys

THREAD_VARIABLES=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
    'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
TELEMETRY_OPTOUTS={'MSBUILD_TELEMETRY_OPTOUT':'1','DOTNET_CLI_TELEMETRY_OPTOUT':'1',
    'POWERSHELL_TELEMETRY_OPTOUT':'1','VSCMD_SKIP_SENDTELEMETRY':'1'}


def main():
    from research.model_selection_v2.confirmation_v2.offline import deny_socket
    sys.addaudithook(deny_socket)
    sys._credit_risk_v2_offline=True
    sys._credit_risk_jobs_offline=True
    for name,value in TELEMETRY_OPTOUTS.items():os.environ[name]=value
    for name in THREAD_VARIABLES:os.environ[name]='2'
    import site
    site.main()
    import runpy
    if len(sys.argv)<2 or not sys.argv[1].startswith('research.model_selection_v2.confirmation_v2_jobs.'):
        raise ValueError('Supply a module in the labelled two-job package')
    module=sys.argv[1];sys.argv=sys.argv[1:]
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
