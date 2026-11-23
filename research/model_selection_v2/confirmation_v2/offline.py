"""Run an authorised local module with Python socket access disabled.

Invoke with .venv/Scripts/python.exe -S -m ...offline MODULE [arguments].
The audit hook is installed before site/package initialisation. It rejects all
Python socket operations, including name resolution, rather than permitting the
unverified telemetry connection rejected by automatic review.
"""
import sys


def deny_socket(event,args):
    if event.startswith('socket.'):
        raise PermissionError('Offline research runner rejects Python socket operations: '+event)


def main():
    sys.addaudithook(deny_socket)
    sys._credit_risk_v2_offline=True
    import site
    site.main()
    import runpy
    if len(sys.argv)<2:
        raise SystemExit('Supply an explicitly authorised local module')
    module=sys.argv[1]
    allowed=('research.model_selection_v2.confirmation_v2.',)
    if not module.startswith(allowed):
        raise ValueError('Only this labelled local package may be executed')
    sys.argv=sys.argv[1:]
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':
    main()
