"""Wait on a specific live Windows process handle, then run a registered kernel search."""
import argparse,ctypes,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--cache-key',required=True);p.add_argument('--cache-family',required=True);p.add_argument('--name',required=True)
    p.add_argument('--original-manifest',type=Path)
    args=p.parse_args();state=json.loads(args.status.read_text());assert state['pid']==args.pid
    record=args.status.parent/'kernel_queue.json'
    def write(status,**kw):record.write_text(json.dumps({'status':status,'controller_pid':os.getpid(),'reference_pid':args.pid,
        'cache_key':args.cache_key,'cache_family':args.cache_family,'name':args.name,**kw},indent=2),encoding='utf-8')
    write('waiting');kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,args.pid)
    if handle:
        try:
            print(f'Waiting on live process handle PID {args.pid}',flush=True)
            while True:
                result=kernel.WaitForSingleObject(handle,1000)
                if result==0:break
                if result!=258:raise OSError(ctypes.get_last_error(),'Process wait failed')
        finally:kernel.CloseHandle(handle)
    state=json.loads(args.status.read_text())
    if state['status']!='completed':
        write('not_started',reason='Predecessor did not complete successfully',predecessor=state)
        raise RuntimeError('Predecessor incomplete; no restart and no kernel run')
    write('running');print('Predecessor completed; starting kernel search',flush=True)
    command=[sys.executable,'-X','utf8',str(Path(__file__).with_name('frozen_kernel_search.py')),
        '--name',args.name,'--cache-family',args.cache_family,'--cache-key',args.cache_key]
    if args.original_manifest:command.extend(['--original-manifest',str(args.original_manifest)])
    result=subprocess.run(command,check=False)
    write('completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
