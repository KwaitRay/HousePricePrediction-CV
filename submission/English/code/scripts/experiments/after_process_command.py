"""Run a recorded experiment command after a specific live Windows process completes."""
import argparse,ctypes,json,subprocess,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--command',type=Path,required=True);args=p.parse_args()
    before=json.loads(args.status.read_text());assert (before.get('pid') or before.get('controller_pid'))==args.pid
    command=json.loads(args.command.read_text());assert isinstance(command,list) and all(isinstance(s,str) for s in command)
    record=args.status.parent/(args.command.stem+'_queue.json')
    def write(state,**kw):record.write_text(json.dumps({'status':state,'controller_pid':os.getpid(),'reference_pid':args.pid,'command':command,**kw},indent=2),encoding='utf-8')
    write('waiting');kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p];handle=kernel.OpenProcess(0x100000,False,args.pid)
    if handle:
        try:
            print(f'Waiting on live process PID {args.pid}',flush=True)
            while True:
                result=kernel.WaitForSingleObject(handle,1000)
                if result==0:break
                if result!=258:raise OSError(ctypes.get_last_error(),'Wait failed')
        finally:kernel.CloseHandle(handle)
    state=json.loads(args.status.read_text())
    if state['status']!='completed':write('not_started',reason='Predecessor not completed',predecessor=state);raise RuntimeError('No automatic restart')
    write('running');print('Predecessor completed; starting recorded experiment',flush=True)
    result=subprocess.run([sys.executable,'-X','utf8',*command],check=False)
    write('completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
