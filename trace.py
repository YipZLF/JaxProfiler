import jax
import jaxlib

class Event(object):
    def __init__(self, name, start_ts, duration, type='X', pid = 0, tid = 0, **kwargs):
        self.ph = type # 'X','B', 'E'
        self.pid = pid
        self.tid = tid
        self.start_ts = start_ts
        self.dur = duration
        self.name = name
        self.args = kwargs
    
    def __str__(self):
        return str(self.__dict__)

class CustomTraceAnnotation(object):
    """Context manager that generates a trace event in the profiler.
    Still buggy in jax.jit
    """
    traceme: jaxlib.xla_client.profiler.TraceMe
    idx: int = 0
    def __init__(self):
        pass #self.traceme = jaxlib.xla_client.profiler.TraceMe(name)
    def __new__(cls):
        cls.idx += 1
        return super(CustomTraceAnnotation, cls).__new__(cls)

    def __enter__(self):
        def _enter_cb(idx=0):
            self.traceme = jaxlib.xla_client.profiler.TraceMe("Step {}".format(idx))

            print("Enter call on {} {}".format(self.idx,self.traceme))
        jax.debug.callback(_enter_cb, self.idx)
        return self.traceme
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        def _exit_cb():
            self.idx += 1
            print("Exit call on {} {}".format(self.idx, self.traceme))
            jaxlib.xla_client.profiler.TraceMe.__exit__(self.traceme, exc_type, exc_val, exc_tb)
        jax.debug.callback(_exit_cb)
        self.idx+=1 
        return None

