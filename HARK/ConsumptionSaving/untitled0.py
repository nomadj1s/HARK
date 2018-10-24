
from multiprocessing import Pool
import time
 
def unwrap_self_f(arg, **kwarg):
    return C.f(*arg, **kwarg)
 
class C:
    def f(self, name1, name2):
        print 'hello %s,'%name1
        time.sleep(5)
        print 'nice to meet you. I am %s'%name2
     
    def run(self):
        pool = Pool(processes=2)
        names1 = ('frank', 'justin', 'osi', 'thomas')
        names2 = ('joy','nikki','arthur','doris')
        [pool.apply(unwrap_self_f, args=(i,)) for i in zip([self]*len(names1), names1, names2)]
 
c = C()
c.run()