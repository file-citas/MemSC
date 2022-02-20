from multiprocessing.pool import ThreadPool
from multiprocessing import Manager
import multiprocessing as mp
import random
import time
import shlex
import subprocess
import sys
import os
import shutil
import argparse
import json
from timeit import default_timer as timer
from threading import Timer

BASE_DIR = "/home/feli/project/mempat_new/mp_model"
MAKEREPLAY_PATH = os.path.join(BASE_DIR, "makereplay.py")
STOP_TOKEN = "STOP!!!"

def print_guest(guest_p, log_fd):
    print("Starting guest printer")
    while True:
        line = guest_p.stdout.readline()
        if line == '':
            time.sleep(0.5)
            continue
        if line == b'\r\n':
            time.sleep(0.5)
            continue
        log_fd.write(line.decode('utf-8'))
        log_fd.flush()
    print("Terminating guest printer")

def log_write(status_q, logfd):
    print("START Log Writer")
    try:
        while True:
            time.sleep(0.5)
            x = status_q.get()[0]
            if x == STOP_TOKEN:
                print("STOP_TOKEN")
                break
            print("LOG: %s" % x)
            logfd.write(x)
            logfd.write("\n")
            logfd.flush()
    except Exception as e:
       print(e)
       print("ERROR Log Writer %s" % str(e))
    print("STOP Log Writer")



def work_single(conf):
    key = conf['recid']
    status_q = conf['status']
    cmd = ["/usr/bin/python3", MAKEREPLAY_PATH,
          "-qtimeout", "%d" % conf['time'],
          "-i", conf['target'], "-c", conf['conf']]
    if conf["trace_all"]:
       cmd.append("--trace_all")
    if conf["exclude_irq"]:
       cmd.append("--exclude_irq")
    #status_q.put(["CMD %d: %s\n" % (0, " ".join(cmd))])
    start_time = time.time()
    try:
        proc = subprocess.Popen(shlex.split(" ".join(cmd)), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ppid = proc.pid
    except Exception as e:
        status_q.put(["CMD %d: ERROR 0 %s (%s)" % (0, e, " ".join(cmd))])
        return

    #try:
    #    os.set_blocking(proc.stdout.fileno(), False)
    #except Exception as e:
    #    status_q.put(["CMD %d: ERROR 1 %s (%s)" % (0, e, " ".join(cmd))])
    #    return

    #log_fd = open("%s_replay.log" % key, "w")
    #try:
    #    print_p = mp.Process(target=print_guest, args=(proc, log_fd))
    #    print_p.start()
    #except Exception as e:
    #    status_q.put(["CMD %d: ERROR 2 %s (%s)\n" % (0, e, " ".join(cmd))])
    #    log_fd.close()
    #    return

    status_q.put(["CMD %d: %s" % (ppid, " ".join(cmd))])
    timeout_timer = Timer(conf['time']+60, proc.kill)
    try:
      timeout_timer.start()
      proc.wait()
      try:
          proc.kill()
      except:
          pass
    finally:
        timeout_timer.cancel()

    end_time = time.time()
    status_q.put(["CMD %d: DONE %d sec %s" % (ppid, end_time-start_time, " ".join(cmd))])
    #status_q.put(["CMD DONE: %s\n" % " ".join(cmd)])
    #print_p.kill()
    #log_fd.close()


def run_single(args):
   tp = ThreadPool(args.workers)
   manager = Manager()
   status_q = manager.Queue()
   log_fd = open("%s_replay.log" % args.recid, "w")
   log_p = mp.Process(target=log_write, args=(status_q, log_fd))
   log_p.start()
   all_targets = []
   for targ_fn in args.targets:
      with open(targ_fn, "r") as fd:
         for l in fd.readlines():
            l = l.rstrip()
            all_targets.append(l)
   random.shuffle(all_targets)
   #for target in args.targets:
   cntr = 0
   for tg in all_targets:
     conf = {
        'target': tg,
        'time': args.timeout,
        'recid': args.recid + "_%d" % cntr,
        'conf': args.conf,
        'trace_all': args.trace_all,
        'exclude_irq': args.exclude_irq,
        'status': status_q,
     }
     cntr += 1
     tp.apply_async(work_single, (conf,))
     time.sleep(1)

   tp.close()
   tp.join()
   print("TERMINATING")
   status_q.put([STOP_TOKEN])
   log_p.join()
   log_p.kill()
   log_fd.close()


def main(args):
    run_single(args)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-targets", nargs="+",
      help="Target modules")
   parser.add_argument("-workers", type=int,
      help="Number of workers", default=16)
   parser.add_argument("-recid", type=str,
      help="recid", required=True)
   parser.add_argument("-conf", type=str,
      help="conf", required=True)
   parser.add_argument("-timeout", type=int,
      help="Timeout", default=10)
   parser.add_argument(
        '--trace_all',
        action='store_true',
        required=False,
        help='Trace all memory accesses.'
   )
   parser.add_argument(
        '--exclude_irq',
        action='store_true',
        required=False,
        help='Trace all memory accesses.'
   )
   args = parser.parse_args()
   main(args)
