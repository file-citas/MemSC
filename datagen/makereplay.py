#!/usr/bin/python

import sys
import os
import subprocess
import shlex
import pickle
import argparse
import configparser
import time
from threading import Timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay Panda Record')
    parser.add_argument(
        '-i', '--record_id',
        type=str,
        required=True,
        help='Panda Record ID.'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='configuration file (Default=./replay.config).',
        default='replay.config'
    )
    parser.add_argument(
        '-f', '--filter',
        type=str,
        required=False,
        help='Dump will only contain these programs.'
    )
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
        help='Exclude IRQ memory accesses.'
    )
    parser.add_argument("-qtimeout", type=int,
       help="qemu timeout", default=900)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    rec_dir = os.path.abspath(config['QEMU']['rec_dir'])
    rec_dir_local = os.path.abspath(config['QEMU']['rec_dir_local'])

    dump_dir = os.path.abspath(config['QEMU']['dump_dir'])
    dump_dir_local = os.path.abspath(config['QEMU']['dump_dir_local'])
    dump_prefix = os.path.join(dump_dir, "%s_ta_%d_ei_%d" % (args.record_id, args.trace_all, args.exclude_irq))

    kernel_conf_file = os.path.abspath(config['QEMU']['osi_kconf_file'])
    kernel_conf_group = config['QEMU']['osi_kconf_group']

    #info_path = os.path.join(rec_dir, args.record_id + ".info")
    info_path = os.path.join(rec_dir_local, args.record_id + ".info")
    info = pickle.load(open(info_path, 'rb'))


    print("=" * 80)


    # Build command to start the replay
    #qemu = os.path.abspath(config['QEMU']['qemu_bin'])
    qemu = config['QEMU']['qemu_bin']
    opts = "-m 512 -os linux-64-bl" #+ " -smp 1"
    osi = "-panda osi -panda osi_linux:"
    osi = osi + "kconf_file=%s," %kernel_conf_file
    osi = osi + "kconf_group=%s,pcpuoff={pcpuoff},kstext={kstext},pageoffset={pageoffset}" %kernel_conf_group
    memtrace = "-panda syscalls2:load-info=1 -panda memaccess-dump_annotated3:"
    memtrace_args = []
    memtrace_args.append("kstext=%s" % info["kstext"])
    if args.filter is not None:
        memtrace_args.append(",target=" + args.filter)
    memtrace_args.append(",annot_a=%s_ma.annotations" %dump_prefix)
    memtrace_args.append(",annot_r=%s_pr.annotations" %dump_prefix)
    memtrace_args.append(",annot_w=%s_pw.annotations" %dump_prefix)
    memtrace_args.append(",ma=%s_ma.dump" %dump_prefix)
    memtrace_args.append(",pr=%s_pr.dump" %dump_prefix)
    memtrace_args.append(",pw=%s_pw.dump" %dump_prefix)
    if args.trace_all:
        memtrace_args.append(",trace_all=1")
    else:
        memtrace_args.append(",trace_all=0")
    if args.exclude_irq:
        memtrace_args.append(",exirq=1")
    else:
        memtrace_args.append(",exirq=0")

    replay = "-replay " + os.path.join(rec_dir, args.record_id)

    cmd = (qemu + " " + opts + " " + osi + " " +
           memtrace + ",".join(memtrace_args) + " " + replay).format(**info)

    print(" [+] Running Replay")
    print(" [+] PCPU_OFFSET: {pcpuoff:x}, CTASK: {curtaskptr:x}, KSTEXT: {kstext:x}, PAGEOFF: {pageoffset:x}".format(**info))
    print(" [ ] %s" % cmd)
    start_time = time.time()
    DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(shlex.split(cmd),
                         stdout=subprocess.PIPE,
                         stderr=DEVNULL,
                         shell=False)
    timeout_timer = Timer(args.qtimeout, p.kill)
    try:
        timeout_timer.start()
        while p.poll() is None:
            l = p.stdout.readline()
            print(l)
        p.wait()
    finally:
        timeout_timer.cancel()
    try:
        p.kill()
    except:
        pass
    end_time = time.time()
    info["replay_time"] = end_time - start_time
    pickle.dump(info, open(info_path, 'wb'))
    print(" [?] check %s_*.dump and %s.annotations" % (dump_prefix, dump_prefix))
