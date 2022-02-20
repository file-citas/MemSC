#!/usr/bin/python

import sys
import os
import subprocess
import shlex
import pickle
import argparse
import configparser

PLOG_READER = "/home/feli/project/mempat_new/panda/panda/scripts/plog_reader.py"

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
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    rec_dir = os.path.abspath(config['QEMU']['rec_dir'])
    rec_dir_local = os.path.abspath(config['QEMU']['rec_dir_local'])

    kernel_conf_file = os.path.abspath(config['QEMU']['osi_kconf_file'])
    kernel_conf_group = config['QEMU']['osi_kconf_group']

    #info_path = os.path.join(rec_dir, args.record_id + ".info")
    info_path = os.path.join(rec_dir_local, args.record_id + ".info")
    info = pickle.load(open(info_path, 'rb'))


    print("=" * 80)


    # Build command to start the replay
    #qemu = os.path.abspath(config['QEMU']['qemu_bin'])
    syscalls_log = os.path.join(config['QEMU']['syscalls_dir'], "%s.syscalls.bin" % args.record_id)
    qemu = config['QEMU']['qemu_bin']
    opts = "-m 512 -os linux-64-bl" #+ " -smp 1"
    osi = "-panda osi:disable-autoload=1 -panda osi_linux:"
    osi = osi + "kconf_file=%s," %kernel_conf_file
    osi = osi + "kconf_group=%s,pcpuoff={pcpuoff},kstext={kstext},pageoffset={pageoffset}" %kernel_conf_group
    syscalls = "-panda syscalls2:load-info=1 -panda syscalls_logger:verbose=1,json:/home/feli/project/mempat_new/vmlinuxdbg.json"
    pandalog = "-pandalog %s" % syscalls_log
    replay = "-replay " + os.path.join(rec_dir, args.record_id)

    cmd = (qemu + " " + opts + " " + osi + " " +
           syscalls  + " " + pandalog + " " + replay).format(**info)

    print(" [+] Running Replay")
    print(" [+] Logging syscalls to %s" % syscalls_log)
    print(" [+] PCPU_OFFSET: {pcpuoff:x}, CTASK: {curtaskptr:x}, KSTEXT: {kstext:x}, PAGEOFF: {pageoffset:x}".format(**info))
    print(" [ ] %s" % cmd)
    DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(shlex.split(cmd),
                         stdout=subprocess.PIPE,
                         stderr=DEVNULL,
                         shell=False)
    while p.poll() is None:
        l = p.stdout.readline()
        print(l)
    p.wait()

    syscalls_parsed_log = os.path.join(config['QEMU']['syscalls_dir'], "%s.syscalls.json" % args.record_id)
    print(" [+] Logging parsed syscalls to %s" % syscalls_parsed_log)
    cmd_sc = [PLOG_READER, syscalls_log]
    p_sc = subprocess.Popen(cmd_sc,
                         stdout=subprocess.PIPE,
                         stderr=DEVNULL,
                         shell=False)
    sc_parsed = ""
    while p_sc.poll() is None:
        l = p_sc.stdout.readline()
        sc_parsed += l.decode('utf-8')

    p_sc.wait()

    with open(syscalls_parsed_log, "w") as fd_sc:
       fd_sc.write(sc_parsed)
    print(" [?] check %s" % (syscalls_parsed_log))
