#!/usr/bin/python
import subprocess
import random
import socket
import paramiko
import os
import shlex
import time
import sys
import traceback
import re
import telnetlib
import pickle
import argparse
import configparser
import multiprocessing as mp
from threading import Timer

def next_free_port(port=10022, max_port=65535 ):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = ' [+] {:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
    print("")

def connect_tel_mon(host, port):
    tn = telnetlib.Telnet(host, port=port, timeout=3)
    tn.read_very_eager()
    #tn.read_all()
    return tn

def ssh_login(user, pw, host, port):
    ssh = paramiko.SSHClient()
    #ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host,
                int(port),
                password=pw,
                username=user,
                timeout=90)
    return ssh

def print_guest(guest_p):
    print("Starting guest printer")
    while True:
        time.sleep(0.01)
        line = guest_p.stdout.readline()
        if line == '':
            time.sleep(0.3)
            continue
        if line == b'':
            time.sleep(0.3)
            continue
        if line == b'\r\n':
            time.sleep(0.3)
            continue
        if line == b'\n':
            time.sleep(0.3)
            continue
        lineutf = line.decode('utf-8')
        if len(lineutf) > 0:
            sys.stdout.write("GUEST: " + lineutf)
        else:
            time.sleep(0.3)
    print("Terminating guest printer")

def exec_guest_cmds(ssh, guest_cmds):
    for cmd in guest_cmds:
        print(" [+] Executing %s" % cmd)
        _, stdout, _ = ssh.exec_command(cmd, timeout=3600)
        stdout.channel.recv_exit_status() # wait for cmd to finish before starting next


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record Guest Execution')
    parser.add_argument(
        '-i', '--record_id',
        type=str,
        required=True,
        help='ID to be assigned to this record'
    )
    parser.add_argument(
        '-c', '-config',
        type=str,
        help='configuration file (Default=./config)',
        default='replay.config'
    )
    parser.add_argument(
        '-g', '--guest_cmds',
        type=str,
        default=[],
        nargs='+',
        help='files containing guest commands to be executed'
    )
    parser.add_argument(
        '-ltp',
        type=str,
        nargs='+',
        default=[],
        help='syscall testcases to execute'
    )
    parser.add_argument(
        '--no_kaslr',
        action='store_true',
        default=False,
        help='Use Kernel Adress Space Layout Randomization.'
    )
    parser.add_argument(
        '-gport',
        type=int,
        default=-1,
        help='guest ssh port',
    )
    parser.add_argument(
        '-mport',
        type=int,
        default=-1,
        help='guest monitor port',
    )

    parser.add_argument("-qtimeout", type=int,
       help="qemu timeout", default=900)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.c)
    no_kaslr = args.no_kaslr

    if no_kaslr:
        nokaslr = "nokaslr"
    else:
        nokaslr = ""

    record_dir = config['QEMU']['rec_dir']
    record_dir_local = config['QEMU']['rec_dir_local']
    assert(os.path.isdir(record_dir_local))

    guest_cmds = []
    random.shuffle(args.guest_cmds)
    for cmd_fn in args.guest_cmds:
        if cmd_fn == "":
            continue
        with open(cmd_fn, 'r') as f:
            for cmd in f.readlines():
                if cmd != "":
                    guest_cmds.append(cmd.rstrip())

    random.shuffle(args.ltp)
    for ltp_syscall in args.ltp:
        if ltp_syscall == "":
            continue
        guest_cmds.append("/opt/ltp/runltp -f syscalls -s %s" % ltp_syscall)

    recid = args.record_id

    print("=" * 80)
    if args.gport >= 1234:
        guest_port = "%d" % args.gport
    else:
        guest_port = "%s" % next_free_port()
    if args.mport >= 1234:
        monitor_port = "%d" % args.mport
    else:
        monitor_port = "%s" % next_free_port(port=1234, max_port=10000)
    print(" [+] using port %s(ssh), %s(monitor)" % (guest_port, monitor_port))
    print(" [+] creating QCOW overlay from %s" % (config['QEMU']['rootfs_qcow']))
    rootfs_qcow = os.path.join(record_dir, "rootfs_%s.qcow2" % recid)
    img_args = [config['QEMU']['qemu_img'] +
            ' create  -F qcow2  -f qcow2 -b %s %s' % (config['QEMU']['rootfs_qcow'], rootfs_qcow)]

    DEVNULL = open(os.devnull, 'wb')
    qemu_img = subprocess.Popen(shlex.split(img_args[0]),
                            stdout=DEVNULL,
                            stderr=DEVNULL,
                            shell=False)
    qemu_img.wait()
    print(" [+] created QCOW overlay %s" % (rootfs_qcow))
    qemu_args = [config['QEMU']['qemu_bin'] +
            ' -kernel ' + config['QEMU']['bzimage'] +
            ' -append "console=ttyS0 root=/dev/sda rw ip=dhcp ' + nokaslr + '"'
            ' -hda ' + rootfs_qcow +
            ' -nographic'
            ' -m 512'
            ' -smp 1'
            ' -monitor telnet:' + config['MONITOR']['listen'] + ':' +
            monitor_port + ',server,nowait' +
            ' -net nic -net user,host=10.0.2.10,hostfwd=tcp::' + guest_port + '-:22'
            , ]

    p_sctable = re.compile(".*SCTABLE--([0-9a-fA-F]+)-.*")
    p_sctablepa = re.compile(".*SCTABLEPA--([0-9a-fA-F]+)-.*")
    p_cpuoff = re.compile(".*PCPUOFF--([0-9a-fA-F]+)-.*")
    p_kstext = re.compile(".*STEXT--([0-9a-fA-F]+)-.*")
    p_pageoffset = re.compile(".*PAGE_OFFSET_BASE--([0-9a-fA-F]+)-.*")
    p_curtaskptr = re.compile(".*CURRENT_TASK_PTR--([0-9a-fA-F]+)-.*")

    info = {
        'pcpuoff':None,
        'kstext':None,
        'pageoffset':None,
        'curtaskptr':None,
        'sctable':None,
        'sctablepa':None,
        'recid':recid,
        'cmds':list(guest_cmds),
        'cmd_fns':list(args.guest_cmds)
    }

    print(" [+] starting qemu")
    print(" [ ] %s" % "".join(qemu_args))
    qemu = subprocess.Popen(shlex.split(qemu_args[0]),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=False)
    timeout_timer = Timer(args.qtimeout, qemu.kill)
    all_ok = False
    try:
        print(" [+] Qemu pid %d" % qemu.pid)
        timeout_timer.start()
        while True:
            line = qemu.stdout.readline()
            #if line == '':
            #    break
            #if line == b'\r\n':
            #    break
            try:
                if line != '' and line != b'\r\n':
                    print(line.decode('ascii').rstrip())
            except:
                continue
            if line.decode('ascii').startswith('Debian GNU/Linux 9 syzkaller ttyS0'):
                break
            if info['sctable'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_sctable.search(l)
                if m:
                    info['sctable'] = int(m.group(1), 16)
                    print(' [+] sc table offset {sctable:#x}'.format(**info))
            if info['sctablepa'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_sctablepa.search(l)
                if m:
                    info['sctablepa'] = int(m.group(1), 16)
                    print(' [+] sc table pa offset {sctablepa:#x}'.format(**info))
            if info['pcpuoff'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_cpuoff.search(l)
                if m:
                    info['pcpuoff'] = int(m.group(1), 16)
                    print(' [+] per cpu offset {pcpuoff:#x}'.format(**info))
            if info['kstext'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_kstext.search(l)
                if m:
                    info['kstext'] = int(m.group(1), 16)
                    print(' [+] kstext {kstext:#x}'.format(**info))
            if info['pageoffset'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_pageoffset.search(l)
                if m:
                    info['pageoffset'] = int(m.group(1), 16)
                    print(' [+] pageoffset {pageoffset:#x}'.format(**info))
            if info['curtaskptr'] is None:
                l = line.rstrip().decode("ascii")
                #print(l)
                m = p_curtaskptr.search(l)
                if m:
                    info['curtaskptr'] = int(m.group(1), 16)
                    print(' [+] curtaskptr {curtaskptr:#x}'.format(**info))
            #if info['pcpuoff'] is not None and \
            #        info['kstext'] is not None and \
            #        info['curtaskptr'] is not None and \
            #        info['sctable'] is not None and \
            #        info['sctablepa'] is not None and \
            #        info['pageoffset'] is not None:
            #    break

        record_name = "{recid:s}".format(**info)
        record_dir = os.path.abspath(record_dir);
        record_path = os.path.join(record_dir, record_name)

        print(" [+] Waiting for guest to finish booting (10sec)")
        countdown(10)
        ssh = None
        print_p = None
        try:
            ssh = ssh_login(user=config['GUEST']['user'],
                            pw=config['GUEST']['pw'],
                            host=config['GUEST']['ip'],
                            port=guest_port)
            print(" [+] SSH connection established")
        except Exception as e:
            print(e)
        else:
            try:
                print(" [+] Connecting monitor (telnet)")
                tn_host, tn_port = config['MONITOR']['ip'], monitor_port
                tn = connect_tel_mon(host=tn_host, port=tn_port)
                countdown(1)
                start_time = time.time()
                print(" [+] Starting Record")
                brec_cmd = "begin_record %s\n" % record_path
                print(" [ ] %s" % brec_cmd, end='')
                tn.write(brec_cmd.encode('utf-8'))
                tn.read_very_eager()

                #os.set_blocking(qemu.stdout.fileno(), False)
                #print_p = mp.Process(target=print_guest, args=(qemu,))
                #print_p.start()

                exec_guest_cmds(ssh,  guest_cmds)
                #print(" [+] Waiting for guest execution to finish (1sec)")
                #countdown(1)
                print(" [ ] end_record")
                tn.write("end_record\n".encode('utf-8'))
                end_time = time.time()
                info["record_time"] = end_time - start_time
                time.sleep(1)
                tn.read_very_eager()
                tn.write("quit\n".encode('utf-8'))
                time.sleep(1)
                tn.read_very_eager()
                tn.close()
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print(e)

        print(" [+] terminating qemu")

        qemu.wait()
        all_ok = True
    finally:
        print(" [+] Cancel timeout qemu")
        timeout_timer.cancel()

    try:
        qemu.kill()
    except:
        print(" [-] qemu already died")
    else:
        print(" [+] terminated qemu")
    if print_p is not None:
        print_p.kill()
    if all_ok:
        info_path = os.path.join(record_dir_local, "{recid:s}.info".format(**info))
        pickle.dump(info, open(info_path, 'wb'))
    if os.path.exists(rootfs_qcow):
        os.remove(rootfs_qcow)
