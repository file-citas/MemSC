#!/usr/bin/python

import shlex
import time
import os
import subprocess
import re
import sys
import getopt

p_fstart = re.compile("([a-f0-9]+) \<(.+)\>:")

class Error(Exception):
    """addr2line exception."""

    def __init__(self, str):
        Exception.__init__(self, str)

class IllegalFileError(Error):
    def __init__(self, addr):
        Error.__init__(self,
                       "The address '0x%x' lacks a valid file mapping." % addr)

class IllegalLineError(Error):
    def __init__(self, addr):
        Error.__init__(self,
                       "The address '0x%x' lacks a line mapping." % addr)

class addr2line:
    def __init__(self, binary, objd_fn, sysmap_fn, addr2line = "/usr/bin/eu-addr2line", text_offset=0xffffffff81000000):
        self.binary = binary
        self.addr2line = addr2line
        self.func_starts = {}
        self.text_offset = text_offset
        with open(sysmap_fn, "r") as fd:
            for l in fd.readlines():
                l = l.rstrip()
                addr, _ , fname = l.split()
                self.func_starts[fname] = int(addr, 16) - self.text_offset
        with open(objd_fn, "r") as fd:
            for l in fd.readlines():
                m = p_fstart.match(l)
                if m:
                    if m.groups()[1] not in self.func_starts.keys():
                        self.func_starts[m.groups()[1]] = int(m.groups()[0], 16) - self.text_offset

    def lookup(self, addrs):
        cmd = [
                self.addr2line,
                "-e", self.binary,
                "-f",
                "-i",
                "-a",
                ]
        for addr in addrs:
            cmd.append("0x%x" % (addr+self.text_offset))

        proc = subprocess.Popen(shlex.split(" ".join(cmd)), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        so, se = proc.communicate()
        sol = so.decode("ascii").splitlines()
        a2f = {}
        li = 0
        while True:
            if li >= len(sol):
                break
            l = sol[li].rstrip()
            if l.startswith("0xff"):
                l_addr = int(l, 16) - self.text_offset
                while True:
                    li = li + 1
                    l = sol[li].rstrip()
                    #print(l)
                    #if 'inline' in l:
                    #    li = li + 1
                    #    continue
                    #if '.h' in l or '.c' in l:
                    #    continue
                    if l.startswith("0xff"):
                        break
                    if li >= len(sol)-2 or sol[li+2].startswith("0xff"):
                        l_fname = l
                        break
                dist = -1
                try:
                    #print("%x" % l_addr)
                    #print("%x" % self.func_starts[l_fname])
                    dist = l_addr - self.func_starts[l_fname]
                except:
                    pass
                a2f[l_addr] = {
                        'fname': l_fname,
                        'dist': dist,
                        }
            li = li + 1

        return a2f

def usage():
    pass
