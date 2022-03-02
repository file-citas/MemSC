# Setup

## Extract Rootfs
```
cd MemSC/image
tar xf stretch5.qcow2.tar.gz
```

## Build Guest Kernel
```
cd MemSC/image
git clone https://github.com/torvalds/linux.git
cd MemSC/image/linux
git checkout 8fe28cb58bcb
git apply ../kernel.patch
cp ../kernel.config .config
make oldconfig
make CC=gcc-7
```

## Create Output Directories
```
mkdir -p MemSC/data/recordings
mkdir -p MemSC/data/memaccess_dumps
mkdir -p MemSC/data/memaccess_preproc
mkdir -p MemSC/data/memaccess_syscalls
```

## Build Panda
```
cd MemSC/panda
git clone https://github.com/panda-re/panda.git
cd panda
git checkout 4ffdb5709b6376a6bc43018ef1a9ad23d53624b5
git apply ../panda.patch
```
Check panda README for build instructions.

## Update Configuration
Replace `<MemSC>` with full path to repo.
```
[QEMU]
qemu_bin = <MemSC>/panda/panda/build/x86_64-softmmu/panda-system-x86_64
qemu_img = <MemSC>/image/qemu/build/qemu-img
bzimage = <MemSC>/image/linux/arch/x86_64/boot/bzImage
rootfs = <MemSC>/image/stretch2.img
rootfs_qcow = <MemSC>/image/stretch5.qcow2
osi_kconf_file = <MemSC>/image/kernelinfo.conf
osi_kconf_group = stretch420
rec_dir = <MemSC>/data/recordings/
dump_dir = <MemSC>/data/memaccess_dumps/
split_dir = <MemSC>/data/memaccess_split/
preproc_dir = <MemSC>/data/memaccess_preproc/
syscalls_dir = <MemSC>/data/memaccess_syscalls/

[MONITOR]
ip = 127.0.0.1
listen = 0.0.0.0
port = 1236

[GUEST]
ip = localhost
user = root
pw = root
port = 10024
```
