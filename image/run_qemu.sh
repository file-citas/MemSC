PP=/home/feli/project/mempat_new
KERNEL=$PP/image/linux
IMAGEP=$PP/image
IMAGE=stretch4.img
QEMU=$IMAGEP/qemu/build

$QEMU/qemu-system-x86_64 \
	-m 2G \
	-smp 1 \
	-kernel $KERNEL/arch/x86/boot/bzImage \
	-append "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0 nokaslr" \
	-monitor tcp:127.0.0.1:55555,server,nowait \
	-drive file=$IMAGEP/$IMAGE,format=raw \
	-net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:10021-:22 \
	-net nic,model=e1000 \
	-accel kvm,dirty-ring-size=1024 \
	-nographic \
	-pidfile vm.pid \
	2>&1 | tee vm.log
