# Generate Test Data:

## Execute VM

```
usage: makerecord_para_verbose.py [-h] -i RECORD_ID [-c C] [-g GUEST_CMDS [GUEST_CMDS ...]] [-ltp LTP [LTP ...]] [--no_kaslr] [-gport GPORT] [-mport MPORT] [-qtimeout QTIMEOUT]

Record Guest Execution

optional arguments:
  -h, --help            show this help message and exit
  -i RECORD_ID, --record_id RECORD_ID
                        ID to be assigned to this record
  -c C, -config C       configuration file (Default=./config)
  -g GUEST_CMDS [GUEST_CMDS ...], --guest_cmds GUEST_CMDS [GUEST_CMDS ...]
                        files containing guest commands to be executed
  -ltp LTP [LTP ...]    syscall testcases to execute
  --no_kaslr            Use Kernel Adress Space Layout Randomization.
  -gport GPORT          guest ssh port
  -mport MPORT          guest monitor port
  -qtimeout QTIMEOUT    qemu timeout
```

Function: Runs a linux instance and records information to collect memory access traces in the next step.

Output:
Records of the run.
Will be stored in `MemSC/data/recordings/<id>.*`.
Files will be processed by makereplay.py

Example:
```
python3 ./makerecord_para_verbose.py -g ./cmds/ssh -c ./config.in -i mytest0
```

## Replay VM
```
usage: makereplay.py [-h] -i RECORD_ID [-c CONFIG] [-f FILTER] [--trace_all] [--exclude_irq] [-qtimeout QTIMEOUT]

Replay Panda Record

optional arguments:
  -h, --help            show this help message and exit
  -i RECORD_ID, --record_id RECORD_ID
                        Panda Record ID.
  -c CONFIG, --config CONFIG
                        configuration file (Default=./replay.config).
  -f FILTER, --filter FILTER
                        Dump will only contain these programs.
  --trace_all           Trace all memory accesses.
  --exclude_irq         Exclude IRQ memory accesses.
  -qtimeout QTIMEOUT    qemu timeout
```
Function: Analyzes the record generated in the previous step to extract memory access traces.
Parameters:
* config: see above
* id: see above (has to be the same id as in makerecord_para_verbose)

Output:
Memory access traces (binary format) and annotations (json format).
Will be stored in `<memaccess_dumps>/<id>.*`.
`<id>_pr.dump`: Physical read access traces
`<id>_pr.annotations`: Annotates system calls in the memory access trace (`<id>_pr.dump`) file.

Example:
```
python3 ./makereplay.py -c ./config.in -i mytest
```

# Prepare Test Data:
## Create differential memory access segments:
```
usage: preproc2.py [-h] -i RECORD_ID -rg RECORD_GLOB -ig INFO_GLOB [-c C] [-ncp NCP] [-n N] [-b B] [-nosys] [-keepirq] [-log] [-nt NT]

convert extracted csv features to random forrest classifiers

optional arguments:
  -h, --help            show this help message and exit
  -i RECORD_ID, --record_id RECORD_ID
                        ID to be assigned to this record
  -rg RECORD_GLOB, --record_glob RECORD_GLOB
                        glob to include dumps
  -ig INFO_GLOB, --info_glob INFO_GLOB
                        glob to include infos
  -c C, -config C       configuration file (Default=./config)
  -ncp NCP              no class percentage
  -n N                  n features
  -b B                  blur
  -nosys                Only generate no sc class
  -keepirq              Keep SCs containing IRQ executions
  -log                  Apply log transform to MA offsets
  -nt NT                number of threads

```

Function: Preprocess chunks to generate a model in the next step. Generated csv files containing feature vectors and labels.

Output:
Compressed csv files in specified output directory.

Example:
```
python3 process/preproc2.py -i nosys4 -c config5.in -ncp 0.0 -n 512 -nosys -keepirq -rg 'feli_nosys*' -ig 'feli_nosys*' -b "$blur" -nt 2
```

## Merge HDF files:
```
usage: process_hdf.py [-h] -o OUTPUT [-drop DROP] [-expand] [-dedup] [-merge] [-nosys] [-sc SC] [-resample] [-max MAX] [-nfeat NFEAT] [-min MIN] [-i INPUT [INPUT ...]]

merge hdf5 files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to output hdf5 file
  -drop DROP            Drop Scs
  -expand               Expand
  -dedup                Deduplicate
  -merge                Add preprocessed feature vectors
  -nosys                Add nosys class
  -sc SC                SC class to dedup
  -resample             Resample
  -max MAX              Max samples per class
  -nfeat NFEAT          Feature Length
  -min MIN              Min samples per class
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Path to input hdf5 files
```

Example:
```
python3 process_hdf.py -o "$t"_nosys_new3_b_"$blur"_512.hdf5 -i $(find <memaccess_preproc>/nosys4_"$t"_nfeat_512_blur_"$blur"_log_0_kirq_1/ -name 'xx_*') -merge
```
