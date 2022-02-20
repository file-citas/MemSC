import MaFeatures
import argparse
import os
import json
import sys
import py7zr
import multiprocessing
import configparser

MAX_CHUNKS = 256
MAX_Q = 16

def Compressor(queue, stop_token):
    while True:
        line = queue.get()
        if line == stop_token:
            return
        MaFeatures.compress(line, '7z', verbose=True)

def getAnnotSlice(start, end, annot, feat_lim=20):
    ret = []
    for a in annot:
        if a["ma"] >= start and a["ma"] < end-feat_lim:
            new_a = a
            new_a["ma"] -= start
            ret.append(new_a)
    return ret

def split(ma_fn, annot_fn, chunk_size, outdir, prefix, feat_lim, compress_q):
    file_number = 1
    annot = json.load(open(annot_fn, "r"))
    with open(ma_fn, 'rb') as fd:
        chunk = fd.read(chunk_size)
        while chunk:
            print('[+] Writing chunk %d (%d-%d)' % (file_number, chunk_size * (file_number-1), chunk_size * file_number))
            new_annot = getAnnotSlice(chunk_size * (file_number-1), chunk_size * file_number, annot, feat_lim=feat_lim)
            with open(os.path.join(outdir, '%s_%d.annotations' % (prefix, file_number)), 'w') as new_annot_fd:
                json.dumps(new_annot)
                json.dump(new_annot, new_annot_fd, indent=2)
            new_ma_fn = os.path.join(outdir, '%s_%d' % (prefix, file_number))
            with open(new_ma_fn, 'wb') as new_ma_fd:
                new_ma_fd.write(chunk)
            if compress_q is not None:
                compress_q.put(new_ma_fn)
                #MaFeatures.compress(new_ma_fn, '7z')
            file_number += 1
            chunk = fd.read(chunk_size)

def main():
    parser = argparse.ArgumentParser(description='Split memory access dump and annotations')
    #parser.add_argument('-m',
    #        help='memory access dump '
    #        )
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
    parser.add_argument('-b',
            type=int,
            help='bytes per split (approximate, must be multiple of 16)'
            )
    parser.add_argument('-x',
            action='store_true',
            help='use 7z compression'
            )
    parser.add_argument('-nf',
            type=int,
            default=400,
            help='feature cutoff'
            )
    parser.add_argument('-nt',
            type=int,
            default=3,
            help='number of threads for compression'
            )

    args = parser.parse_args()

    if args.b % 64 != 0:
        print("Error chunk size must be multiple of 64")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.c)
    ma_dump = os.path.join(config['QEMU']['dump_dir'], '%s_ma.dump' % args.record_id)
    pr_dump = os.path.join(config['QEMU']['dump_dir'], '%s_pr.dump' % args.record_id)
    pw_dump = os.path.join(config['QEMU']['dump_dir'], '%s_pw.dump' % args.record_id)
    fsize = os.path.getsize(ma_dump)
    if fsize/args.b > MAX_CHUNKS:
        print("Error too many chunks %d (max %d), select larger chunksize" % (fsize/args.b, MAX_CHUNKS))
        sys.exit(1)

    fsize = os.path.getsize(pr_dump)
    if fsize/args.b > MAX_CHUNKS:
        print("Error too many chunks %d (max %d), select larger chunksize" % (fsize/args.b, MAX_CHUNKS))
        sys.exit(1)

    fsize = os.path.getsize(pw_dump)
    if fsize/args.b > MAX_CHUNKS:
        print("Error too many chunks %d (max %d), select larger chunksize" % (fsize/args.b, MAX_CHUNKS))
        sys.exit(1)


    ma_out_dir = os.path.join(config['QEMU']['split_dir'], '%s_ma' % args.record_id)
    if not os.path.exists(ma_out_dir):
        os.mkdir(ma_out_dir)
    pr_out_dir = os.path.join(config['QEMU']['split_dir'], '%s_pr' % args.record_id)
    if not os.path.exists(pr_out_dir):
        os.mkdir(pr_out_dir)
    pw_out_dir = os.path.join(config['QEMU']['split_dir'], '%s_pw' % args.record_id)
    if not os.path.exists(pw_out_dir):
        os.mkdir(pw_out_dir)

    ma_annot = os.path.join(config['QEMU']['dump_dir'], '%s_ma.annotations' % args.record_id)
    pr_annot = os.path.join(config['QEMU']['dump_dir'], '%s_pr.annotations' % args.record_id)
    pw_annot = os.path.join(config['QEMU']['dump_dir'], '%s_pw.annotations' % args.record_id)

    compress_q = None
    compressors = []
    STOP_TOKEN="STOP!!!"
    if args.x:
        m = multiprocessing.Manager()
        compress_q = m.Queue(MAX_Q)
        for i in range(args.nt):
            p = multiprocessing.Process(target = Compressor,
                    args=(compress_q, STOP_TOKEN))
            p.start()
            compressors.append(p)

    print("Splitting MA")
    split(ma_dump, ma_annot, args.b, ma_out_dir, "%s_ma" % args.record_id, args.nf, compress_q)
    print("Splitting PR")
    split(pr_dump, pr_annot, args.b, pr_out_dir, "%s_pr" % args.record_id, args.nf, compress_q)
    print("Splitting PW")
    split(pw_dump, pw_annot, args.b, pw_out_dir, "%s_pw" % args.record_id, args.nf, compress_q)

    for i in range(len(compressors)):
        compress_q.put(STOP_TOKEN)
    for p in compressors:
        p.join()

if __name__ == "__main__":
    main()
