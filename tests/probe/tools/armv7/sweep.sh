#!/bin/zsh
# Build golden cases for armv7-unknown-linux-gnueabihf on the host and run
# them under qemu-user in an armv7 Debian container (glibc 2.36).
#
#   tests/probe/tools/armv7/sweep.sh <lyc> <outdir> <case.py>...
#
# Prints one line per case: OK, OUT (stdout differs), RC (exit code differs),
# COMPILE-FAIL or LINK-FAIL. stderr patterns are not checked here; the .err
# files are left in <outdir>.
#
# One-time setup on macOS (Apple silicon cannot execute AArch32, so the VM
# needs qemu-user, and Fedora's package does not register it on aarch64):
#
#   podman machine init && podman machine start
#   podman machine ssh 'sudo rpm-ostree install -y qemu-user-static-arm'
#   podman machine stop && podman machine start
#   podman machine ssh "echo ':qemu-arm:M::\\x7fELF\\x01\\x01\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x02\\x00\\x28\\x00:\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\x00\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xfe\\xff\\xff\\xff:/usr/bin/qemu-arm-static:F' | sudo tee /etc/binfmt.d/qemu-arm-static.conf && sudo systemctl restart systemd-binfmt"
#   podman --connection podman-machine-default-root build --platform linux/arm/v7 \
#       -t lython-armv7 tests/probe/tools/armv7
#
# The binary is linked inside the container because the host has no armv7
# sysroot; the object comes from `lyc --emit-llvm` and llc, so it is the
# module lyc would have linked. Sources are mounted at their host paths so a
# traceback can read its source lines.
set -u
LYC=$1; OUT=${2:A}; shift 2
LLC=$(brew --prefix llvm)/bin/llc
mkdir -p $OUT
compile() {
  local f=${1:A} b=${1:t:r} d=${1:A:h} k=${1:A:h:t}.${1:t:r}
  if $LYC $f --target armv7-unknown-linux-gnueabihf --emit-llvm -o $OUT/$k.ll > $OUT/$k.cerr 2>&1 &&
     $LLC -O1 -filetype=obj -relocation-model=pic $OUT/$k.ll -o $OUT/$k.o 2>>$OUT/$k.cerr; then
    for ext in stdout exitcode; do [ -f $d/$b.$ext ] && cp $d/$b.$ext $OUT/$k.$ext; done
  else
    echo "COMPILE-FAIL $k: $(grep -m1 error $OUT/$k.cerr | cut -c1-200)"
  fi
  rm -f "${OUT:?}/${k:?}.ll"
}
for f in "$@"; do compile $f & while (( $(jobs | wc -l) >= $(sysctl -n hw.ncpu) )); do sleep 0.2; done; done; wait
cat > $OUT/run1.sh <<'EOR'
o=$1; b=${o%.o}
g++ $o -o $b.bin -lm 2>/dev/null || { echo "LINK-FAIL $b"; exit 0; }
exp=0; [ -f $b.exitcode ] && exp=$(cat $b.exitcode)
timeout 300 ./$b.bin > $b.out 2> $b.err < /dev/null; rc=$?
if [ $rc -ne $exp ]; then echo "RC $b got=$rc exp=$exp"
elif [ -f $b.stdout ] && ! cmp -s $b.out $b.stdout; then echo "OUT $b"
else echo "OK $b"; fi
rm -f $b.bin
EOR
# ${...} around every expansion that precedes a `:`: zsh reads `$d:ro` as `$d`
# with the `:r` modifier and mounts at `<dir>o`.
dirs=(); for f in "$@"; do dirs+=(${f:A:h}); done
mounts=(-v ${OUT}:/w:Z); for d in ${(u)dirs}; do mounts+=(-v ${d}:${d}:ro); done
podman --connection podman-machine-default-root run --rm --platform linux/arm/v7 \
  $mounts -w /w lython-armv7 sh -c 'ls *.o | xargs -P 6 -n 1 sh ./run1.sh'
