#!/usr/bin/env bash
# Copied from fracture-md scripts/setup_dft_env.sh (commit bffe3e1). Only the paths changed: the lock and
# sssp_manifest.py sit next to this script, and the pseudopotentials go to tools/dft/fracture_md_qe/weights/pseudos.
# Reproducible setup of the OPTIONAL plane-wave DFT stack (Quantum ESPRESSO pw.x + SSSP pseudopotentials).
#
# What it does (idempotent; re-running only verifies):
#   1. creates an ISOLATED conda env  $FMD_DFT_ENV  (default ~/miniconda3/envs/fmd-dft) from conda-forge:
#      qe 7.5 (pw.x, MPI build against Open MPI 5, OpenBLAS, FFTW, ScaLAPACK, ELPA) + python 3.12.
#      If fmd-dft-<platform>.lock in this folder (conda "explicit" spec with md5s) exists for this platform it is
#      used, so the exact package builds are reproduced; otherwise the version pins below are solved and a
#      lock is written next to this script.  base, .venv and .venv-mace are never touched.
#   2. downloads the SSSP 1.3.0 PBE efficiency + precision libraries from the Materials Cloud Archive
#      (record rcyfm-68h65, DOI 10.24435/materialscloud:f3-ym, CC-BY-4.0 with per-family licences, see
#      LICENSE.txt in the download), verifies every archive against the md5 published by the archive,
#      extracts them to $FMD_PSEUDO_ROOT/SSSP_1.3.0_PBE_{efficiency,precision}/ and verifies every UPF
#      against the md5 listed in the library JSON;
#   3. writes $FMD_PSEUDO_ROOT/MANIFEST_SSSP_1.3.0_PBE.json (source, DOI, licence, md5 + sha256 of every
#      file) and SHA256SUMS; engines.dft_qe re-hashes the UPF files it uses at run time;
#   4. (SMOKE=1) runs a 2-atom Si SCF with pw.x on $FMD_DFT_CORES (default: one rank, no pinning).
#
#   bash tools/dft/fracture_md_qe/setup/setup_dft_env.sh
#   SMOKE=1 FMD_DFT_CORES=10,11,12,13 bash tools/dft/fracture_md_qe/setup/setup_dft_env.sh
#
# weights/ is git-ignored: the pseudopotentials are downloaded, never committed.
# Recorded 2026-10-04 on linux-aarch64 (DGX Spark, 20 Arm cores): qe 7.5 hc91ee90_1, openmpi 5.0.11,
# libblas 3.11.0 (openblas), fftw 3.3.11, scalapack 2.2.0, elpa 2025.06.001.  See docs/DFT.md.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="${FMD_DFT_ENV:-$HOME/miniconda3/envs/fmd-dft}"
QE_SPEC="${QE_SPEC:-qe=7.5}"
PY_SPEC="${PY_SPEC:-python=3.12}"
PSEUDO_ROOT="${FMD_PSEUDO_ROOT:-$SETUP_DIR/../weights/pseudos}"
SSSP_VERSION="1.3.0"
SSSP_RECORD="rcyfm-68h65"
SSSP_DOI="10.24435/materialscloud:f3-ym"
SSSP_API="https://archive.materialscloud.org/api/records/${SSSP_RECORD}/files"
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
LOCK="$SETUP_DIR/fmd-dft-${PLATFORM}.lock"

# md5 checksums published by the Materials Cloud Archive for record rcyfm-68h65 (files listing, 2026-10-04)
declare -A SSSP_MD5=(
  [LICENSE.txt]=c0f9df05fb4cd4921f4caa25606815ef
  [README.md]=23264819e0fa3ac4726a25f8c48b8a29
  [CHANGELOG.txt]=0dca28e48b3415f1d2304a099a59ba63
  [versions.yaml]=12becada8baa932e5386b8d6f6843626
  [files_description.md]=05d400b6796d012e2f49c74b3c04b037
  [SSSP_1.3.0_PBE_efficiency.json]=3153c4b20fc90a44fba0236627525644
  [SSSP_1.3.0_PBE_precision.json]=1692c5c9ce89e1c7c783f8f0eee0cbfa
  [SSSP_1.3.0_PBE_efficiency.tar.gz]=a58f1b3373f330179fd0832c48bb9a52
  [SSSP_1.3.0_PBE_precision.tar.gz]=fde94756886f32ada7bf597547557eb5
)

log() { echo "[setup_dft_env] $*"; }

# ---------------------------------------------------------------------------------------------- 1. conda env
if [[ -x "$ENV_PREFIX/bin/pw.x" ]]; then
  log "env exists: $ENV_PREFIX"
else
  if [[ -f "$LOCK" ]]; then
    log "creating $ENV_PREFIX from lock $LOCK"
    "$CONDA" create -y -p "$ENV_PREFIX" --file "$LOCK"
  else
    log "creating $ENV_PREFIX from conda-forge ($QE_SPEC $PY_SPEC)"
    "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels "$QE_SPEC" "$PY_SPEC"
  fi
fi
if [[ ! -f "$LOCK" ]]; then
  "$CONDA" list -p "$ENV_PREFIX" --explicit --md5 > "$LOCK"
  log "wrote lock $LOCK"
fi
log "packages: $(ls "$ENV_PREFIX/conda-meta" | grep -E '^(qe|openmpi|libblas|fftw|scalapack|elpa)-' | sed 's/\.json$//' | tr '\n' ' ')"

# ------------------------------------------------------------------------------------- 2. SSSP download
DL="$PSEUDO_ROOT/_downloads"
mkdir -p "$DL"
for f in "${!SSSP_MD5[@]}"; do
  want="${SSSP_MD5[$f]}"
  if [[ -f "$DL/$f" ]] && [[ "$(md5sum "$DL/$f" | cut -d' ' -f1)" == "$want" ]]; then
    continue
  fi
  log "downloading $f"
  curl -sSL --fail -o "$DL/$f.part" "$SSSP_API/$f/content"
  got="$(md5sum "$DL/$f.part" | cut -d' ' -f1)"
  if [[ "$got" != "$want" ]]; then
    echo "md5 mismatch for $f: got $got, archive lists $want" >&2
    exit 1
  fi
  mv "$DL/$f.part" "$DL/$f"
done
log "all SSSP archive files match the archive md5s"

for proto in efficiency precision; do
  name="SSSP_${SSSP_VERSION}_PBE_${proto}"
  dest="$PSEUDO_ROOT/$name"
  if [[ ! -d "$dest" ]] || [[ -z "$(ls -A "$dest" 2>/dev/null)" ]]; then
    mkdir -p "$dest"
    tar xzf "$DL/$name.tar.gz" -C "$dest"
    log "extracted $name ($(ls "$dest" | wc -l) files)"
  fi
  cp -f "$DL/$name.json" "$PSEUDO_ROOT/$name.json"
done
cp -f "$DL/LICENSE.txt" "$PSEUDO_ROOT/SSSP_LICENSE.txt"

# ------------------------------------------------------------------- 3. verify UPF md5s, write the manifest
PY="${PYTHON:-$ENV_PREFIX/bin/python}"
"$PY" "$SETUP_DIR/sssp_manifest.py" --root "$PSEUDO_ROOT" --version "$SSSP_VERSION" \
  --record "$SSSP_RECORD" --doi "$SSSP_DOI"

# ------------------------------------------------------------------------------------------ 4. smoke test
if [[ "${SMOKE:-0}" == "1" ]]; then
  WORK="$(mktemp -d)"
  CORES="${FMD_DFT_CORES:-}"
  cat > "$WORK/si.pwi" <<EOF
&CONTROL
  calculation = 'scf', tstress = .true., tprnfor = .true., outdir = './tmp', prefix = 'si',
  pseudo_dir = '$PSEUDO_ROOT/SSSP_${SSSP_VERSION}_PBE_efficiency'
/
&SYSTEM
  ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1, ecutwfc = 30, ecutrho = 240
/
&ELECTRONS
  conv_thr = 1e-10
/
ATOMIC_SPECIES
Si 28.085 Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS automatic
6 6 6 0 0 0
EOF
  cd "$WORK"
  if [[ -n "$CORES" ]]; then
    NP="$(echo "$CORES" | tr ',' '\n' | wc -l)"
    # same launch as the adapter (dft_qe.QERunner): shared-memory transport, which avoids ~3 s of UCX start-up
    OMP_NUM_THREADS=1 "$ENV_PREFIX/bin/mpirun" -np "$NP" --map-by "pe-list=${CORES}:ordered" --bind-to core \
      --mca pml ob1 --mca btl self,sm \
      -x OMP_NUM_THREADS=1 "$ENV_PREFIX/bin/pw.x" -nk "$NP" -in si.pwi > si.pwo 2> si.err
  else
    OMP_NUM_THREADS=1 "$ENV_PREFIX/bin/pw.x" -in si.pwi > si.pwo 2> si.err
  fi
  grep -E "Program PWSCF|^!|P=|PWSCF  *:|JOB DONE" si.pwo
  cd /
  if [[ "${KEEP_SMOKE:-0}" == "1" ]]; then
    log "smoke test output kept in $WORK"
  else
    rm -rf -- "$WORK"
    log "smoke test directory removed (KEEP_SMOKE=1 keeps it)"
  fi
fi
log "done.  pw.x = $ENV_PREFIX/bin/pw.x ; pseudopotentials under $PSEUDO_ROOT"
