#!/bin/bash
# Publish Krun One weights and benchmarks to the Hugging Face Hub (needs `hf auth login` with write access to $ORG).
#   ORG=ferreslabs bash scripts/release_hf.sh stage     # build release/stage/* (hard links, no copies of big files)
#   ORG=ferreslabs bash scripts/release_hf.sh check     # verify every artifact checksum inside the staging folders
#   ORG=ferreslabs bash scripts/release_hf.sh upload    # create the repos (public) and upload
set -eu
cd "$(dirname "$0")/.."
ORG=${ORG:-ferreslabs}
ST=release/stage
stage() {
  rm -rf $ST; mkdir -p $ST/{krun-one-v1,krun-one-v1-multimodal,krunbench-v1,kruntechbench}
  cp release/hf/krun-one-v1/README.md $ST/krun-one-v1/
  cp -al artifacts/krun-one-v1-text-final $ST/krun-one-v1/krun-one-v1-text-final
  cp release/hf/krun-one-v1-multimodal/README.md $ST/krun-one-v1-multimodal/
  cp -al artifacts/krun-one-v1-multimodal-candidate $ST/krun-one-v1-multimodal/krun-one-v1-multimodal-candidate
  cp release/hf/krunbench-v1/README.md docs/KRUNBENCH_V1_SPEC.md $ST/krunbench-v1/
  cp -l data/krunbench_v1/{dev,val_fit,val_eval,final_protected}.jsonl data/krunbench_v1/protected_manifest.json $ST/krunbench-v1/
  cp release/hf/kruntechbench/README.md data/v11/swe_smith_allowlist.json $ST/kruntechbench/
  cp -l data/v11/techbench/{protected_v1_labeled,val_v1_labeled,protected_r3,val_r3}.jsonl $ST/kruntechbench/
  cp -l data/v11/tetris_representation/test.jsonl $ST/kruntechbench/tetris_representation_test.jsonl
  du -sh $ST/*
}
check() {
  (cd $ST/krun-one-v1/krun-one-v1-text-final && sha256sum -c --quiet SHA256SUMS) && echo "text artifact OK"
  (cd $ST/krun-one-v1-multimodal/krun-one-v1-multimodal-candidate && sha256sum -c --quiet SHA256SUMS) && echo "multimodal package OK"
  sha256sum $ST/krun-one-v1/krun-one-v1-text-final/model.safetensors | grep -q ^781e5c7b7006da63 && echo "v1 weights sha 781e5c7b OK"
}
upload() {
  hf auth whoami
  hf repos create $ORG/krun-one-v1 --type model --public --exist-ok
  hf repos create $ORG/krun-one-v1-multimodal --type model --public --exist-ok
  hf repos create $ORG/krunbench-v1 --type dataset --public --exist-ok
  hf repos create $ORG/kruntechbench --type dataset --public --exist-ok
  hf upload $ORG/krun-one-v1 $ST/krun-one-v1 . --repo-type model --commit-message "Krun One v1 (text)"
  hf upload $ORG/krun-one-v1-multimodal $ST/krun-one-v1-multimodal . --repo-type model --commit-message "Krun One v1 multimodal (M1.5c)"
  hf upload $ORG/krunbench-v1 $ST/krunbench-v1 . --repo-type dataset --commit-message "KrunBench v1"
  hf upload $ORG/kruntechbench $ST/kruntechbench . --repo-type dataset --commit-message "KrunTechBench v1 + R3 extension"
}
"$@"
