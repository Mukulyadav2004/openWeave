#!/usr/bin/env bash
#
# Generate Python gRPC stubs from openweave.proto.  bash proto/generate.sh
set -euo pipefail

PROTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROTO_DIR}/.." && pwd)"

OUT_DIRS=("${ROOT_DIR}/ingestion-server" "${ROOT_DIR}/sdk" "${ROOT_DIR}/benchmark")

for OUT_DIR in "${OUT_DIRS[@]}"; do
  echo "Generating stubs into ${OUT_DIR} ..."
  mkdir -p "${OUT_DIR}"
  python3 -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/openweave.proto"
done

# grpc_tools emits `import openweave_pb2` (not relative), which only resolves
# when the stub's own directory is on sys.path. That is true for each service
# here, so no rewrite is needed — but it is why the stubs are generated INTO
# each service rather than imported from proto/.
echo "Done: openweave_pb2.py + openweave_pb2_grpc.py in ${OUT_DIRS[*]}"
