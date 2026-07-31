#!/usr/bin/env bash
# Usage: scripts/copy-to-pvc.sh <local-path> <pvc-name> [dest-path]
set -euo pipefail
SRC="$1"
PVC="$2"
DEST_PATH="${3:-/workspace}"
TMP_POD_NAME="copy-to-pvc-$(date +%s)"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${TMP_POD_NAME}
spec:
  restartPolicy: Never
  containers:
    - name: copy
      image: alpine:3.18
      command: ["/bin/sh","-c","sleep 3600"]
      volumeMounts:
        - name: workspace
          mountPath: /workspace
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: ${PVC}
EOF
kubectl wait --for=condition=ready pod/${TMP_POD_NAME} --timeout=60s
kubectl cp "${SRC}" ${TMP_POD_NAME}:${DEST_PATH}
kubectl delete pod ${TMP_POD_NAME}
