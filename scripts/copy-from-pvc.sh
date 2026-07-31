#!/usr/bin/env bash
# Usage: scripts/copy-from-pvc.sh <pvc-name> <remote-path> <local-dest>
set -euo pipefail
PVC="$1"
REMOTE_PATH="$2"
LOCAL_DEST="$3"
TMP_POD_NAME="copy-from-pvc-$(date +%s)"
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
kubectl cp ${TMP_POD_NAME}:${REMOTE_PATH} "${LOCAL_DEST}"
kubectl delete pod ${TMP_POD_NAME}
