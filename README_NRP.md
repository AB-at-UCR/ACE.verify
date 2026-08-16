NRP / Nautilus deployment notes

1) Create PVC (one-time):
   kubectl apply -f assets/templates/nrp-pvc.yaml

2) Stage code + data into PVC:
   bash scripts/copy-to-pvc.sh ./ /workspace aceverify-pvc

3) Build and push container image:
   docker build -t adityabhardwaj24/aceverify:latest .
   docker push adityabhardwaj24/aceverify:latest

4) Launch training Job:
   kubectl apply -f assets/templates/nrp-gpu-job.yaml
   kubectl wait --for=condition=complete job/aceverify-train --timeout=48h

5) Fetch results:
   bash scripts/copy-from-pvc.sh aceverify-pvc /workspace/results ./out

Notes:
- Jobs expect PVC mounted at /workspace with `code/`, `data/`, `results/` subfolders.
- The container `ENTRYPOINT` should run from /workspace so relative paths work.
