import datetime
from airflow import models

from xlml.utils import gke

job_yaml = """
"apiVersion": "batch/v1"
"kind": "Job"
"metadata":
  "generateName": "pt-nightly-resnet50-mp-fake-v100-x2-"
  "labels":
    "accelerator": "v100-x2"
    "benchmarkId": "pt-nightly-resnet50-mp-fake-v100-x2"
    "frameworkVersion": "pt-nightly"
    "mode": "fake"
    "model": "resnet50-mp"
"spec":
  "activeDeadlineSeconds": 10800
  "backoffLimit": 0
  "completionMode": "Indexed"
  "completions": 2
  "parallelism": 2
  "template":
    metadata:
      labels:
        "headless-svc": 'true'
    "spec":
      "subdomain": headless-svc
      "containers":
      - command:
        - bash
        - -cxeu
        - |
          export PATH=/usr/local/nvidia/bin${PATH:+:${PATH}}
          export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

          nvidia-smi
          pip3 uninstall -y torch torchvision
          pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

          # job_name=$(JOB_NAME)
          # ip=$(getent hosts ${job_name}-0.headless-svc | awk {'print $1'})
          # echo ip: ${ip}

          # torchrun --nnodes=2 --node_rank=$(JOB_COMPLETION_INDEX) --nproc_per_node=2 --rdzv_endpoint=${ip}:12355 /src/xla-master/test/test_train_mp_imagenet.py --model=resnet50 --log_steps=200 --fake_data --nometrics_debug --pjrt_distributed

          echo "${@:0}"

      #     bash
        "args":
        # - "bash"
        - "torchrun"
        - "--nnodes=2"
        - "--node_rank=$(JOB_COMPLETION_INDEX)"
        - "--nproc_per_node=2"
        - "--rdzv_endpoint=$(JOB_NAME)-0.headless-svc:12355"
        - "/src/xla-master/test/test_train_mp_imagenet.py"
        - "--model=resnet50"
        - "--log_steps=200"
        - "--fake_data"
        - "--nometrics_debug"
        - "--num_epochs=1"
        - "--pjrt_distributed"
        # stdin: true
        # tty: true
        "env":
        - "name": "POD_NAME"
          "valueFrom":
            "fieldRef":
              "fieldPath": "metadata.name"
        # - "name": "POD_UID"
        #   "valueFrom":
        #     "fieldRef":
        #       "fieldPath": "metadata.uid"
        - "name": "POD_NAMESPACE"
          "valueFrom":
            "fieldRef":
              "fieldPath": "metadata.namespace"
        - "name": "JOB_NAME"
          "valueFrom":
            "fieldRef":
              "fieldPath": "metadata.labels['job-name']"
        # - "name": "MODEL_DIR"
        #   "value": "$(OUTPUT_BUCKET)/pt-nightly/resnet50-mp/fake/v100-x2/$(JOB_NAME)"
        # - "name": "GPU_NUM_DEVICES"
        #   "value": "2"
        - name: PJRT_DEVICE
          value: CUDA
        # - "name": "XLA_USE_BF16"
        #   "value": "0"
        "image": us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1
        # "image": us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1
        "imagePullPolicy": "Always"
        "name": "train"
        "resources":
          "limits":
            "nvidia.com/gpu": 2
          "requests":
            "cpu": "7.0"
            "memory": "40Gi"
        "volumeMounts":
        - "mountPath": "/dev/shm"
          "name": "dshm"
          "readOnly": false
        - "mountPath": "/src"
          "name": "dshm"
          "readOnly": false
      initContainers:
      - name: clone
        image: alpine
        command:
        - sh
        - -c
        - |
          cd /src
          wget https://github.com/pytorch/xla/archive/refs/heads/master.tar.gz -O - | tar xzf -
        volumeMounts:
        - "mountPath": "/src"
          "name": "dshm"
          "readOnly": false
      "nodeSelector":
        "cloud.google.com/gke-accelerator": "nvidia-tesla-v100"
      "restartPolicy": "Never"
      "volumes":
      - "emptyDir":
          "medium": "Memory"
        "name": "dshm"
      - "emptyDir":
          "medium": "Memory"
        "name": "src"
  "ttlSecondsAfterFinished": 604800
"""


with models.DAG(
    dag_id="pytorchxla-multigpu",
    schedule=None,
    tags=["pytorchxla", "latest", "supported", "xlml"],
    catchup=False,
    start_date=datetime.datetime(2023, 7, 12),
):
    resnet_v100_2x2 = gke.deploy_job(job_yaml)
