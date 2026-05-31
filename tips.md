Go to proj2-scratch.ipynb to see the plan first.

You can look at what's going at this the rapidfireai link. For example:

```
http://127.0.0.1:8853 
```

To ensure it runs as long as possible add this before Always to the config file

```{bash}
-l dsmlp/timeout=86400
```

To check if you actually have 24 hours 

```{bash}
kubectl describe pod vscode-dsmlp
```

You can find that Label looks like:

```
Name:                 vscode-dsmlp
Namespace:            ...
Priority:             ...
Priority Class Name:  normal
Service Account:      default
Node:                 its-dsmlp-n33.ucsd.edu/172.17.86.53
Start Time:           Thu, 28 May 2026 17:20:40 -0700
Labels:               app=launch-sh
                      dsmlp/course=CSE234_SP26_A00
                      dsmlp/timeout=86400 ✅
                      dsmlp/user=emc001
                      gpu-class=medium
                      svcreg=true
Annotations:          cni.projectcalico.org/containerID: 98d1a105ca83ee56b83b034f7bb0e5363998e55d0c908134d0c6e77782af3df4
                      cni.projectcalico.org/podIP: 10.40.158.160/32
                      cni.projectcalico.org/podIPs: 10.40.158.160/32
                      kubectl.kubernetes.io/default-container: c1
Status:               Running
IP:                   10.40.158.160
```

To check GPU, run

```{bash}
nvidia-smi
```

To check disk storage
```{bash}
du -sh ~/.cache/* 2>/dev/null | sort -rh | head -20
```

how to run main.py
```{bash}
# run 1
python main.py \
      --input validation_input.json \
      --output preds_run1.json \
      --schemas_dir schemas/ \
      --checkpoint_path /home/emc001/schema-linker-sft-pipeline/try/logs/proj2-trl-20260531-1859/runs/1/checkpoints/final_checkpoint \
      --model_name Qwen/Qwen2.5-0.5B-Instruct  # or 1.5B depending on which run

# run 2
  python main.py \
    --input validation_input.json \
    --output preds_run2.json \
    --schemas_dir schemas/ \
    --checkpoint_path /home/emc001/schema-linker-sft-pipeline/try/logs/proj2-debug-2config-20260530-1216/runs/2/checkpoints/final_checkpoint \
    --model_name Qwen/Qwen2.5-1.5B-Instruct
```

To restore files from unexpected deletion:
```{bash}
git checkout HEAD -- try/proj2-debug-config.ipynb
```

If you want to run locally:
```{bash}
  KMP_DUPLICATE_LIB_OK=TRUE python main.py \
    --input validation_input.json \
    --output preds_run1.json \
    --schemas_dir schemas/ \
    --checkpoint_path /Users/emilychen/code/schema-linker-sft-pipeline/try/proj2-trl-20260531-1859/runs/0/checkpoints/final_checkpoint \
    --model_name Qwen/Qwen2.5-0.5B-Instruct
```

For leaderboard score:
```
 KMP_DUPLICATE_LIB_OK=TRUE python eval.py \
    --predictions preds_run0-2023.json \
    --gold validation_gold_schema_links.json \
    --schemas_dir schemas/ \
    --questions_input validation_input.json \
    --per_question_out per_question_run0-2023.csv \
    > eval_run1-2023.txt 2>&1
```

proj2-trl-20260531-2023
Gave me:
Evaluated:                  101 questions
Schema-invalid identifiers: tables=0, columns=0

---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.4229
  Recall_T    : 0.4200
  F1_T        : 0.3825
  Table Score : 0.4085     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.2442
  Recall_C    : 0.2785
  F1_C        : 0.1980
  Column Score: 0.2402     ((P+R+F1)/3)

==> Leaderboard Score : 0.3243   (0.5*Table + 0.5*Column)

Per-question metrics written to per_question_run0-2023.csv
