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