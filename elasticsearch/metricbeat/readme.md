# Notes:

We use yq tool to modify the metricbeat config
to set experiment and variant fields dynamically, so that every metric beat will contain what experiment that log is from

## Install yq
http://mikefarah.github.io/yq/

https://github.com/mikefarah/yq/releases/tag/1.14.0

```
wget https://github.com/mikefarah/yq/releases/download/1.14.0/yq_linux_amd64

chmod +x yq_linux_amd64
```

```
sudo /home/ubuntu/yq_linux_amd64 w -i /etc/metricbeat/metricbeat.yml fields.experiment experiment_id
sudo /home/ubuntu/yq_linux_amd64 w -i /etc/metricbeat/metricbeat.yml fields.variant variant_id
```

Logs will look like these
```
{
  "@timestamp": "2017-12-20T11:39:59.405Z",
  "beat": {
    "hostname": "ip-172-30-0-191",
    "name": "ip-172-30-0-191",
    "version": "5.6.5"
  },
  "fields": {
    "experiment": "lolyq",
    "variant": 2
  },
  "metricset": {
    "module": "system",
    "name": "load",
    "rtt": 165
  },
  "system": {
    "load": {
      "1": 0.05,
      "15": 0,
      "5": 0.01,
      "norm": {
        "1": 0.025,
        "15": 0,
        "5": 0.005
      }
    }
  },
  "type": "metricsets"
}
{
  "@timestamp": "2017-12-20T11:39:59.405Z",
  "beat": {
    "hostname": "ip-172-30-0-191",
    "name": "ip-172-30-0-191",
    "version": "5.6.5"
  },
  "fields": {
    "experiment": "lolyq",
    "variant": 2
  },
  "metricset": {
    "module": "system",
    "name": "cpu",
    "rtt": 103
  },
  "system": {
    "cpu": {
      "cores": 2,
      "idle": {
        "pct": 1.994
      },
      "iowait": {
        "pct": 0
      },
      "irq": {
        "pct": 0
      },
      "nice": {
        "pct": 0
      },
      "softirq": {
        "pct": 0
      },
      "steal": {
        "pct": 0
      },
      "system": {
        "pct": 0.004
      },
      "user": {
        "pct": 0.002
      }
    }
  },
  "type": "metricsets"
}
{
  "@timestamp": "2017-12-20T11:39:59.405Z",
  "beat": {
    "hostname": "ip-172-30-0-191",
    "name": "ip-172-30-0-191",
    "version": "5.6.5"
  },
  "fields": {
    "experiment": "lolyq",
    "variant": 2
  },
  "metricset": {
    "module": "system",
    "name": "memory",
    "rtt": 139
  },
  "system": {
    "memory": {
      "actual": {
        "free": 3621609472,
        "used": {
          "bytes": 324071424,
          "pct": 0.0821
        }
      },
      "free": 2777767936,
      "swap": {
        "free": 0,
        "total": 0,
        "used": {
          "bytes": 0,
          "pct": 0
        }
      },
      "total": 3945680896,
      "used": {
        "bytes": 1167912960,
        "pct": 0.296
      }
    }
  },
  "type": "metricsets"
}
```