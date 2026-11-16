# Local installed-package patches

These two patches preserve the installed-package changes that a Git commit of the Omni checkout alone would miss.
They are intended for the inspected environment, not for arbitrary package revisions.
The Omni tracing helper must be available before enabling these patches.

| Patch | Original SHA256 | Patched SHA256 |
| --- | --- | --- |
| `vllm-0.33.0-profile-spans.patch` | `9145e2601f9e93db5e6c7e648c6741cff55f2a49f046800f79311e4977550121` | `957ea780bb75ddac98e8622c98928741ea1972d334d07d875474f7484c2af210` |
| `flashinfer-sampling-profile-spans.patch` | `f56ff6bf9561834362303d4c5b7a80018466babbfdf2f8f73775b1af0746b687` | `f30ba5d1842a130e1e98eb9033300a9faef156dcb66b01a4948f8e5bfa0c91a4` |

Apply only to files matching the original hashes, after making backups. These patches are already applied on the current GH200 environment; do not apply them a second time.

```bash
SITE_PACKAGES=/home/lyy/vllm-omni-startup/envs/vllm-omni-033/lib/python3.12/site-packages
patch --dry-run -p1 -d "$SITE_PACKAGES" < benchmarks/startup/patches/vllm-0.33.0-profile-spans.patch
patch --dry-run -p1 -d "$SITE_PACKAGES" < benchmarks/startup/patches/flashinfer-sampling-profile-spans.patch
# On a matching unpatched environment, repeat the commands without --dry-run.
```

Package reinstall/upgrade can overwrite these local changes. No weights, binaries or compiler caches belong in this directory.
