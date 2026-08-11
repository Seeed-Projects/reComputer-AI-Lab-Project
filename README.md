# reComputer AI Lab Projects

This repository hosts deployable community projects for the reComputer AI Lab.
Projects are grouped by hardware platform, while container definitions are kept
under `docker/`.

## Raspberry Pi projects

| Project | Hardware | Description |
| --- | --- | --- |
| [Retail Shelf Detection](pi_project/retail_shelf_detection/README.md) | Raspberry Pi 5 + Hailo-8 | Product detection, inventory tracking, pickup events, and low-stock alerts from video |

Published container:

```text
ghcr.io/seeed-projects/recomputer-ai-lab-project/retail_shelf_detection:latest
```

## Container definitions

Hailo-8 Dockerfiles are stored under `docker/hailo8/`. Each Dockerfile uses its
project directory as the build context; project-specific build and run commands
are documented in that project's README.

GitHub Actions builds `linux/arm64` images and publishes them to GHCR when the
corresponding project or Dockerfile changes on `main`.
