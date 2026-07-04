#!/usr/bin/env bash
# build-run-image.sh — build the per-RUN Docker isolation image (Dockerfile.run)
# and tag it with both the current git SHA and `latest`, so DockerRunner
# (src/bernstein/core/docker_runner.py) can pin runs to `bernstein:<sha>`
# for reproducibility (design doc risk #4) while `bernstein:latest` stays
# the floating dev-convenience tag.
#
# Usage:
#   ./scripts/build-run-image.sh [repo-specific-dockerfile]
#
#   repo-specific-dockerfile   optional. Follow-up hook: a Dockerfile
#                              fragment layered on top of the base
#                              bernstein-run image for a target repo's own
#                              toolchain (e.g. extra apt packages). NOT
#                              implemented yet — passing this arg prints a
#                              clear message and exits nonzero rather than
#                              silently ignoring it.
#
# Uses BuildKit with --cache-from against the existing `bernstein:latest`
# tag so repeated builds reuse cached layers (wheel build + apt install are
# the expensive, cacheable layers; see Dockerfile.run's layer ordering).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOCKERFILE="Dockerfile.run"
REPO_SPECIFIC_DOCKERFILE="${1:-}"

if [[ -n "$REPO_SPECIFIC_DOCKERFILE" ]]; then
  echo "ERROR: repo-specific Dockerfile layering is not implemented yet." >&2
  echo "       Requested overlay: $REPO_SPECIFIC_DOCKERFILE" >&2
  echo "       Follow-up: layer a second 'FROM bernstein:<sha>' build stage" >&2
  echo "       that COPYs in repo-specific setup (extra apt packages, etc.)." >&2
  exit 1
fi

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: $DOCKERFILE not found in $ROOT_DIR" >&2
  exit 1
fi

echo "==> Resolving git SHA for image tag"
GIT_SHA="$(git rev-parse --short HEAD)"
echo "    SHA: $GIT_SHA"

SHA_TAG="bernstein:${GIT_SHA}"
LATEST_TAG="bernstein:latest"

echo "==> Building $DOCKERFILE"
echo "    Tags:       $SHA_TAG, $LATEST_TAG"
echo "    Cache from: $LATEST_TAG (if it exists locally)"

export DOCKER_BUILDKIT=1

set +e
docker build \
  --file "$DOCKERFILE" \
  --cache-from "$LATEST_TAG" \
  --tag "$SHA_TAG" \
  --tag "$LATEST_TAG" \
  .
BUILD_EXIT_CODE=$?
set -e

if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
  echo "ERROR: docker build failed (exit code $BUILD_EXIT_CODE)" >&2
  exit "$BUILD_EXIT_CODE"
fi

echo ""
echo "==> Done"
echo "    Built and tagged: $SHA_TAG"
echo "    Built and tagged: $LATEST_TAG"
echo ""
echo "Verify:"
echo "    docker images | grep bernstein"
