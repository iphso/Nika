#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... CF_ACCOUNT_ID=... ./scripts/upload_r2_assets.sh
# Or set R2_ENDPOINT directly:
#   R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com ./scripts/upload_r2_assets.sh

R2_BUCKET=${R2_BUCKET:-nika-assets}
VIDEO_DIR=${VIDEO_DIR:-tmp/r2_upload_videos}
PLOTS_DIR=${PLOTS_DIR:-docs/assets/plots}

if [[ -z "${R2_ENDPOINT:-}" ]]; then
  if [[ -z "${CF_ACCOUNT_ID:-}" ]]; then
    echo "error: set CF_ACCOUNT_ID or R2_ENDPOINT" >&2
    exit 1
  fi
  R2_ENDPOINT="https://${CF_ACCOUNT_ID}.r2.cloudflarestorage.com"
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found" >&2
  exit 1
fi

# Create bucket if missing
aws --endpoint-url "$R2_ENDPOINT" s3api head-bucket --bucket "$R2_BUCKET" 2>/dev/null || \
  aws --endpoint-url "$R2_ENDPOINT" s3api create-bucket --bucket "$R2_BUCKET" --region auto

# Upload videos (if present)
if [[ -d "$VIDEO_DIR" ]]; then
  aws --endpoint-url "$R2_ENDPOINT" s3 sync "$VIDEO_DIR" "s3://$R2_BUCKET/videos" \
    --cache-control "public, max-age=31536000, immutable"
fi

# Upload plots
if [[ -d "$PLOTS_DIR" ]]; then
  aws --endpoint-url "$R2_ENDPOINT" s3 sync "$PLOTS_DIR" "s3://$R2_BUCKET/plots" \
    --cache-control "public, max-age=31536000, immutable"
else
  echo "error: plots dir not found: $PLOTS_DIR" >&2
  exit 1
fi

echo "Upload complete: s3://$R2_BUCKET/videos and s3://$R2_BUCKET/plots"
