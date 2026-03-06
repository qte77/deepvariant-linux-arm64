#!/bin/bash
# request_aws_quota.sh — Check and request AWS vCPU quota for ARM64 instances.
#
# Checks current On-Demand vCPU quota across common regions, then prints
# the CLI command to request an increase in a specific region.
#
# Usage:
#   bash scripts/request_aws_quota.sh              # Check all regions
#   bash scripts/request_aws_quota.sh us-east-1 64  # Request 64 vCPUs in us-east-1
#
# Prerequisites: AWS CLI configured with credentials that have
# servicequotas:GetServiceQuota and servicequotas:RequestServiceQuotaIncrease
# permissions.

set -euo pipefail

# Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances
QUOTA_CODE="L-1216C47A"
SERVICE_CODE="ec2"

REGIONS=(us-east-1 us-east-2 us-west-2 eu-west-1 eu-central-1 ap-southeast-1)

if [[ $# -ge 2 ]]; then
  # Request mode: request_aws_quota.sh <region> <desired_vcpus>
  REGION="$1"
  DESIRED="$2"
  echo "Requesting vCPU quota increase to ${DESIRED} in ${REGION}..."
  aws service-quotas request-service-quota-increase \
    --service-code "${SERVICE_CODE}" \
    --quota-code "${QUOTA_CODE}" \
    --desired-value "${DESIRED}" \
    --region "${REGION}"
  echo "Request submitted. Check status with:"
  echo "  aws service-quotas list-requested-service-quota-change-history-by-quota \\"
  echo "    --service-code ${SERVICE_CODE} --quota-code ${QUOTA_CODE} --region ${REGION}"
  exit 0
fi

# Check mode: scan all regions
echo "=== AWS On-Demand vCPU Quota (Standard instances) ==="
echo "Quota code: ${QUOTA_CODE}"
echo ""

for region in "${REGIONS[@]}"; do
  current=$(aws service-quotas get-service-quota \
    --service-code "${SERVICE_CODE}" \
    --quota-code "${QUOTA_CODE}" \
    --region "${region}" 2>/dev/null \
    | python3 -c "import sys,json; print(int(json.load(sys.stdin)['Quota']['Value']))" 2>/dev/null \
    || echo "N/A")
  printf "  %-20s %s vCPUs\n" "${region}" "${current}"
done

echo ""
echo "To request increase:"
echo "  bash scripts/request_aws_quota.sh <REGION> <DESIRED_VCPUS>"
echo ""
echo "Example (64 vCPUs in us-east-1):"
echo "  bash scripts/request_aws_quota.sh us-east-1 64"
echo ""
echo "Required for 32-vCPU benchmarks:"
echo "  c7g.8xlarge (Graviton3, 32 vCPU, \$1.15/hr)"
echo "  c8g.8xlarge (Graviton4, 32 vCPU, 64 GB, \$1.36/hr)"
