# MinIO Storage Setup Guide

This guide will help you set up MinIO object storage for UniRouter profiles on Railway.

## Overview

UniRouter loads profile data from MinIO storage instead of local JSON files. The local files (`adaptive_router/data/unirouter/clusters/*.json`) are preserved as backups but are no longer used.

## What is MinIO?

**MinIO** is an open-source, S3-compatible object storage solution that can be self-hosted or deployed on platforms like Railway.

### MinIO Advantages
- 🔓 **No vendor lock-in** - Open source S3-compatible storage
- 💰 **Predictable costs** - Fixed Railway service cost (~$5-10/month)
- 🔒 **Private deployment** - Your data stays on your infrastructure
- 🚀 **Zero egress fees** - Free data transfer within Railway network

**For detailed MinIO setup instructions**, see `scripts/S3_MIGRATION_GUIDE.md`.

## Prerequisites

- Railway account with project deployed
- MinIO deployed on Railway (see `scripts/S3_MIGRATION_GUIDE.md`)

## Step 1: Deploy MinIO on Railway

Follow the detailed guide in `scripts/S3_MIGRATION_GUIDE.md` to:
1. Deploy MinIO using Railway template or manual setup
2. Create the `adaptive-router-profiles` bucket
3. Get your MinIO endpoint URL and credentials

## Step 2: Migrate Data to MinIO

### 2.1 Install Dependencies

```bash
cd adaptive_router
uv sync  # Install boto3 and other dependencies
```

### 2.2 Set Environment Variables

```bash
export S3_BUCKET_NAME=adaptive-router-profiles
export S3_REGION=us-east-1  # Required by boto3, ignored by MinIO
export MINIO_PUBLIC_ENDPOINT=https://minio-production-xxxx.up.railway.app
export MINIO_ROOT_USER=your-minio-user
export MINIO_ROOT_PASSWORD=your-minio-password
```

### 2.3 Run Migration Script

```bash
# Run migration (from project root)
uv run python adaptive_router/scripts/migrate_to_storage.py
```

Expected output:
```
2025-01-20 10:00:00 - INFO - Loading MinIO configuration from environment...
2025-01-20 10:00:00 - INFO - MinIO storage configured: bucket=adaptive-router-profiles, endpoint=https://minio-production-xxxx.up.railway.app
2025-01-20 10:00:00 - INFO - Loading local JSON files from adaptive_router/data/unirouter/clusters
2025-01-20 10:00:01 - INFO - Loaded profile: 20 clusters, 10 models
2025-01-20 10:00:01 - INFO - Uploading to MinIO: s3://adaptive-router-profiles/global/profile.json
2025-01-20 10:00:02 - INFO - Successfully uploaded to MinIO (size: 376.5KB, version: unversioned)
2025-01-20 10:00:02 - INFO - Verifying upload...
2025-01-20 10:00:03 - INFO - ✅ Verification successful - data matches
============================================================
✅ Migration complete!
============================================================
Storage: MinIO
Endpoint: https://minio-production-xxxx.up.railway.app
Location: s3://adaptive-router-profiles/global/profile.json
Version ID: unversioned
Local files: PRESERVED (not deleted)
============================================================
```

### 2.4 Verify MinIO Upload

Using MinIO CLI:
```bash
# Install MinIO CLI
brew install minio/stable/mc  # macOS

# Configure MinIO client
mc alias set railway https://minio-production-xxxx.up.railway.app your-user your-password

# List files
mc ls railway/adaptive-router-profiles/global/
# Should show: profile.json (376KB)
```

Or using MinIO web console:
1. Open your MinIO console URL
2. Navigate to the `adaptive-router-profiles` bucket
3. Verify `global/profile.json` exists (~376KB)

## Step 3: Configure Railway

### 3.1 Set Environment Variables in Railway

Using Railway CLI:
```bash
railway variables set S3_BUCKET_NAME=adaptive-router-profiles
railway variables set S3_REGION=us-east-1
railway variables set MINIO_PUBLIC_ENDPOINT=https://minio-production-xxxx.up.railway.app
railway variables set MINIO_ROOT_USER=your-minio-user
railway variables set MINIO_ROOT_PASSWORD=your-minio-password
```

Or via Railway Dashboard:
1. Go to your project
2. Click on "Variables" tab
3. Add the following variables:
   - `S3_BUCKET_NAME` = `adaptive-router-profiles`
   - `S3_REGION` = `us-east-1`
   - `MINIO_PUBLIC_ENDPOINT` = `https://minio-production-xxxx.up.railway.app`
   - `MINIO_ROOT_USER` = `your-minio-user`
   - `MINIO_ROOT_PASSWORD` = `your-minio-password`

### 3.2 Deploy to Railway

```bash
# Commit changes
git add .
git commit -m "feat: migrate UniRouter to MinIO storage"

# Deploy
railway up
```

## Step 4: Verify Deployment

### 4.1 Check Railway Logs

```bash
railway logs
```

Look for:
```
INFO - Loading UniRouter profile from MinIO storage...
INFO - MinIO storage configured: bucket=adaptive-router-profiles, endpoint=https://minio-production-xxxx.up.railway.app
INFO - Loading profile from MinIO: s3://adaptive-router-profiles/global/profile.json
INFO - Successfully loaded profile from MinIO (size: 376.5KB, n_clusters: 20)
INFO - Loaded profile from MinIO: 20 clusters
INFO - UniRouter initialized from MinIO: 10 models, lambda range [0.0, 1.0]
```

### 4.2 Test API Endpoint

```bash
# Test routing (if you have an API endpoint)
curl -X POST https://your-railway-app.railway.app/select_model \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python sorting function", "cost_bias": 0.5}'
```

## Troubleshooting

### Error: "MinIO configuration error"

**Cause:** Missing environment variables

**Solution:** Verify all required env vars are set:
```bash
railway variables
```

Required variables:
- `S3_BUCKET_NAME`
- `MINIO_PUBLIC_ENDPOINT`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`

### Error: "Profile not found in MinIO"

**Cause:** Migration script didn't run or bucket doesn't exist

**Solution:**
1. Verify bucket exists in MinIO console
2. Re-run migration script locally (Step 2)
3. Check `mc ls railway/adaptive-router-profiles/global/`

### Error: "Access Denied"

**Cause:** Incorrect MinIO credentials

**Solution:**
1. Verify `MINIO_ROOT_USER` matches your MinIO user
2. Verify `MINIO_ROOT_PASSWORD` matches your MinIO password
3. Check MinIO service is running: `railway logs -s minio`

### Error: "Corrupted profile data"

**Cause:** JSON file was corrupted during upload

**Solution:**
1. Re-run migration script: `uv run python adaptive_router/scripts/migrate_to_storage.py`
2. Download and verify: `mc cat railway/adaptive-router-profiles/global/profile.json | jq .`

## Cost Estimation

### MinIO on Railway (1 global profile)
```
Railway service cost:
- MinIO service: ~$5-10/month (fixed)
- Storage: Free for first 5GB (376KB profile is negligible)
- Network: Free within Railway network

Total: ~$5-10/month (fixed cost)
```

### Future Scaling (1000 user profiles)
```
Storage:
- 376MB × included in Railway plan

Total: ~$5-10/month (same fixed cost, no per-request charges)
```

## Rollback Procedure

If you need to revert to local files:

1. **Stop using MinIO temporarily:**
   ```bash
   # Remove MinIO env vars from Railway
   railway variables delete S3_BUCKET_NAME
   railway variables delete MINIO_PUBLIC_ENDPOINT
   railway variables delete MINIO_ROOT_USER
   railway variables delete MINIO_ROOT_PASSWORD
   ```

2. **Code rollback:**
   ```bash
   git revert <commit-hash>
   railway up
   ```

3. **Local files are still there:**
   - Location: `adaptive_router/data/unirouter/clusters/*.json`
   - Never deleted by migration script
   - Can be used as-is

## Next Steps

### For Per-User Profiles (Future)

When you need per-user profiles:

1. **Update MinIO key pattern:**
   - Current: `global/profile.json`
   - Future: `users/{user_id}/profile.json`

2. **Update code:**
   ```python
   # In unirouter_service.py
   profile_data = storage_loader.load_profile(user_id=request.user_id)
   ```

3. **Migration:**
   - Each user trains their model → uploads to MinIO
   - Key: `s3://adaptive-router-profiles/users/{user_id}/profile.json`

## Support

If you encounter issues:
1. Check Railway logs: `railway logs`
2. Verify MinIO access: `mc ls railway/adaptive-router-profiles/`
3. Test locally with env vars set
4. Review error messages in logs
5. Check MinIO console for bucket status

## Summary

✅ **Local JSON files preserved** (backup)
✅ **MinIO storage configured** (production)
✅ **Railway environment configured**
✅ **Migration verified**
✅ **Cost: ~$5-10/month** (fixed, predictable)
✅ **No vendor lock-in** (open source)

You're now using MinIO for UniRouter profile storage! 🎉
