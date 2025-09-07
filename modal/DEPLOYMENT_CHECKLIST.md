# Modal Deployment Checklist

Use this checklist to ensure successful deployment of the NVIDIA prompt classifier to Modal.

## Pre-Deployment ✅

- [ ] Modal account created and verified
- [ ] Dependencies installed: `uv sync`
- [ ] Modal CLI authenticated: `modal setup`
- [ ] JWT secret generated (strong, 32+ characters)
- [ ] Modal secrets created: `modal secret create jwt-auth JWT_SECRET="..."`
- [ ] Verified secrets: `modal secret list`

## Deployment ✅

- [ ] Code deployed: `modal deploy app.py`
- [ ] Deployment successful (no errors)
- [ ] Modal endpoint URL saved from deployment output
- [ ] Health check passes: `curl https://YOUR-MODAL-URL/health`

## Configuration ✅

- [ ] adaptive_ai environment variables configured:
  - [ ] `MODAL_CLASSIFIER_URL` set to Modal endpoint
  - [ ] `JWT_SECRET` matches Modal secret
- [ ] adaptive_ai service restarted/reloaded
- [ ] Modal client logs show successful initialization

## Testing ✅

- [ ] Integration test passes: `uv run python test_integration.py`
- [ ] Health check returns `{"status": "healthy", ...}`
- [ ] JWT authentication working (401 on invalid token)
- [ ] Classification working (returns task types and complexity scores)
- [ ] adaptive_ai service successfully uses Modal API

## Verification ✅

- [ ] Modal dashboard shows active deployment
- [ ] GPU utilization visible during classification requests
- [ ] No errors in Modal logs: `modal logs nvidia-prompt-classifier`
- [ ] adaptive_ai logs show Modal API usage
- [ ] End-to-end classification pipeline working

## Security ✅

- [ ] JWT secret is strong and secure (32+ characters)
- [ ] JWT secret not committed to version control
- [ ] Modal secrets properly configured
- [ ] Authentication working as expected

## Performance ✅

- [ ] Classification latency acceptable (<5 seconds for batch)
- [ ] GPU utilization efficient (T4 sufficient for DeBERTa-v3)
- [ ] Container idle timeout configured (5 minutes)
- [ ] No memory or resource issues

## Documentation ✅

- [ ] Modal endpoint URL documented
- [ ] JWT secret stored securely
- [ ] Team members have access to Modal dashboard
- [ ] Troubleshooting documentation available

## Post-Deployment

- [ ] Monitor Modal dashboard for usage and costs
- [ ] Set up alerts for errors or high usage
- [ ] Plan for secret rotation
- [ ] Document any custom configuration changes

## Rollback Plan (if needed)

- [ ] Keep old adaptive_ai prompt_classifier.py as backup
- [ ] Environment variable to disable Modal client
- [ ] Plan to revert to local GPU inference if needed

---

## Quick Commands Reference

```bash
# Deploy
modal deploy app.py

# Check status  
modal app list
modal logs nvidia-prompt-classifier

# Test
curl -X GET "https://YOUR-MODAL-URL/health"
MODAL_CLASSIFIER_URL="https://YOUR-MODAL-URL" JWT_SECRET="your-secret" uv run python test_integration.py

# Debug
modal secret list
modal logs nvidia-prompt-classifier --follow
```

## Success Criteria

✅ **Deployment Successful**: Modal service running, health check passes  
✅ **Authentication Working**: JWT tokens validated, 401 on invalid tokens  
✅ **Classification Working**: Prompts classified with task types and complexity  
✅ **Integration Complete**: adaptive_ai service uses Modal API seamlessly  
✅ **Performance Acceptable**: Sub-5 second classification, efficient GPU usage  

Your NVIDIA prompt classifier is successfully deployed on Modal! 🎉