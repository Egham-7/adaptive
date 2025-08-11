# 🚀 Web Performance Testing Suite

A powerful, easy-to-use performance testing framework designed to help you load test your APIs with confidence. Built for developers who want professional-grade performance testing without the complexity.

## ✨ Why Use This?

- **🎯 Dead Simple Setup** - Get running in 30 seconds
- **📈 Scales Effortlessly** - Test 1 endpoint or 100+ endpoints  
- **🔧 Zero Configuration Hassle** - Smart defaults that just work
- **📊 Beautiful Reports** - Professional HTML reports with charts
- **⚡ Multiple Test Engines** - Choose what fits your needs best

## 🚀 Quick Start

```bash
# 1. Test it works
uv run web-perf validate

# 2. See what you can test
uv run web-perf list-endpoints  

# 3. Run your first performance test
uv run web-perf test quick_test

# 4. Get detailed results
open results/performance_report_*.html
```

**That's it! You just ran a professional performance test.** 🎉

## 🎯 Perfect For

- **API Load Testing** - How many users can your API handle?
- **Performance Monitoring** - Catch slowdowns before users do  
- **CI/CD Integration** - Fail builds if performance degrades
- **Capacity Planning** - Plan for traffic spikes with confidence

## 📋 What You Get Out of the Box

Your API endpoint `https://llmadaptive.uk/api/v1/select-model` is already configured and ready to test with:

- ✅ **4 Test Scenarios**: From quick tests to heavy load
- ✅ **Smart Test Data**: Realistic prompts that stress your API properly  
- ✅ **Professional Metrics**: Response times, success rates, throughput
- ✅ **Automatic Reports**: HTML, CSV, and JSON exports with charts

## 🎪 Available Test Scenarios

| Scenario | Users | Duration | Best For |
|----------|--------|----------|----------|
| `quick_test` | 10 users | 1 min | Development & debugging |
| `light_load` | 10 users | 1 min | Basic load validation |
| `medium_load` | 50 users | 5 min | Typical production load |  
| `heavy_load` | 200 users | 10 min | Peak traffic simulation |

```bash
# Run any scenario
uv run web-perf test <scenario_name>
```

## ⚡ Choose Your Engine

### 🏃‍♂️ Async Engine (Recommended)
Super fast, handles lots of concurrent users efficiently.
```bash
uv run web-perf test medium_load --engine async
```

### 🌐 Locust Engine  
Industry standard with real-time web dashboard.
```bash
# Generate a standalone locustfile
uv run web-perf generate-locustfile --output locustfile.py

# Start Locust web UI
uv run locust -f locustfile.py --web-port 8092
# Then open http://localhost:8092
```

### 🔧 Sync Engine
Simple and predictable for basic testing.
```bash
uv run web-perf test quick_test --engine sync
```

## 📊 What Your Reports Look Like

After each test, you automatically get:

- **📈 HTML Report** - Beautiful charts and metrics dashboard
- **📋 CSV Data** - Raw data for custom analysis  
- **🔍 JSON Results** - Machine-readable metrics for CI/CD
- **📊 Charts** - Response time distributions and trends

## 🎯 Adding More Endpoints

Want to test more endpoints? Just edit `config/endpoints.yaml`:

```yaml
endpoints:
  # Your existing endpoint
  select_model:
    path: "/api/v1/select-model"
    method: "POST"
    weight: 70
    enabled: true
    test_data:
      - prompt: "What is machine learning?"
      - prompt: "Explain quantum computing"

  # Add new endpoints easily!
  chat_completions:
    path: "/api/v1/chat/completions" 
    method: "POST"
    weight: 20
    enabled: true
    test_data:
      - messages: [{"role": "user", "content": "Hello!"}]
        
  health_check:
    path: "/health"
    method: "GET" 
    weight: 10
    enabled: true
```

**The framework automatically:**
- ✅ Discovers your new endpoints
- ✅ Distributes load based on weights
- ✅ Generates per-endpoint metrics  
- ✅ Includes everything in reports

## 🛠️ All Commands

| Command | What It Does |
|---------|-------------|
| `web-perf validate` | Check your configuration |
| `web-perf list-endpoints` | Show configured API endpoints |
| `web-perf list-scenarios` | Show available test scenarios |
| `web-perf test <scenario>` | Run a performance test |
| `web-perf generate-locustfile` | Generate standalone locustfile for Locust web UI |
| `web-perf init` | Create new configuration |

## 🚨 Understanding Your Results  

### ✅ Good Performance
- **Success Rate**: 95%+ 
- **P95 Response Time**: Under 2 seconds
- **P99 Response Time**: Under 5 seconds

### ⚠️ Needs Attention  
- **Success Rate**: Below 95%
- **Response Times**: Trending upward
- **Error Rate**: Above 5%

### 🔥 Performance Issues
- **Timeouts**: Requests timing out
- **5xx Errors**: Server errors under load
- **Memory Issues**: Response times spiking

## 🔗 Integration Examples

### GitHub Actions
```yaml
- name: Performance Test
  run: |
    cd benchmarks/web-perf
    uv run web-perf test load_test
    # Build fails if performance thresholds aren't met
```

### Daily Monitoring
```bash
# Add to cron for daily performance monitoring
0 2 * * * cd /path/to/web-perf && uv run web-perf test medium_load
```

## 🆘 Need Help?

**Common Issues:**
- **"Connection refused"** → Check if your API is running
- **"Timeouts"** → Your API might be under heavy load  
- **"Configuration errors"** → Run `web-perf validate` to check

**Commands to try:**
```bash
# Test your API is reachable
curl https://llmadaptive.uk/api/v1/select-model

# Check configuration 
uv run web-perf validate

# Start small
uv run web-perf test quick_test
```

## 🎉 Ready to Scale?

This framework grows with you:
- **Start small** with quick tests during development
- **Scale up** to production load testing  
- **Integrate** with your CI/CD pipeline
- **Monitor** performance continuously

Your API performance testing journey starts now! 🚀

---

*Built with ❤️ for developers who care about performance*