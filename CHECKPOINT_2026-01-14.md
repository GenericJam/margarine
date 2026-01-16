# Checkpoint: Memory Leak Fixes Applied - Ready to Test

**Date**: 2026-01-14 18:35 PST
**Status**: All memory leak fixes implemented, about to run integration tests
**Session**: Post-crash analysis and comprehensive memory leak remediation

## What Happened Before

Computer crashed during integration test #4 (1024x1024 image generation) at VAE decode step. Last known memory: 29GB available. Crash likely caused by accumulated memory leaks.

## What We Fixed (All Complete ✅)

### 1. Globals Accumulation (CRITICAL FIX)
**File**: `lib/margarine/python/pythonx_server.ex`
- **Issue**: Python globals dictionary accumulated 200MB+ per call, never cleaned
- **Fix**: Added `build_call_globals/2` helper function (lines 425-438)
  - Only keeps essential module refs: `["flux_pythonx", "initialized", "init_result"]`
  - Discards `new_globals` in ALL handle_call functions (lines 241, 296, 327, 353, 372)
- **Impact**: Prevents multi-GB memory leaks from accumulated binary data

### 2. Python Garbage Collection
**File**: `priv/python/flux_pythonx.py`
- **Issue**: Intermediate tensors (200MB-2GB) not freed until memory pressure
- **Fix**: Added explicit cleanup in 3 functions:
  - `transformer_forward()` (lines 284-289): del + gc.collect()
  - `vae_decode()` (lines 358-362): del + gc.collect()
  - `encode_prompt()` (lines 167-170): del + gc.collect()
- **Impact**: Forces immediate memory release instead of waiting for Python GC

### 3. Memory Monitoring
**File**: `lib/margarine/pipeline.ex`
- **Issue**: No visibility into memory usage during generation
- **Fix**: Added monitoring functions:
  - `check_memory_and_gc/3` (lines 332-353): Monitor before each denoising step
  - `force_python_gc/1` (lines 355-373): Trigger Python GC on demand
  - Pre-VAE decode check (lines 241-258): Check memory before decode
- **Trigger**: Forces Python GC if memory drops below 5GB
- **Impact**: Early warning + proactive cleanup prevents OOM

### 4. Test Memory Monitoring
**File**: `test/integration/flux_generation_test.exs`
- **Issue**: Can't track memory between test runs
- **Fix**: Enhanced setup/teardown (lines 67-124):
  - Log memory before test
  - Log memory before cleanup
  - Force Erlang GC: `:erlang.garbage_collect()`
  - Log memory after cleanup
- **Impact**: Clear visibility into memory usage patterns

### 5. Memory Leak Detection Test
**File**: `test/integration/memory_leak_test.exs` (NEW FILE)
- **Tests**:
  1. 10 generations of 512x512 images (small)
  2. 3 generations of 1024x1024 images (large)
- **Assertions**:
  - Fails if >3GB leaked (small)
  - Fails if >4GB leaked (large)
- **Impact**: Automated leak detection for CI

## What We're About to Test

Running integration tests with ALL memory leak fixes applied.

**Command**: `mix test --only integration --trace`

**Expected behavior**:
1. ✅ Memory logs at each step: `[Pipeline] Step 1/4: 25.3GB available`
2. ✅ No accumulation between tests
3. ✅ Python GC triggers if memory drops
4. ✅ Cleanup messages: `✓ Python GC completed`
5. ✅ All 4 tests pass without crash

**Potential outcomes**:
- **PASS**: All tests complete, memory stable → Fixes worked! 🎉
- **CRASH at same point**: May indicate Pythonx library issue or MPS backend problem
- **CRASH earlier**: Something wrong with our fixes
- **CRASH later**: Fixes helped but not enough, need more aggressive cleanup

## If Crash Occurs Again

### Immediate Actions
1. Check for crash dump: `ls -la ~/code/margarine/erl_crash.dump`
2. Check memory status: `vm_stat | head -10`
3. Check running processes: `ps aux | grep -E "(python|beam)" | grep -v grep`
4. Check last test output: `tail -100 integration_test_output.log`

### Next Investigation Steps
1. **Pythonx Memory Leak**: May need to work around library issues
2. **Increase GC frequency**: Lower threshold from 5GB to 10GB
3. **Add delays**: Sleep between denoising steps to allow cleanup
4. **Reduce parallelism**: Even more conservative resource usage
5. **MPS backend**: Try with CPU backend to isolate MPS issues

### Files to Check
- `lib/margarine/python/pythonx_server.ex` - Globals handling
- `priv/python/flux_pythonx.py` - Python GC
- `lib/margarine/pipeline.ex` - Memory monitoring
- `test/integration/flux_generation_test.exs` - Test setup

## Current System State

**Memory**: ~38GB free (vm_stat shows 2,350,517 free pages × 16KB = ~38GB)
**Processes**: 2 BEAM processes (ElixirLS), no Python processes
**Build**: Successfully compiled with minor warnings (unused variables)
**Git status**: Working directory with uncommitted changes (all fixes)

## Code Locations of All Fixes

```
lib/margarine/python/pythonx_server.ex
  - Line 230: build_call_globals() call in encode_prompt
  - Line 241: Discard new_globals in encode_prompt
  - Line 280: build_call_globals() call in transformer_forward
  - Line 296: Discard new_globals in transformer_forward
  - Line 317: build_call_globals() call in vae_decode
  - Line 327: Discard new_globals in vae_decode
  - Line 342: build_call_globals() call in generate_latents
  - Line 353: Discard new_globals in generate_latents
  - Line 372: Discard new_globals in get_model_info
  - Line 425-438: build_call_globals() helper function

priv/python/flux_pythonx.py
  - Line 167-170: GC in encode_prompt()
  - Line 284-289: GC in transformer_forward()
  - Line 358-362: GC in vae_decode()

lib/margarine/pipeline.ex
  - Line 204: check_memory_and_gc() call in denoising loop
  - Line 241-258: Memory check before VAE decode
  - Line 332-353: check_memory_and_gc() function
  - Line 355-373: force_python_gc() function

test/integration/flux_generation_test.exs
  - Line 68-76: Memory logging in setup
  - Line 84-92: Memory logging before cleanup
  - Line 108-120: Erlang GC + memory logging after cleanup

test/integration/memory_leak_test.exs
  - Entire new file with 2 leak detection tests
```

## Test Plan

**Step 1**: Run integration tests (safer, ~5-10 min)
```bash
cd ~/code/margarine
mix test --only integration --trace 2>&1 | tee integration_test_output_with_fixes.log
```

**If Step 1 passes**: Run memory leak test (thorough, ~15-20 min)
```bash
mix test --only integration:memory_leak --trace 2>&1 | tee memory_leak_test_output.log
```

**If Step 1 crashes**: Analyze crash, check logs, implement additional fixes

## Success Criteria

✅ All 4 integration tests pass
✅ Memory logs show stable/decreasing usage
✅ No crash during VAE decode
✅ Memory cleanup visible in logs
✅ System memory stable after tests

## Ready to Test

All fixes applied and documented. Running tests now...
