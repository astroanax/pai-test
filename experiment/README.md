execution pullback pilot, project-i workspace

layout, all under execution-pullback/
- experiment/config.json holds every default consumed by the code
- experiment/core.py holds student, integration, finite differences, vjp probes, six losses, scale matching
- experiment/hri_adapter.py holds strict checkpoint load and complete history replay
- experiment/pilot.py holds normalizer, collect, metrics, train, evaluate
- experiment/diagnose.py holds held out sensitivity diagnostic
- experiment/analyze.py holds paired summary with exact discordant test and joint scene bootstrap
- experiment/sanity.py holds algebra and synthetic backward checks
- experiment/preflight.py checks imports, cuda, assets, exits nonzero on missing items
- experiment/lock_protocol.py records hashes and scene ids before final outcomes are examined
- experiment/run_comparison.sh runs matched comparison

library notes from source reads and deepwiki
- hri flow_matching example flow_pusht.py: ConditionalFlowMatcher sigma 0, x0 gaussian randn, xt target loss against ut, ConditionalUnet1D input dim 2 global cond 514, pred horizon 16 action horizon 8 obs horizon 1, resnet18 with fc replaced by identity plus replace_bn_with_gn, ema power 0.75, adamw lr 1e-4 wd 1e-6, cosine schedule 500 warmup
- test branch in the same example uses uniform rand source with one euler step, so the pilot treats 16 step gaussian as an explicit adaptation and gates native fast versus slow before compression
- pusht.py: PushTEnv sim 100 hz control 10 hz pd kp 100 kv 20, action space 0 to 512, observation 5 vector, image env renders 96 rgb with agent pos, dataset normalizes action and agent pos from zarr min max to minus 1 to 1, success is goal coverage above 0.95, legacy flag changes set state order then steps physics once
- unet.py: ConditionalUnet1D takes sample BxTxC timestep global cond, sinusoidal time embedding plus film conditioning, down dims 256 512 1024 kernel 5, batch independence required for summed output vjp
- resnet.py: get_resnet wraps torchvision resnet, fc set to identity so resnet18 output is 512 features, replace_bn_with_gn swaps every BatchNorm2d for GroupNorm with 16 features per group
- torchcfm via deepwiki: sample_location_and_conditional_flow draws t uniform, eps gaussian, xt mu_t plus sigma eps, ut x1 minus x0 for sigma 0, models module holds mlp and unet, utils holds sampling and plotting helpers
- requirements.txt: torch torchvision zarr diffusers gym pygame pymunk shapely opencv scikit-image scikit-video gdown matplotlib ipython torchcfm torchdyn torchsde torchdiffeq

setup, run relative to execution-pullback/
- place upstream checkout at external/flow_matching, record commit in assets/upstream_commit.txt
- place teacher checkpoint at assets/flow_pusht.pth and demonstration archive under assets
- build assets/normalizer.npz with pilot.py normalizer before training
- weights_only tensor checkpoint loading, strict key and shape checks

audit record, static only, nothing executed
- label fix: collect now saves warm student midpoint ztilde, metrics sets midpoint ztilde, target_mid y0 half map, target_end y1 second half from ztilde, teacher_end at second half from y0, suffix is half map 0.5 to 1, prefixes from at and y1
- appendix randint bound 140 corrected to 0, 2
- teacher_half_maps parameterized by step count instead of hardcoded 8 plus 8
- metrics path dead code removed, suffix closure uses configured step count
- eval noise slot reshape fixed to keep batch dim
- check_batch_independence simplified to avoid unused tensor
- hri_adapter imports moved inside functions so core and sanity import without the simulator stack
- pilot inserts repository model path at build_adapter time
- sanity torch check imports experiment.core for the project-i layout
- preflight adds torchcfm to the required module list
- scalar mode preserves per context trace while removing direction, matching appendix a.6
- endpoint penalty uses full student endpoint error through jacobian_start, pullback early term uses probe projection of mid error, late term uses jacobian_end, matching equations 5 and 8
- metric scales use global positive median over training rows, minimum 8 nonzero, never per context division
- replay restores scene from seed and replays full physical prefix, validates signature, holds terminal feature, duplicate finite difference branch recorded
