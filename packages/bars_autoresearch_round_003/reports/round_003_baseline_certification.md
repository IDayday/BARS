# Round 003 Baseline Certification

Primary question: Can we certify a public-quality GAS backbone using official/full-budget artifacts, then certify the BARS adapter against the official GAS evaluation loop?

Secondary question: If medium official artifacts remain unavailable, can certification shift to official-artifact OGBench tasks without weakening the research claim?

## Gate Result

- Baseline certification: PASS_BASELINE_CERTIFICATION
- Certified envs: antmaze-giant-stitch-v0, antmaze-large-explore-v0, scene-play-v0

## Rows

| env | artifact | score pp | lower bound pp | protocol match | status | reason |
| --- | --- | ---: | ---: | --- | --- | --- |
| antmaze-giant-stitch-v0 | OFFICIAL_FULL_BUDGET | 92.0 | 81.1 | True | PASS_BASELINE_CERTIFICATION | official full-budget artifact score meets public lower bound under matched protocol |
| antmaze-large-explore-v0 | OFFICIAL_FULL_BUDGET | 96.8 | 88.2 | True | PASS_BASELINE_CERTIFICATION | official full-budget artifact score meets public lower bound under matched protocol |
| scene-play-v0 | OFFICIAL_FULL_BUDGET | 79.60000000000001 | 57.599999999999994 | True | PASS_BASELINE_CERTIFICATION | official full-budget artifact score meets public lower bound under matched protocol |

## Interpretation Boundary

- PASS rows are certified baseline diagnostics only; they do not by themselves certify BARS.
- No p_bridge, integrated BARS-v3, or failure-taxonomy causal interpretation is unlocked until adapter certification also passes.
