# Do VLMs look at the pedestrian? Zero-shot crossing intent on JAAD

A small, self-contained study of whether open vision-language models (VLMs) can predict
pedestrian crossing intent from a dashcam frame, and whether they reach their answer by
looking at the pedestrian or by relying on scene and position priors.

**Dataset:** [JAAD](https://github.com/ykotseruba/JAAD) (Rasouli, Kotseruba & Tsotsos, 2017).
The JAAD videos are not included in this repo. Download them from the official source.

## Setup

| | |
|---|---|
| Samples | 300 observation windows, 150 crossing / 150 not crossing, 174 pedestrians, 127 videos (`samples.csv`) |
| Protocol | Follows the JAAD/PIE benchmark (Kotseruba et al., WACV 2021). The observation ends 1, 1.5 or 2 s before the crossing event (for non-crossers, the end of the track). Behavioral pedestrians only, box height ≥ 50 px, not occluded. |
| Models | Qwen2.5-VL-7B-Instruct, LLaVA-OneVision-Qwen2-7B, SmolVLM-Instruct (2B). All fp16, zero-shot, on Kaggle 2×T4 |
| Scoring | One forward pass, P(Yes) vs. P(No) from next-token logits, so no answer parsing is needed. `mass_yes_no` records how much probability the model put on Yes/No at all |
| Statistics | AUC and accuracy with 95% CIs from a **cluster bootstrap over pedestrians** (one pedestrian can contribute up to 3 windows) |

### Conditions
- **Prompts:** three paraphrases of the same question (p0, p1, p2)
- **Distractors:** two pedestrian crops from other videos, scaled to the target's height and pasted where they don't overlap it
- **Clip:** four frames (0.7 s) instead of one (Qwen only)
- **Counterfactual ablations** (Qwen, prompt p2), each image paired with the original:
  - `mask`: target pedestrian inpainted out, red box kept
  - `boxshift`: box moved to a pedestrian-free spot at the same height
  - `blank`: grey frame containing only the box
  - `near` / `far`: one distractor on the target's ground line, close (< 0.2 W) or far (> 0.4 W)
  - `n4`: four distractors

### Baselines
Logistic regression with grouped 5-fold CV by pedestrian:
- **(a)** five box-geometry features (height, position, horizontal motion, time-to-event)
- **(b)** those five plus JAAD's annotated cues (looking, walking, crosswalk, signal, intersection, group size)

These are *trained*, while the VLMs are zero-shot, so they are reference points rather than a head-to-head comparison.

## Results
See `RESULTS.md` and `figures/`.

## In progress (v2)
These are set up and tested on a few samples, but not yet run in full:
- A larger, independent sample set: 433 pedestrians, one window each, from JAAD-all (`make_samples_v2.py`)
- **Close-up crops**, to separate a resolution problem from a reasoning problem: crop alone, scene + crop, and scene + crop with the pedestrian inpainted out
- **Vehicle speed** in the prompt, and **step-by-step reasoning** with a scored final answer
- **Linear probes** on Qwen2.5-VL hidden states (`probe.py`): is crossing intent decodable from internal features even though the model's answer doesn't track it?
- Scaling across Qwen2.5-VL 3B / 7B / 32B and Qwen3-VL-8B

Code: `kernels/vlm/run_v2.py`, `analyze_v2.py`, `probe.py`.

## Reproduce
```
python make_samples.py            # needs JAAD annotations (git clone the JAAD repo here)
# kernels/frames: Kaggle CPU kernel that downloads JAAD clips and extracts the frames
# kernels/vlm/run_vlm.py <model>    # run on a GPU (transformers 4.57; 5.0 hangs while loading on Kaggle)
python analyze.py output/preds    # accuracy / F1 / AUC + distractor effect
python cues.py output/preds       # baselines + cue alignment
python figures.py output/preds
```

## Limitations
- 300 samples. CIs are wide, and differences smaller than ~0.08 AUC are not resolvable.
- Zero-shot, single-turn prompting only. No chain-of-thought, no vehicle speed (which helped in arXiv:2507.04141), no fine-tuning.
- Distractors are pasted crops, without lighting or perspective matching. Some `n4` placements are physically implausible.
- Non-crossers' "event" is the end of their track, so box geometry may partly encode how samples were selected rather than behavior (this affects baseline a).
- The red box is a visual prompt. Models may handle box overlays differently.

## References
- Rasouli, Kotseruba, Tsotsos. *Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior.* ICCV Workshops, 2017.
- Kotseruba, Rasouli, Tsotsos. *Benchmark for Evaluating Pedestrian Action Prediction.* WACV, 2021.
- Liu, Kotseruba, Tsotsos. *Disentangling Pictorial Cue Understanding from Language Bias in VLMs via Depth Ordering Task.* ECCV, 2026.
- Rosenfeld, Zemel, Tsotsos. *The Elephant in the Room.* arXiv, 2018.
- PedestrianQA: a benchmark for vision-language models on pedestrian behavior. ICRA, 2026.
- Uziel et al. *Optimizing Vision-Language Model for Road Crossing Intention Estimation.* WACV, 2025.
- *Pedestrian Intention Prediction via Vision-Language Foundation Models.* arXiv:2507.04141, 2025.
