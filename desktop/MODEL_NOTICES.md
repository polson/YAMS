# Model notices

The Linux desktop downloads its model weights on first use. The release packages include the inference engine, not these weights.

## Current models

- **BS-RoFormer SW, six stems:** the default model for six-track processing and the first stage of eight-track processing. YAMS downloads a pinned copy from [FractalSurfer](https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem). Its model page says the original author did not declare a license and that the mirror grants no additional rights.
- **DrumSep:** the second model used for eight-track processing. YAMS downloads it from [lainlives](https://huggingface.co/lainlives/audio-separator-models), whose repository has no model card or license statement. The original author’s referenced repository is no longer available. A [separate mirror](https://huggingface.co/Sucial/MSST-WebUI/blob/main/All_Models/multi_stem_models/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt) labels a copy CC BY-NC-SA 4.0, but that is not confirmation from the model author.

The MIT license included with [the inference code](engine/inference/LICENSE) covers that code, not model weights. This repository has no top-level `LICENSE` file, so YAMS source reuse terms have not been stated.

Before a public release that downloads these weights, confirm their terms with the rights holders or choose models with clear terms. Separately, choose a YAMS code license if you want to invite source reuse.
