# Modularized Zero-shot VQA with Pre-trained Models

This includes an original implementation of "[Modularized Zero-shot VQA with Pre-trained Models][paper]" by Rui Cao, Jing Jiang.

<p align="center">
  <img src="Mod-Zero-VQA-arch.PNG" width="80%" height="80%">
</p>

This code provides:
- Codes for pre-processing VQA datasets.
- Instructions and codes to run the models and get numbers reported in main experiment of the paper.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{cao-jiang-2023-modularized,
    title = "Modularized Zero-shot {VQA} with Pre-trained Models",
    author = "Cao, Rui  and
      Jiang, Jing",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    pages = "58--76"
}
```

### Announcements
05/2023: Our paper is accepted by ACL, Findings 2023. 

## Content
1. [Installation](#installation)
2. [Dataset Prep-processing](#dataset-pre-processing)
    * [Step 1: Downloading Datasets](#step-1-downloading-datasets) (Section 4.1 of the paper) 
    * [Step 2: Layout Generation](#step-2-layout-generation) (Appendix F)
    * [Step 3: Object Detection](#step-3-object-detection) (Section 3.3 of the paper)
    * [Step 4: Masked Statement Generation for Open-ended Questions](#step-4-masked-statement-generation-for-open-ended-questions) (Section 3.3 of the paper)
    * [Step 5: Candidate Answers Generation](#step-4-candidate-answers-generation) (Appendix G) 
3. [Experiments](#experiments) (Section 4 of the paper)
    * [Testing on GQA](#testing-on-GQA)
    * [Testing on VQA](#testing-on-VQA)

## Installation
The code is tested with python 3.8. To run the code, you should install the package of transformers provided by Huggingface. The code is implemented with the CUDA of 11.2 (you can also implement with other compatible versions) and takes one Tesla V 100 GPU card (with 32G dedicated memory) for experiments. To run the experiments, we rely on the following packages: PyTorch (version 1.9.0), HuggingFace Transformers (version 4.19.2), Stanza (version 1.4.0) and NLTK (version 3.2.5). We also used pre-trained models as modules in our modularized networks. The following pre-trained models are considered: [OWL][owl_code], [MDETR][mdetr_code] and [CLIP][clip_code]. 

###
## Dataset Prep-processing

### Step 1: Downloading Datasets
To leverage our code, you need to download the testing data. We tested on two VQA benchmarks: [VQAv2][vqav2] and [GQA][gqa]. Both datasets are publicly available. Then you can apply our pre-processing code over the downloaded datasets.

### Step 2: Layout Generation
We next prompt frozen PT-VLMs with questions and cleaned images to obtain Pro-Cap. You can generate Pro-Cap with our code at [codes/Pro-Cap-Generation.ipynb](codes/Pro-Cap-Generation.ipynb). Or you can alternatively use generated Pro-Cap shared in [codes/Ask-Captions](codes/Ask-Captions).

### Step 3: Object Detection

### Step 4: Masked Statement Generation for Open-ended Questions

### Step 5: Candidate Answers Generation

## Experiments 
To be done

### Testing on GQA
Before uploading codes, we re-run the codes. Because of the updating of the versions of transformers package, we observe a small variance compared with the reported performance in the paper. We conclude both the reported results and the re-implemented result in the Figure above. There is no significant difference according to p-value. We share both the re-implemented logger files and the logger files for the reported performance in [codes/logger](codes/logger) and [codes/reported](codes/reporte).

### Testing on VQA
To obtain our reported performance, please run the script in [codes/src](codes/src):
```bash
bash run.sh
```

[paper]: https://aclanthology.org/2023.findings-acl.5.pdf
[owl_code]: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
[mdetr_code]: https://github.com/ashkamath/mdetr
[clip_code]: https://github.com/openai/CLIP
[vqav2]: https://visualqa.org/
[gqa]: https://cs.stanford.edu/people/dorarad/gqa/about.html
