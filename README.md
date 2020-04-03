# Knolwedge Tracing Models

*PyTorch implementations of various Knowledge Tracing models* 

## Pre-processed Dataset
* Download Link: https://bit.ly/2w7J3On
* Dataset format: log files are seperated by users. Once you download and unzip each tar.gz file, there's a folder `processed`, and there are 5 subdirectories named from `1` to `5` for cross validation. Each subdirectory has its own train/val/test separation, where test dataset is shared by all 5 separations. Separation ratio are given in the following table. 
For each user, `{user_id}.csv` contains two columns (with headers): tag(skill_id, or question_id, which is a single positive integer) and correctness (0 or 1). 

| Dataset          | Train:Val:Test | Link | 
|------------------|----------------|------|
| ASSISTments2009  |       56:14:30       | https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010 |
| ASSISTments2012  |     6:2:2                 | https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect |
| ASSISTments2015  |       56:14:30       | https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data |
| ASSISTmentsChall |       6:2:2          | https://sites.google.com/view/assistmentsdatamining | 
| STATICS          |       56:14:30       | https://pslcdatashop.web.cmu.edu/Project?id=48 |
| Junyi Academy    |       6:2:2           | https://pslcdatashop.web.cmu.edu/Project?id=244 | 
| KDDCup2010       |       6:2:2           | https://pslcdatashop.web.cmu.edu/KDDCup/ |
| EdNet-KT1        |       6:2:2           | https://github.com/riiid/ednet |

* For ASSISTments2009, ASSISTments2015, and STATICS data we use the same data (with different format) that used in [this](https://github.com/jennyzhang0215/DKVMN) DKVMN implementation. Also, two files with same name (same `user_id`) but belong to different subdirectories may not coincides in this case, which actually does not important when we train & test models. 

## DKT (Deep Knowledge Tracing)
* Paper: https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
* Performances: 

| Dataset          | ACC | AUC | Hyper Parameters |
|------------------|-----|-----|------------------|
| ASSISTments2009  |     |     |                  |
| ASSISTments2015  |     |     |                  |
| ASSISTments2012  |     |     |                  |
| ASSISTmentsChall |     |     |                  |
| STATICS          |     |     |                  |
| Junyi Academy    |     |     |                  |
| KDDCup2010       |     |     |                  |
| EdNet-KT1        |     |     |                  |

## DKVMN (Dynamic Key-Value Memory Network) (TODO)
* Paper: http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p765.pdf
* Performances: 

## SAKT (Self-Attentive Knowledge Tracing) (TODO)
* Paper: https://files.eric.ed.gov/fulltext/ED599186.pdf
* Performances: 

## NPA (Neural Padagogical Agency) (TODO)
* Paper: https://arxiv.org/abs/1906.10910
* Performances: 

## SAINT (Separated self-AttentIve kNowledge Tracing) (TODO)
* Paper: https://arxiv.org/pdf/2002.07033.pdf
* Performances
