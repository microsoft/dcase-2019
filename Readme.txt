train_branches_20190711_3_one_part_one_model_LearningRateChangeSlower.py

load_dataset_20190711_1_one_part_one_model.py


Make DCASE Submission Files:

The only difference is they use different coarse classification model 

Make DCASE Submission Hierarchical-20190715-1-OneModel-3-CNNVGGModel1  use 
COARSE_CHECKPOINT = '/dcase/models/20190610_170308_coarse=0.787_fine=0.646.ckpt'

Get 
 * Micro AUPRC:           0.6429200022744848
 * Micro F1-score (@0.5): 0.5163853028798411
 * Macro AUPRC:           0.41221486377413197


Make DCASE Submission Hierarchical-20190715-1-OneModel-3-CNNVGGModel2 use 
COARSE_CHECKPOINT = '/dcase/models/20190609_230306_coarse=0.769_fine=0.656.ckpt'

Get 
* Micro AUPRC:           0.6229723518111516
 * Micro F1-score (@0.5): 0.5240431795878312
 * Macro AUPRC:           0.3858323991035093