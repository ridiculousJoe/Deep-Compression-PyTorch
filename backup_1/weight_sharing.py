import torch

from backup_1.quantization import do_weight_sharing

use_cuda = torch.cuda.is_available()


# Define the model
model = torch.load(f=F, map_location="model_after_retraining.ptmodel", pickle_module=torch.nn.Module)
print('accuracy before weight sharing')
# util.test(model, use_cuda)

# Weight sharing
do_weight_sharing(model)
print('accuacy after weight sharing')
# util.test(model, use_cuda)

