from function import *
import gc


# ======================================================================================================================
# Data preprocessing
train_loader = create_data_loader(CFG.train_data_path, CFG.tokenizer, CFG.batch_size, is_train=True)
valid_loader = create_data_loader(CFG.valid_data_path, CFG.tokenizer, CFG.batch_size, is_train=False)
test_loader = create_data_loader(CFG.test_data_path, CFG.tokenizer, CFG.batch_size, is_train=False)
        
# ======================================================================================================================
# Task 
is_train = True  # Whether train or not

if is_train == True:
    model = CustomModel().to(CFG.device) # Build model object.
    model, acc_train, _ = training(model, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)  # Train model based on the training set (you should fine-tune your model based on validation set.)
    
else:
    model = load_model(CFG.model_save_path, CFG.device)  # Build model object.
    acc_train = None
_, acc_test = testing(model, CFG.test_data_path, CFG.tokenizer, CFG.batch_size, CFG.device)  # Test model based on the test set.
            
gc.collect()  # Some code to free memory if necessary.


# ======================================================================================================================
## Print out your results with following format:
print('Tsak:{},{}'.format(acc_train, acc_test))
