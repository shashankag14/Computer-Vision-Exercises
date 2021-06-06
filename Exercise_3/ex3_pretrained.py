import torch
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
layer_config= [512, 256]
num_classes = 10
num_epochs = 30
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg=0#0.001
num_training= 49000
num_validation =1000
fine_tune = False
pretrained = True
print("fine_tune = ", fine_tune, " and pretrained = ", pretrained)

# Based on pretrained and fine_tune flags, we are creating a category of models for 4a and 4b(fine tune and baseline)
if (pretrained == True and fine_tune == True):
    model_flag = 'part_4a'
elif (pretrained == True and fine_tune == False):
    model_flag = 'part_4b_1'
elif (pretrained == False and fine_tune == False):
    model_flag = 'part_4b_2'
print('Running model for {}'.format(model_flag))

earlystop = True
patience = 10
max_earlystop_acc = None
patience_counter = 0
if(earlystop!=None):
  print("Running Model has Early Stopping, patience = ", patience)
else:
  print("Running Model does not have Early Stopping")
  
# Lists used for learning curves  
loss_curve = []
val_acc_curve = []
train_acc_curve = []

data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]#, transforms.RandomGrayscale(p=0.05)]
#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
# Q1,
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=norm_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        self.vgg = models.vgg11_bn(pretrained=pretrained)
        del self.vgg.avgpool
        del self.vgg.classifier
        
        self.vgg.classifier = nn.Sequential(nn.Linear(layer_config[0], layer_config[1]),
                                       nn.BatchNorm1d(layer_config[1]),
                                       nn.ReLU(),
                                       nn.Linear(layer_config[1], num_classes)
                                       )
        
        
        set_parameter_requires_grad(self.vgg.features ,fine_tune)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        temp1=self.vgg.features(x)
        temp1 = temp1.squeeze()
        out=self.vgg.classifier(temp1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

# Initialize the model for this run
model= VggModel(num_classes, fine_tune, pretrained)

# Print the model we just instantiated
print(model)

#################################################################################
# TODO: Only select the required parameters to pass to the optimizer. No need to#
# update parameters which should be held fixed (conv layers).                   #
#################################################################################
print("Params to learn:")

params_to_update = []
# Part A
if (pretrained == True and fine_tune == True):
    params_to_update = model.vgg.classifier.parameters()

    for name,param in model.vgg.classifier.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    
# Part B.1 - Fine tune model without baseline
elif (pretrained == True and fine_tune == False):
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    
# Part B.2 - Baseline model which trains the model from scratch without 
# loading any pretrained weights
elif (pretrained == False and fine_tune == False):
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # To update train accuracy per epoch list
    current_train_acc = 100 * correct / total
    print('Training accuracy is: {} %'.format(100 * correct / total))
    train_acc_curve.append(current_train_acc)
    
    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #################################################################################
        # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
        # the model which has acheieved the best validation accuracy so-far.            #
        #################################################################################
        #best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        current_val_acc = 100 * correct / total
        val_acc_curve.append(current_val_acc)
        if (earlystop == True) :
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if max_earlystop_acc is None :
                max_earlystop_acc = current_val_acc
                torch.save(model.state_dict(), 'best_model_{}.ckpt'.format(model_flag))

            #'''If current acc is greater than/equal to max acc, increment the patience counter'''
            elif max_earlystop_acc >= current_val_acc :
              patience_counter += 1
              print("{Early Stopping} Counter: ", patience_counter, " out of ", patience)
              if patience_counter >= patience :
                print("{Early stopping} Best model's Accuracy : ", max_earlystop_acc)
                
                # Saves the latest model (the model trained after 10th patience level)
                torch.save(model.state_dict(), 'model_{}.ckpt'.format(model_flag))
                break
                
            else :
                max_earlystop_acc = current_val_acc
                # Saves the latest model (the model trained after 10th patience level)
                torch.save(model.state_dict(), 'best_model_{}.ckpt'.format(model_flag))
                patience_counter = 0

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print('Validataion accuracy is: {} %'.format(100 * correct / total))

#################################################################################
# TODO: Use the early stopping mechanism from previous question to load the     #
# weights from the best model so far and perform testing with this model.       #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model.load_state_dict(torch.load( 'best_model_{}.ckpt'.format(model_flag)))
model.eval()
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images for {} : {} %'.format(total, model_flag, 100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')

plt.plot(range(1,len(val_acc_curve)+1), val_acc_curve, label = "Val")
plt.plot(range(1,len(train_acc_curve)+1), train_acc_curve, label = "Train")
# find position of highest validation accuracy for early stopping
if(earlystop==True) :
  maxposs = val_acc_curve.index(max(val_acc_curve))+1
  plt.axvline(maxposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.title('Learning Curve : Accuracy v/s Epochs')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(range(1,len(loss_curve)+1), loss_curve, label = "Learning Curve : Training Loss v/s Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Training Loss")
plt.show()





