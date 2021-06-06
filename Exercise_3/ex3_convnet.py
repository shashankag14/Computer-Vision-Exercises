import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
'''Only for testing purpose'''
from torchsummary import summary


import matplotlib.pyplot as plt

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
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 30 # 20: By default, 50: For early stopping
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
print(hidden_size)

# Batch-Norm params
norm_layer = None
if(norm_layer!=None):
  print("Running Model has Batch-Norm")
else:
  print("Running Model does not have Batch-Norm")
  
# Dropout params
dropout = None #np.arange(start=0.1,stop=0.9,step=0.1) 
if(dropout!=None):
  print("Running Model has Dropout, p = ", dropout)
else:
  print("Running Model does not have dropout")

# Early Stopping params
earlystop = None
patience = 10             # Early Stop Patience
max_earlystop_acc = None
patience_counter = 0
if(earlystop!=None):
  print("Running Model has Early Stopping, patience = ", patience)
else:
  print("Running Model does not have Early Stopping")

# Lists for getting learning curves
train_acc_curve = []
val_acc_curve = []
loss_curve = []
#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
'''To implement data augmentation, uncomment the corresponding technique'''
data_aug_transforms = [
    #  transforms.RandomHorizontalFlip(), 
    #  transforms.RandomVerticalFlip(p=0.5),
    #  transforms.RandomAffine(degrees=75),
    #  transforms.RandomAffine(degrees=0,translate=(0.5,0.5)),
    #  transforms.RandomAffine(degrees=0,scale=(0.7,0.7)),
    #  transforms.RandomGrayscale(p=0.5),
    #  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #  transforms.Resize((50, 50)),
    #  transforms.RandomCrop((32, 32))      
]
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
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


#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None, dropout=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #'''Without Batch-Norm and Dropout'''
        if (norm_layer==None and dropout==None):
          self.conv_layer = torch.nn.Sequential(

            torch.nn.Conv2d(input_size, hidden_size[0], kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(hidden_size[3], hidden_size[4], kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU()
          )

        #'''Batch-Norm without dropout'''
        elif (norm_layer!=None and dropout==None):
          self.conv_layer = torch.nn.Sequential(

          torch.nn.Conv2d(input_size, hidden_size[0], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[0]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),

          torch.nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[1]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),

          torch.nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[2]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),

          torch.nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[3]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),

          torch.nn.Conv2d(hidden_size[3], hidden_size[4], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[4]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU()
        )

        #'''Batch-Norm with dropout'''
        elif (norm_layer!=None and dropout!=None):
          self.conv_layer = torch.nn.Sequential(

          torch.nn.Conv2d(input_size, hidden_size[0], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[0]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),
          torch.nn.Dropout(dropout),

          torch.nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[1]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),
          torch.nn.Dropout(dropout),

          torch.nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[2]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),
          torch.nn.Dropout(dropout),

          torch.nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[3]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),
          torch.nn.Dropout(dropout),

          torch.nn.Conv2d(hidden_size[3], hidden_size[4], kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(hidden_size[4]),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.ReLU(),
          torch.nn.Dropout(dropout),
        )

        self.fc_layer = torch.nn.Linear(hidden_size[5], num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.conv_layer(x)
        '''x.shape = 200x512x1x1. Flattening it, except for the batch size (200)'''
        x = torch.flatten(x, start_dim = 1)
        out = self.fc_layer(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model_sz = np.sum(np.fromiter((param.numel() for param in model.parameters() if param.requires_grad), dtype = np.uint32))
    if disp:
      print("\nModel Size : ", model_sz)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''Returns 3x3x3 weights of 128 channels from the first layer'''
    weight_first_conv_layer = (model.conv_layer[0].weight.data).cpu().numpy()
    fig = plt.figure()
    vis_weight = 0
    
    for weight_index in range(hidden_size[0]):
        ax = fig.add_subplot(8, 16, weight_index + 1)
        ax.set_xticks([]) # To remove the ticks from each subplot
        ax.set_yticks([])

        '''Fetch 3x3x3 weights one by one for each channel'''
        weight_img = weight_first_conv_layer[vis_weight]
        low, high = np.min(weight_img), np.max(weight_img)
        norm_weight_img = (weight_img - low) / (high - low)
        
        plt.imshow((255 * norm_weight_img).astype(np.uint8))
        
        vis_weight += 1
    plt.show()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return 0
#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer, dropout=dropout).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)
#summary(model)       #Kept only for test purpose
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
print("First layer weight visualization before training :")
VisualizeFilter(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

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
        loss_curve.append(loss)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

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
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validatation accuracy is: {} %'.format(100 * correct / total))
        current_val_acc = 100 * correct / total
        val_acc_curve.append(current_val_acc)
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################
        best_model = None
        if (earlystop == True) :
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if max_earlystop_acc is None :
              max_earlystop_acc = current_val_acc
              torch.save(model.state_dict(), 'best_model.ckpt')
              best_model = model

            #'''If current acc is greater than/equal to max acc, increment the patience counter'''
            elif max_earlystop_acc >= current_val_acc :
              patience_counter += 1
              print("{Early Stopping} Counter: ", patience_counter, " out of ", patience)
              if patience_counter >= patience :
                print("{Early stopping} Best model's Accuracy : ", max_earlystop_acc)
                break
                
            else :
              max_earlystop_acc = current_val_acc
              # Save the model checkpoint
              torch.save(model.state_dict(), 'best_model.ckpt')
              best_model = model
              patience_counter = 0

        '''Below code was used for testing performance wherein we used early stopping only to
         save the best model, but there was no patience used. Rather it runs for all the epochs.'''
        # if max_earlystop_acc is None :
        #   max_earlystop_acc = current_val_acc
        #   best_model = model

        # #'''If current accuracy is higher than max acc, this is the best model till now'''
        # elif max_earlystop_acc < current_val_acc :
        #   max_earlystop_acc = current_val_acc
        #   best_model = model
        #   print("{Early stopping} Best model found. Accuracy : ", max_earlystop_acc)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.train()

#''' Plotting learning curves after training the model'''
plt.plot(range(1,len(loss_curve)+1), loss_curve, label = "Learning Curve : Training Loss v/s Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Training Loss")
plt.show()

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

print('\nRunning the model on the training set ...')
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
if (earlystop == True) :
  model.load_state_dict(torch.load('best_model_with_batchnorm.ckpt'))
  model.eval()
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
print("\nFirst layer weight visualization after training :")
VisualizeFilter(model)
# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')


