# Pytroch Cookbook(常用代码端整理)

```
token
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  

git remote add origin https://github.com/guanghaojiang/learn-note.git
git branch -M master
git push -u origin master
```



# 1.基础配置

## 检查PyTorch 版本

```python
torch.__version__               # PyTorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
```

 ## 固定随机种子

```python
torch.manual_seed(0)
torch.cuda.manual_seed(0)
```

## 指定程序运行在GPU卡上

在命令行指定环境变量

```python
CUDA_VISIBLE_DEVICES = 0，1 python train.py
```

或在代码中指定

```python
os.environ['CUDA_VIDIBLE_DEVICES'] ='0,1'
```

## 判断是否有CUDA支持

```python
torch.cuda.is_available()
```

## 清除GPU存储

有时Control-C中止运行后GPU存储没有及时释放，需要手动清空。在PyTorch内部可以

```python
torch.cuda.empty_cache()
```

或者直接重置没有被清空的GPU

```python
nvidia-smi --gpu-reset -i [gpu_id]
```

# 2. 模型

## 2.1 模型权重初始化

注意model.modules()和model.children()的区别：model.modules()会迭代地遍历模型的所有子层，而model.children()只会遍历模型下的一层

```python
# Common practise for initialization

for layer in model.modules():
    if isinstace(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',nonlinearity='relu')
        
        if layer.nn.init.constant_(layer.bias, val=0.0)
     elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
     
     elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)   
      
```

## 2.2 部分层使用与训练模型

​	注意如果模型保存的模型是torch.nn.DataParallel,则当前的模型也需要是torch.nn.DataParallel.

​	torch.nn.DataParallel(model).module == model

```python
model.load_state_dict(torch.load('model, pth'), strict =False)
```

## 2.3 将在GPU保存的模型加载到CPU

```python
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
```

## 2.4 微调全连接层

```python
model =torchvidion.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100) # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

##  2.5 以较大学习率微调全连接层，较小学习率微调卷积层

```python
model = torchvidion.models.resnet18(prerained =True)
finetuned_paramters = list(map(id, model.fc.parameters()))
conv_parameters = ( p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params': conv_parameters, 'lr':1e-3},
              {'params': model.fc.parameters()}]

optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

#  3. 模型训练

## 3.1 常用训练和验证数据预处理

​	其中ToTensor操作会将PIL.Image 或形状为H X W X D,数值范围为[0,255]的np.ndarray转换为形状为D X H  X W,数值范围为[0.0，1，0]的torch.Tensor

```python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])
 val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
]
```

## 3.2 训练的基本代码框架

```python
for t in epoch(80):
	for images, labels in tqdm(train_loader, desc = 'Epoch %3d' % (t + 1)):
		images, labels = images.cuda(), labels.cuda()
		scores = model(images)
		loss = loss_function(dcores, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
```

## 3.3 标记平滑（label smoothing）

```python
for images, labels in train_loader:
	images,labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes
    smoothed_labels = torch.full(size=(N, C),fill_value=0.1 / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1),value=0.9)
    score = model(images)
    log_prob = torch.nn.functionall.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 3.4 Mixup

```python
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

    # Mixup loss.    
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, labels) 
            + (1 - lambda_) * loss_function(scores, labels[index]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 3.5 梯度裁剪（gradient clipping）

```python
torch.nn.utils.clip_grad_norm_(model.paramters(), max_norm=20)
```

## 3.6 计算Softmax输出的准确率

```python
score = model(images)
prediction = torch.argmax(score, dim=1)
num_correct = torch.sum(prediction == labels).item()
accuruacy = num_correct / labels.size(0)
```

## 3.7 保存与加载断点

注意为了能够恢复训练，我们需要同时保存模型和优化器的状态，以及当前的训练轮数

```python
# Save checkpoint
id_best = current_acc > best_acc 
best_acc = max(best_acc, current_acc)
checkpoint = {
    'best_acc': best_acc,
    'epoch': t + 1，
    'model': model.state.dict(),
    'optimizer': optimizer.state_dict(),
}

model_path = os.path.join('model','checkpoint.pth.tar')
torch.save(checkpoint,model_path)
if is_best:
    shutil.copy('checkpoint.pth.tar', model_path)
    
    
# Load checkpoint

if resume:
    model+path = os.path.join('model','checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
   	print('Load checkpoint at epoch %d.' % start_epoch)
```

## 3.8 计算准确率、查准率（precision）、查全率（recall) 

```python 
# data['label'] and data['prediction'] are groundtruth label and prediction 
# for each image, respectively.
accuracy = np.mean(data['label'] == data['prediction']) * 100

# Compute recision and recall for each class.
for c in range(len(num_classes)):
    tp = np.dot((data['label'] == c).astype(int),
                (data['prediction'] == c).astype(int))
    tp_fp = np.sum(data['prediction'] == c)
    tp_fn = np.sum(data['label'] == c)
    precision = tp / tp_fp * 100
    recall = tp / tp_fn * 100
```

# 4. 模型测试

## 计算每个类别的查准率（precision）、查全率(recall) 、F1分数和总体指标

```python
    def comfusion_matrix(preds_list, lables_list, CLASS_NUM):

        confusion = np.zeros((CLASS_NUMS, CLASS_NUMS), dtype=np.int)# 统计结果

        for i in range(len(label_list)):
            if label_list[i] == preds_list[i]:
                confusion[lables_list[i]][lables_list[i]] +=1

            else:
                confusion[labels_list[i]][preds_lsit[i]] +=1
         return confusion
	


    all_label = []
    all_prediction = []
    for images, labels in tqdm.tqdm(data_loader):
         # Data.
         images, labels = images.cuda(), labels.cuda()
         # Forward pass.
         score = model(images)
         # Save label and predictions.
         prediction = torch.argmax(score, dim=1)
         all_label.append(labels.cpu().numpy())
         all_prediction.append(prediction.cpu().numpy())
    	
 # Compute RP and confusion matrix
 	all_pred = np.concatenate(all_pred, axis=0)
    all_label= np.concatenate(all_label, axis=0)
    matrix = confusion_matrix(all_pred, all_label, CLASS_NUMS=3)
 	confusion_file_path = 'ca_m600_confusion_matrix_swin_ce_1108.csv'
    confusion_file = open(confusion_file_path, 'w+')
    np.savetxt(confusion_file, matrix, fmt='%d', delimiter=',')
    print(confusion_file_path)
    confusion_file.close()

    epsilon = 0
    res = {}
    # 计算每一类的查准率（precision）、查全率(recall) 、F1分数
    for ix in range(3):
        a = sum(matrix[:,ix])
        print(a)
        p = matrix[ix][ix]/(sum(matrix[ix,:])+epsilon)
        r = matrix[ix][ix]/(sum(matrix[:,ix])+epsilon)
        f1 = 2 * ((p * r) / (p + r + epsilon))
        res[ix]={'r':r,'p':p,'f1':f1}

        for ix in range(3):
            print('Class ',ix,res[ix])
```

