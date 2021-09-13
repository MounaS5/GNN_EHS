from sklearn.metrics import roc_auc_score
def Evaluator_CR(pred_dict,task_type):
  y_true = pred_dict['y_true']
  y_pred = pred_dict['y_pred']
  if task_type == 'Classification':
    rocauc_list = [] # Can work for multiclass#AUC is only defined when there is at least one positive data.
    for i in range(y_true.shape[1]):
      if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
        rocauc_list.append(roc_auc_score(y_true[:,i], y_pred[:,i]))
    # for i in range(y_true.shape[0]):
    #     if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
    #       rocauc_list.append(roc_auc_score(y_true[i], y_pred[i]))
    if len(rocauc_list) == 0: raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
    output_dict  = {"ROC_AUC": sum(rocauc_list)/len(rocauc_list)}
  elif task_type == 'Regression':
    r2  = sklearn.metrics.r2_score(y_true, y_pred)
    rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
    output_dict  = {"R2": r2, "RMSE": rmse}
  return output_dict 

def criterion(prediction, real,task_type):
    if task_type == 'Regression':
      loss = torch.nn.MSELoss()
    elif task_type == 'Classification':
      loss = torch.nn.BCEWithLogitsLoss()#You can give weights as an input
      #loss = torch.nn.CrossEntropyLoss()
    else:
      ValueError(f'Invalid task_type {task_type}')
    output_loss = loss(prediction, real) # Order is correct
    #https://www.programcreek.com/python/example/107675/torch.nn.BCELoss
    return output_loss

# from demo_module import gnnModels_S
# from demo_module import gnnModels_C
def initialiseModel(model_typ,par_dict):
    #model = GCN(hidden_channels=64,num_layers=4)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_typ == 'GNN_C':
      #model = GNN(par_dict["num_layer"], par_dict["drop_ratio"], par_dict["conv_dim"],par_dict["gnn_type"] ,par_dict["JK"],par_dict["graph_pooling"]) # Here goes your new model and its variables
      model    = GNN(par_dict["num_layer"], par_dict["drop_ratio"], par_dict["conv_dim"],par_dict["gnn_type"] ,par_dict["JK"],par_dict["graph_pooling"]).to(device) # compare
    elif model_typ == 'GNN_S':
      model = Net(par_dict['conv_dim'],par_dict['node_dim'],par_dict['num_layer']).to(device)
  
    return model,device

def drawAndSavePlot (new_dict,task_type):
    plt.figure(figsize=(8, 5))
    plt.plot(new_dict['Train Loss'], label='Training', alpha=0.5)
    plt.plot(new_dict['Test Loss'], label='Testing', alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.title(name_dict['plt_title'],loc='right')
    plt.legend()
    local_download_path = os.path.expanduser('drive/My Drive/GNNs/'+name_dict['fold_Name'])#data_T/Carcinogenicity/'+name2
    plot_filepath = os.path.join(local_download_path, name_dict['plt_title']+'.png')
    plt.savefig(plot_filepath)

#####################################################
#### Functions that require task type################
#####################################################

import collections
def func_task_type(task_type):
  if task_type == 'Classification':
    terms_lst = ['Epoch','Train Loss','Train Loss_1','Test Loss','Valid Loss','Train Acc','Test Acc','Valid Acc','Train ROC_AUC','Test ROC_AUC','Valid ROC_AUC']
  if task_type == 'Regression':
    terms_lst = ['Epoch','Train Loss','Train Loss_1','Test Loss','Valid Loss','Train Acc','Test Acc','Valid Acc','Train RMSE','Test RMSE','Valid RMSE','Train R2','Test R2','Valid R2']
  new_dict = collections.OrderedDict()
  for variable in terms_lst: new_dict[variable] = []
  return new_dict

def update_dict(task_type,epoch,loss,train_acc=None,test_acc=None,valid_acc=None):
  temp = collections.OrderedDict()
  # Loss
  temp.update(epoch = epoch, train_loss = loss, train_loss_1 = train_acc['Loss'] , test_loss = test_acc['Loss'])
  if valid_acc:temp.update(valid_loss = valid_acc['Loss'])
  else:temp.update(valid_loss = 0)
  # Accuracy
  temp.update(train_acc = train_acc['Acc'],test_acc = test_acc['Acc'])
  if valid_acc:temp.update(valid_acc = valid_acc['Acc'])
  else:temp.update(valid_acc = 0)

  if task_type == 'Classification':
    # ROC_AUC
    temp.update(train_roc = train_acc['ROC_AUC'],test_roc = test_acc['ROC_AUC'])
    if valid_acc:temp.update(valid_roc = valid_acc['ROC_AUC'])
    else:temp.update(valid_roc = 0)

  if task_type == 'Regression':
    # RMSE
    temp.update(train_RMSE = train_acc['RMSE'],train_acc = train_acc['Acc'],test_RMSE = test_acc['RMSE'])
    if valid_acc:temp.update(valid_RMSE = valid_acc['RMSE'])
    else:temp.update(valid_RMSE = 0)
    # R2
    temp.update(train_R2 = train_acc['R2'],test_R2 = test_acc['R2'])
    if valid_acc: temp.update(valid_R2 = valid_acc['R2'])
    else:temp.update(valid_R2 = 0)
  return temp

def best_epoch_model(model_lst,new_dict,task_type,valid_acc=None):
  if task_type == 'Classification':
    if valid_acc:best_val_epoch = np.argmax(np.array(new_dict['Valid ROC_AUC']))
    else:best_val_epoch = np.argmax(np.array(new_dict['Test ROC_AUC']))
  if task_type == 'Regression':
    if valid_acc:best_val_epoch = np.argmin(np.array(new_dict['Valid Loss']))
    else:best_val_epoch = np.argmax(np.array(new_dict['Test Loss']))

  return model_lst[best_val_epoch],best_val_epoch

#################################################################
#################### Training and Evaluation Functions ##########\
#################################################################
from sklearn.metrics import roc_auc_score
def train(optimizer,model,device,epoch,loader):
    model.train()
# Instead of CPU think about detaching.
    # if epoch == 5:
    #     for param_group in optimizer.param_groups:
    #         print(param_group)
    #         param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    #loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # 2 ways: either you detach output or you attach 
        # real to device after dtype conversion
        #data.x, data.edge_index, data.batch
        if model_typ == 'GNN_C':
            output = model(data)
        elif model_typ == 'GNN_S':
            output = model(data.x, data.edge_index, data.batch)#.detach().cpu().flatten()
        #print(output)
        # out_pred = output.to(torch.float32)
        #print(out_pred)
        #real = data.y.type(torch.float32).reshape(out_pred.shape)
        #print(real)
        #loss = F.nll_loss(output, real)
        #break
        out_pred=output.to(torch.float32)
        real = data.y.type(torch.float32).reshape(out_pred.shape)
        loss = criterion(out_pred, real,task_type)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    n = float(sum([data.num_graphs for data in loader])) 
    return loss_all / n


def test(model,device,loader,evaluator,task_type):
    model.eval()
    y_true = []
    y_pred = []
    pred_dict={}
    correct = 0
    loss_sum=0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            if model_typ == 'GNN_C':
                output = model(data)
            elif model_typ == 'GNN_S':
                output = model(data.x, data.edge_index, data.batch)#.detach().cpu().flatten()
          
        # y_true.append(data.y.view(output.shape).detach().cpu())
        # y_pred.append(output.detach().cpu())
        # #if task_type == 'Classification': pred = output.max(dim=1)[1]
        # prediction=output.to(torch.float32)
        # real = data.y.type(torch.float32).reshape(prediction.shape)
        # correct += output.eq(data.y).sum().item()
        # #correct += pred.eq(data.y).sum().item()
        # # Loss 
        # loss =criterion(prediction, real,task_type)
    #print(output)
            ### ROC_AUC I'll need probability scores
        y_true.append(data.y.view(output.shape).detach().cpu())
        m = nn.Sigmoid()
        y_pred.append(output.detach().cpu())
    
        ## Loss calculation. I don't need any thing
        prediction=output.to(torch.float32)
        #real = data.y.type(torch.float32).reshape(prediction.shape)
        real = data.y.type(torch.float32).reshape(output.shape)
        loss =criterion(prediction, real,task_type)
        
        ## Pred. I need in 0/1
        output_1 = m(output).to(torch.float32)
        pred=output_1.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    
    
    
        
        loss_sum += loss.item() * data.num_graphs
        
        
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if task_type == 'Regression':
      y_pred = y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
      y_true = y_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).ravel()
    
        
    pred_dict  = {"y_true": y_true, "y_pred": y_pred}
    prediction = evaluator(pred_dict,task_type)
    #print(prediction)
    n = float(sum([batch.num_graphs for batch in loader])) 
    prediction['Acc'] = correct / n
    prediction['Loss'] = loss_sum/n
    #print(prediction['Acc'])
    return prediction, y_pred,y_true

#from torchtools import EarlyStopping   
def results_GNN(task_type,train_loader,test_loader,valid_loader,Evaluator_CR,lst_hyp,model_typ):
    n_epochs =300
    lr = lst_hyp[2]
    if model_typ == 'GNN_C':
      par_dict = { "num_layer": lst_hyp[1], "drop_ratio":0.5, "conv_dim": lst_hyp[0], "gnn_type" : 'NNConv',"JK": "last"                     ,"graph_pooling": "add"}
    elif model_typ == 'GNN_S':
      #print(lst_hyp)
      par_dict = { "num_layer": lst_hyp[1], "conv_dim": lst_hyp[0], "node_dim": lst_hyp[3]}
    model,device = initialiseModel(model_typ,par_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.7,
                                                            patience=5,
                                                            min_lr=0.0000005,
                                                            verbose=True)
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    Y_pred ={}
    Y_true ={}
    model_lst = []
    #loss = 0
    new_dict = func_task_type(task_type)
    val_l = 0
    for epoch in range(n_epochs-1):
        # Train the model
        
        # Print Learning Rate
        
        loss = train(optimizer,model,device,epoch,train_loader)
        #print(loss)
        # Evaluate the model
        train_acc,y_pred_Tr,y_true_Tr = test(model,device,train_loader,Evaluator_CR,task_type)
        test_acc,y_pred_T,y_true_T  = test(model,device,test_loader,Evaluator_CR,task_type)
        valid_acc,y_pred_V,y_true_V  = test(model,device,valid_loader,Evaluator_CR,task_type)
        #print(valid_acc)
        scheduler.step(valid_acc['Loss'])
        #print('Epoch:', epoch,'LR:', scheduler.get_lr())
        # Collect the prediction at each epoch    
        Y_true.update({epoch : y_true_T})
        Y_pred.update({epoch : y_pred_T}) 

        # To update output data frame
        temp = update_dict(task_type,epoch,loss,train_acc,test_acc,valid_acc)
        
        for (k,v) in zip(new_dict, temp.values()):new_dict[k].append(v)
    

        # To choose best model and epoch
        model_lst.append(copy.deepcopy(model))
        BestModel,BestEpoch = best_epoch_model(model_lst,new_dict,task_type,valid_acc)
        # early_stopping(valid_loss, model)
        
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    #print(BestEpoch)
    return BestModel,BestEpoch,Y_pred,new_dict,Y_true
    
def updatePred(up_pth,i,Y_pred,BestEpoch):
    df_pred = pd.read_csv(up_pth)
    df_pred = df_pred.drop(['Unnamed: 0'], axis=1)
    df_pred[str(i)] = np.squeeze(Y_pred[BestEpoch]).tolist() 
    df_pred.to_csv(up_pth)


###########################################################
###############To get output###############################
###########################################################

import itertools

def predOutput(y_results,label,split_type,task_type,bestEpoch):
  # Converting a huge dictinary into a dataframe
  y_results_df = pd.DataFrame(y_results)
  # Selecting the data for best epoch
  y_P=y_results_df.iloc[bestEpoch[0]]
  # Coverting the data into a list containing only test values
  y_list_P1 = y_P[label+split_type[0]].ravel().tolist()
  if len(split_type) == 2:
    y_list_P0 = y_P[label+split_type[1]].ravel().tolist()
    y_list_P = y_list_P1 + y_list_P0
  else:
    y_list_P = y_list_P1

  # 
  if task_type == 'Regression':
    outputL = y_list_P
  elif task_type == 'Classification':
    if label == 'y_pred':
      y_P_array=np.array(y_list_P).reshape(-1, 1)
      y_P_t = torch.from_numpy(y_P_array)
# Pred score - Convert to zeros and ones
      m = torch.nn.Softmax(dim=1)
      output = m(y_P_t)
      output1 = output.tolist()
      flat=itertools.chain.from_iterable(output1)
      outputL = list(flat)
    else:
      outputL = y_list_P

  
  return outputL