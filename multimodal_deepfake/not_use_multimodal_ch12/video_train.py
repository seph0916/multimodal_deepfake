import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from model.SimpleCNN import SimpleCNN
from model.resnet import ResNet


image_data_dir='/home/josephlee/multimodal/image'
# 데이터 전처리 설정
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로딩
image_datasets = datasets.ImageFolder(image_data_dir, transform=data_transforms)
# 데이터셋의 클래스 확인
class_names = image_datasets.classes
# 훈련을 위한 DataLoader 생성
image_train_loader = DataLoader(dataset=image_datasets, batch_size=16, shuffle=True)
# 데이터셋 크기 확인
dataset_size = len(image_datasets)

# 이미지 모델 훈련 함수 정의
def train_video_model(model, image_train_loader, optimizer, criterion,  
                      device,num_epochs=10, eval_frequency=1, log_frequency=1, save_path='/home/josephlee/multimodal/best_model'):
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_correct = 0.0
        num_total = 0.0
        for data, target in image_train_loader:
            data=data.to(device)
            curr_batch_size=data.size(0)
            num_total += curr_batch_size
            output = model(data).to(device)
            target=target.unsqueeze(1).type(torch.float32).to(device)
            batch_loss = criterion(output, target)
            batch_pred = (torch.sigmoid(output) + 0.5).int()
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            # accumulate loss
            print(batch_loss)
            total_loss += batch_loss.item() * curr_batch_size
            print(total_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # get loss for this epoch
        total_loss /= num_total
        train_acc = (num_correct / num_total) * 100
        # get training accuracy for this epoch
        # 일단 보류 train_acc = (num_correct / num_total) * 100
        
         # 평가 및 로깅
        if (epoch + 1) % eval_frequency == 0:
            avg_loss = evaluate_model(model, image_train_loader, criterion)
            if (epoch + 1) % log_frequency == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss}, Train Acc: {train_acc}")

            # 최적의 모델 저장
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict()
    # 최적의 모델 저장
    if save_path:
        if best_model_state:
            torch.save(best_model_state, save_path+'/image_best_model.pth')
            print(f"Best model saved at epoch {best_epoch} with loss: {best_loss}") 


        
def evaluate_model(model, dataloader, criterion):
    model.eval()
    num_correct = 0.0
    num_total = 0.0
    total_loss=0
    with torch.no_grad(): 
        for data, target in dataloader:
            # data=data.to(device)
            # output = model(data)
            # output = torch.sigmoid(output).to(device)
            # target=target.unsqueeze(1).type(torch.float32).to(device)
            # loss = criterion(output, target)
            # total_loss += loss.item() * data.size(0)
            # total_samples += data.size(0)
            data=data.to(device)
            curr_batch_size=data.size(0)
            num_total += curr_batch_size
            output = model(data).to(device)
            target=target.unsqueeze(1).type(torch.float32).to(device)
            batch_loss = criterion(output, target)
            batch_pred = (torch.sigmoid(output) + 0.5).int()
            num_correct += (batch_pred ==target.int()).sum(dim=0).item()
            # accumulate loss
            print(batch_loss)
            total_loss += batch_loss.item() * curr_batch_size
            print(total_loss)
    avg_loss = total_loss / num_total
    return avg_loss


device=torch.device("cuda" if torch.cuda.is_available else "cpu")
# 이미지 모델 초기화 및 훈련
video_model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(video_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss().to(device)

train_video_model(video_model, image_train_loader, optimizer, criterion, device=device)



# test 부분
        # best_model = None
        # best_acc = 0
        # model.eval()
        # num_correct = 0.0
        # num_total = 0.0
        # # save test label and predictions
        # y_true = []
        # y_pred = []

        # for batch_x, _, _, batch_y in test_loader:
        #     # get actual batch size
        #     curr_batch_size = batch_x.size(0)
        #     num_total += curr_batch_size
        #     # get batch input x
        #     batch_x = batch_x.to(self.device)
        #     # make batch label y a vector
        #     batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
        #     y_true.append(batch_y.clone().detach().int().cpu())
        #     # forward / inference
        #     batch_out = model(batch_x)
        #     # get binary prediction {0, 1}
        #     batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        #     y_pred.append(batch_pred.clone().detach().cpu())
        #     # count number of correct predictions
        #     num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

        #     # get test accuracy
        # test_acc = (num_correct / num_total) * 100
        # # get all labels and predictions
        # y_true: np.ndarray = torch.cat(y_true, dim=0).numpy()
        # y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()
        # # get auc and eer
        # test_eer = alt_compute_eer(y_true, y_pred)

        # LOGGER.info(
        #     f"[{epoch:03d}]: loss: {round(total_loss, 4)} - train acc: {round(train_acc, 2)} - test acc: {round(test_acc, 2)} - test eer : {round(test_eer, 4)}"
        # )

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     LOGGER.info(f"Best Test Accuracy: {round(best_acc, 3)}")

        #     if save_dir:
        #         # save model checkpoint
        #         save_path = save_dir / "best.pt"
        #         save_checkpoint(
        #             epoch=epoch,
        #             model=model,
        #             optimizer=optim,
        #             model_kwargs=self.__dict__,
        #             filename=save_path,
        #         )
        #         LOGGER.info(f"Best Model Saved: {save_path}")
        #         # save labels and predictions
        #         save_path = save_dir / "best_pred.json"
        #         save_pred(y_true, y_pred, save_path)
        #         LOGGER.info(f"Prediction Saved: {save_path}")
       