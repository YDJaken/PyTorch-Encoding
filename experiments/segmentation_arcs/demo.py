import torch
import os,sys
sys.path.append(r"E:\Project\PyTorch-Encoding\build\lib.win-amd64-3.6")
import encoding
from encoding.models import get_segmentation_model
from encoding.nn import SyncBatchNorm
from encoding.parallel import DataParallelModel
from osgeo import gdal
import threading
gdal.AllRegister()
# Get the model
checkpoint = torch.load(r'E:\Project\PyTorch-Encoding\runs\arcs\deeplab\resnest269\model_best.pth.tar\model_best.pth.tar')
model = get_segmentation_model("deeplab", dataset="arcs",backbone="resnest269", aux=True,se_loss=False, norm_layer=SyncBatchNorm,base_size=128, crop_size=128)
model = DataParallelModel(model).cuda()
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()


def processData(tmpName):
    oriTileDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\OriginTileData"
    # maskTileDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\MaskTileData"
    tmpDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\tmpTrainTest"
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    length = dataSet[tmpName]["length"]
    for i in range(length):

        filename = oriTileDir + "\\" + str(i) +".tif"
        img = encoding.utils.load_image(filename)
        img = img.cuda().unsqueeze(0)

        # Make prediction
        output = model(img)
        output = output[0]
        predict = torch.max(output, 1)[1].cpu().numpy() + 1

        # Get color pallete for visualization
        mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
        mask.save(tmpDir + "\\" + str(i) +".png")


class myThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("开始线程：" + self.name)
        processData(self.name)
        print("退出线程：" + self.name)


dataSet = {
    "1990": {"length": 941},
    "2000": {"length": 971},
    "2010": {"length": 1071},
    "2020": {"length": 1991}
}

# processData("1990")

threads = []

for tmpName in dataSet:
    thread = myThread(tmpName)
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()
print("完成")
