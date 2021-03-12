import torchvision.transforms as transform
import os,sys
sys.path.append(r"E:\Project\PyTorch-Encoding\build\lib.win-amd64-3.6")
import torch
import encoding
from PIL import Image
import numpy as np
from encoding.models import get_segmentation_model
from encoding.nn import SyncBatchNorm
from encoding.parallel import DataParallelModel
from osgeo import gdal
import threading
gdal.AllRegister()
# Get the model
checkpoint = torch.load(
    r'E:\Project\PyTorch-Encoding\runs\arcs\deeplab\resnest269\model_best.pth.tar\model_best.pth.tar')
model = get_segmentation_model("deeplab", dataset="arcs", backbone="resnest269",
                               aux=True, se_loss=False, norm_layer=SyncBatchNorm, base_size=128, crop_size=128)
model = DataParallelModel(model).cuda()
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()


class container:
    pass


def processData(tmpName):
    tileSize = 50000
    tileHeight = 128
    tileWidth = 128
    img = gdal.Open(dataSet[tmpName]["tif"])

    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    currentImg = container()
    currentImg.width = img.RasterXSize
    currentImg.height = img.RasterYSize
    currentImg.startWidth = 512
    currentImg.startHeigh = 1024
    bandNumber = img.RasterCount
    imgData = img.ReadAsArray(0, 0, 1, 0)
    datetype = 0

    if "int8" in imgData.dtype.name:
        datetype = gdal.GDT_Byte
    elif "uint8" in imgData.dtype.name:
        datetype = gdal.GDT_Byte
    elif "int16" in imgData.dtype.name:
        datetype = gdal.GDT_Int16
    elif "uint16" in imgData.dtype.name:
        datetype = gdal.GDT_UInt16
    else:
        datetype = gdal.GDT_Float32

    tmpDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\tmpRealTest"
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)

    i = 0
    while i < tileSize:
        widthInside = (currentImg.startWidth + tileWidth) <= currentImg.width
        heightInside = (currentImg.startHeigh + tileHeight) <= currentImg.height
        if widthInside and heightInside:
            if not os.path.exists(tmpDir + "\\" + str(i) + ".png"):
                # 所求瓦片在图片内可满足
                # 生成测试原数据
                tileData = img.ReadAsArray(currentImg.startWidth, currentImg.startHeigh, tileWidth, tileHeight)
                fileName = tmpDir + "\\" + str(i) + "_ori.tif"
                dst_ds = img.GetDriver().Create(fileName, tileWidth, tileHeight, bandNumber, datetype)
                dst_ds.GetRasterBand(1).WriteArray(tileData[0])
                dst_ds.GetRasterBand(2).WriteArray(tileData[1])
                dst_ds.GetRasterBand(3).WriteArray(tileData[2])
                dst_ds.FlushCache()
                imgData = Image.new('RGB',(tileWidth,tileHeight))
                for y in range(tileHeight):
                    for x in range(tileWidth):                    
                        imgData.putpixel((x,y),( tileData[0][y][x], tileData[1][y][x], tileData[2][y][x]))
            
                imgData = input_transform(imgData)
                imgData = imgData.cuda().unsqueeze(0)
                output = model(imgData)
                output = output[0]
                predict = torch.max(output, 1)[1].cpu().numpy() + 1

                # Get color pallete for visualization
                mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
                mask.save(tmpDir + "\\" + str(i) + ".png")
            currentImg.startWidth += tileWidth
        else:
            if heightInside and widthInside != True:
                # 高度足够 宽度不足
                currentImg.startWidth = 512
                currentImg.startHeigh += tileHeight
            else:
                break
        i += 1

    gdal.Open(tmpDir + "\\00019.tif")


class myThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("开始线程：" + self.name)
        processData(self.name)
        print("退出线程：" + self.name)


dataSet = {
    "1990": {
        "tif": r"F:\色林错\landsat\1990.tif",
        "shp": r"F:\色林错\湖泊水面\湖泊水面1990OK.shp"
    },
    "2000": {
        "tif": r"F:\色林错\landsat\2000.tif",
        "shp": r"F:\色林错\湖泊水面\湖泊水面2000OK.shp"
    },
    "2010": {
        "tif": r"F:\色林错\landsat\2010.tif",
        "shp": r"F:\色林错\湖泊水面\湖泊水面2010年OK.shp"
    },
    "2020": {
        "tif": r"F:\色林错\landsat\2020-2000.tif",
        "shp": r"F:\色林错\湖泊水面\湖泊水面2020OK.shp"
    }
}

threads = []

for tmpName in dataSet:
    thread = myThread(tmpName)
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()
print("完成")
