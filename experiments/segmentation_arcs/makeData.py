import encoding,os,math,threading,random
from shutil import copyfile
# Get the model
model = encoding.models.get_model('deeplab_resnest101_pcontext', pretrained=True).cpu()
model.eval()

def processData(tmpName):
    oriTileDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\OriginTileData"
    maskTileDir = "F:\\色林错\\dataSet\\" + str(tmpName) + r"\MaskTileData"
    trainDir = r"F:\色林错\dataSet\train"
    valDir = r"F:\色林错\dataSet\validation"
    
    if not os.path.exists(trainDir):
        os.makedirs(trainDir+r"\data")
        os.makedirs(trainDir+r"\mask")
    
    if not os.path.exists(valDir):
        os.makedirs(valDir+r"\data")
        os.makedirs(valDir+r"\mask")

    length = dataSet[tmpName]["length"]
    for i in range(length):
        targetDir = valDir if random.random() >= 0.9 else trainDir
        filename = oriTileDir + "\\" + str(i) +".tif"
        distname = targetDir + "\\data\\" + str(tmpName) + "_" + str(i) + ".tif"
        copyfile(src=filename,dst=distname)
        filename = maskTileDir + "\\" + str(i) +".tif"
        distname = targetDir + "\\mask\\" + str(tmpName) + "_" + str(i) + ".tif"
        copyfile(src=filename,dst=distname)


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

threads = []

for tmpName in dataSet:
    thread = myThread(tmpName)
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()
print("完成")
