# Detection

### 关于coco 	9.28

由于某些原因，还是加了一个关于coco格式数据集的说明

coco格式用固定格式的json进行对标注的存储，而可以直接使用pycocotools包进行数据集的准备，此外还可以提供额外的数据集信息，在调用现有接口时也会更加方便...

添加的to_coco.py是用于生成三个数据集的json文件，然后使用这些json文件，即可以生成对应的数据集，方法如下：https://blog.csdn.net/qq_34914551/article/details/103793104，修改地址即可



