# Detection

### 关于coco 	9.28

由于某些原因，还是加了一个关于coco格式数据集的说明

coco格式用固定格式的json进行对标注的存储，而可以直接使用pycocotools包进行数据集的准备，此外还可以提供额外的数据集信息，在调用现有接口时也会更加方便...

添加的to_coco.py是用于生成三个数据集的json文件，然后使用这些json文件，即可以生成对应的数据集，方法如下：https://blog.csdn.net/qq_34914551/article/details/103793104 在此基础上修改地址即可

实际上，coco数据集的搭建过程中，主要完成的是对json的载入，而后进行基本的操作即可。基本搭建过程参见https://www.cnblogs.com/Meumax/p/12021913.html，而后可以直接进行操作。

