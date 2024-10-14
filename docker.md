# MFLocator

本文件介绍如何导入并使用我们的docker镜像。

1. 下载镜像压缩包，链接：https://figshare.com/articles/dataset/MFLocator_tar/27225828 (部分浏览器显示不全，进入后请下拉页面)

2. 从压缩包中加载镜像

   ```
   sudo docker load -i MFLocator.tar
   ```

3. 用镜像创建容器

   ```
   sudo docker run --name MFLocator -idt mflocator
   ```

4. 进入容器

   ```
   sudo docker exec -it MFLocator /bin/bash
   ```

5. 激活环境

   ```
   conda activate pytorch
   ```

6. 进入MFLocator文件目录

   ```
   cd root/MFLocator/
   ```

7. 运行代码，具体请见 [README_V2.md](https://www.gitlink.org.cn/xxxdc/competition-vd/tree/master/README_V2.md)