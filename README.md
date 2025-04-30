## :rocket: 背景

近年来，开源软件供应链遭受持续的软件投毒和恶意代码攻击，造成了无法估计的损失。例如，Apache Log4j2远程代码执行漏洞被认为是近10年最严重的漏洞之一，攻击者可以在目标服务器上执行任意代码和嗅探系统信息。网络安全专家认为Log4j 中的远程代码执行漏洞可能需要数月甚至数年时间才能得到妥善解决。受Log4J漏洞影响组件包括Apache的Struts2、Solr、Druid、Flink等，Github上60,644个开源项目发布321,094软件存在风险。因此，当前急需智能化技术辅助降低漏洞风险，提高漏洞工程能力，减少漏洞损失。


##  MFLocator  

:trophy:荣获 “开源代码评注赛道——智能化开源漏洞工程赛” **一等奖**

本项目为完成第七届CCF开源创新大赛-智能开源漏洞工程赛-漏洞定位到补丁赛题所使用的代码及相关数据文件。


## 文件结构

本项目主要的文件结构如下所示。代码部分包含我们的方法MFLocator以及作为baseline的VCMATCH方法。

```bash
├── code/
│   ├── collect_data.py              # 爬取CVE和NVD的页面信息
│   ├── create_dataset.py            # 生成数据集
│   ├── get_token.py                 # 得到有效词词袋以及IDF值
│   ├── process_data.py              # 获得手工特征生成所需的信息
│   ├── feature.py                   # 生成手工特征
│   ├── encoding_module.py          # 生成语义特征的函数文件
│   ├── ranking.py                   # 生成语义特征，进行特征拼接，输入模型得到代码提交最终排名
│   ├── VCMATCH_encoding_module.py    # VCMATCH生成语义特征的函数文件
│   ├── VCMATCH_ranking.py          # VCMATCH 得到代码提交最终排名
│   ├── util.py                       # 自定义函数文件
├── data/
│   ├── data.csv                     # 1669个漏洞CVE ID及安全补丁的代码提交哈希值
│   ├── cve_desc.csv                 # 1669个漏洞的描述信息
│   ├── vuln_type_impact.json        # 漏洞类型和影响文件
├── README.md                       # 赛题原始README文件
├── README_V2.md                     # 本项目README文件
├── Report.pdf                     # 本作品对应报告
├── docker.md                      # 所构建docker镜像的下载链接及运行方法
├── 队员名单.md                      # 本参赛队伍人员组成
```



## 配置运行环境

实验在Ubuntu 20.04、cuda 12.2的机器上进行。

* 创建并激活conda虚拟环境

```bash
conda create -n MFLocator python=3.9
conda activate MFLocator
```

* 安装 PyTorch, torchaudio, torchvision

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

* 安装其他依赖库

```bash
pip install gitpython
pip install Beautifulsoup4
pip install tqdm
pip install nltk
pip install scikit-learn
pip install xgboost
pip install lightgbm
pip install catboost
```



## 如何运行

项目名运行流程：

1. 配置好实验所需虚拟环境。

2. 新建gitrepo文件夹，克隆代码存储库。尝试运行以下代码，借助GitPython库克隆实验过程中需要使用的十个代码存储库。

   ```python
   import git
   from tqdm import tqdm
   
   git_urls = ('git@github.com:FFmpeg/FFmpeg.git', 
              'git@github.com:torvalds/linux.git',
              'git@github.com:ImageMagick/ImageMagick.git',
              'git@github.com:php/php-src.git',
              'git@github.com:phpmyadmin/phpmyadmin.git',
              'git@github.com:moodle/moodle.git',
              'git@github.com:wireshark/wireshark.git',
              'git@github.com:openssl/openssl.git',
              'git@github.com:jenkinsci/jenkins.git',
              'git@github.com:qemu/qemu.git')
   
   repo_dir = '../gitrepo/'
   if not os.path.exists(repo_dir):
       os.makedirs(repo_dir)
   for git_url in tqdm(git_urls):
       git.Git(repo_dir).clone(git_url)
   ```

   如果上述代码运行时出现报错，且无法解决，可以通过手动的方式克隆代码存储库至gitrepo文件夹下。

   ```bash
   # by hand
   git clone git@github.com:FFmpeg/FFmpeg.git
   git clone git@github.com:torvalds/linux.git
   git clone git@github.com:ImageMagick/ImageMagick.git
   git clone git@github.com:php/php-src.git
   git clone git@github.com:phpmyadmin/phpmyadmin.git
   git clone git@github.com:moodle/moodle.git
   git clone git@github.com:wireshark/wireshark.git
   git clone git@github.com:openssl/openssl.git
   git clone git@github.com:jenkinsci/jenkins.git
   git clone git@github.com:qemu/qemu.git
   ```

3. 新建pretrained_model文件夹，用于存放需要使用的预训练模型。

4. 在pretrained_model文件夹下新建roberta-large文件夹，将[roberta-large](https://huggingface.co/FacebookAI/roberta-large/tree/main)模型的相关文件下载至该文件夹内。

5. 依次按照collect_data.py, create_dataset.py, get_tokens.py, process_data.py, feature.py, ranking.py的顺序运行文件。

6. 在ranking.py指定的result_path文件夹中查看性能指标文件。
