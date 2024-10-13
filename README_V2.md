# MFLocator

本项目为完成第七届CCF开源创新大赛-智能开源漏洞工程赛-漏洞定位到补丁赛题所使用的代码及相关数据文件。



## 文件结构

本项目主要的文件结构如下所示。代码部分包含我们的方法MFLocator以及作为baseline的VCMATCH方法。

```bash
├── code/
│   ├── collect_data.py				# 爬取CVE和NVD的页面信息
│   ├── create_dataset.py			# 生成数据集
│   ├── get_token.py				# 得到有效词词袋以及IDF值
│   ├── process_data.py				# 获得手工特征生成所需的信息
│   ├── feature.py					# 生成手工特征
│   ├── encoding_module.py			# 生成语义特征的函数文件
│   ├── ranking.py					# 生成语义特征，进行特征拼接，输入模型得到代码提交最终排名
│   ├── VCMATCH_encoding_module.py	# VCMATCH生成语义特征的函数文件
│   ├── VCMATCH_ranking.py			# VCMATCH 得到代码提交最终排名
│   ├── util.py						# 自定义函数文件
├── data/
│   ├── data.csv					# 1669个漏洞CVE ID及安全补丁的代码提交哈希值
│   ├── cve_desc.csv				# 1669个漏洞的描述信息
│   ├── vuln_type_impact.json		# 漏洞类型和影响文件
├── README_V2.md
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



Baseline-VCMATCH运行流程 

1. 在pretrained_model文件夹下新建BERT文件夹，将[VCMatch](https://figshare.com/s/0f3ed11f9348e2f3a9f8?file=32403518)源代码中使用的BERT模型相关文件(data/bert_model_path)下载至该文件夹内。
1. 若项目名代码已成功运行结束，运行VCMATCH_ranking.py即可。
1. 在VCMATCH_ranking.py指定的result_path文件夹中查看性能指标文件。

