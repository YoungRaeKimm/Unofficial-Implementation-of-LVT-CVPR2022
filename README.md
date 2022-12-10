# CS492-projcet
## [Continual Learning with Lifelong Vision Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Learning_With_Lifelong_Vision_Transformer_CVPR_2022_paper.pdf "논문")
**Team2: Changhui Kim(20200172), Youngrae Kim(20223126)**


### Get Started
~~~
git clone https://github.com/YoungRaeKimm/CS492-project.git
cd CS492-project
pip install -r requirements.txt
~~~

### Dataset
We supports the CIFAR100, and Tinyimagenet200 datasets. Also, our code supports the auto download option. But, you should specify the dataset path through script files in **./scripts/**.

### Test
You can test our code using pretrained model. (You can download them in [Google Drive](https://drive.google.com/file/d/1BtuslR4NkxjOSaQAHwJq1iyQ7mjD5vrR/view?usp=sharing "논문")). We provide pretrained model on 10 splits, Task Incremental-Learning.
Unzip the model and move the checkpoints to **./ckpt/best_models**. If there is no directory, please make it yourself. 
Then, just run the script considering GPU number in script.
Below is example.
~~~
cd scripts
bash test_cifar.sh
~~~

### Train
Just run the scripts according to the dataset.
~~~
cd scripts
bash cifar.sh
~~~

### Citation
~~~
@inproceedings{wang2022continual,
  title={Continual Learning With Lifelong Vision Transformer},
  author={Wang, Zhen and Liu, Liu and Duan, Yiqun and Kong, Yajing and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={171--181},
  year={2022}
}
~~~


### 
