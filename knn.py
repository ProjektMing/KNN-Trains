import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

class knn:
    
    def __init__(self, dataset, k=3, size=0.4, scale=32, random_state=42):
        """
        初始化一个用于图像分类的KNN分类器。

        参数:
        ----------
        dataset : dict
            字典，其中键为类别标签，值为属于该类别的图像路径列表。
        k : int, 可选
            分类时考虑的最近邻居数量，默认为3。
        size : float, 可选
            用作测试数据的数据集比例（介于0和1之间），默认为0.4。
        scale : int, 可选
            图像将被缩放的尺寸（scale x scale像素），默认为32。
        random_state : int, 可选
            随机数生成的种子，确保结果可重现，默认为42，生命、宇宙以及任何事情的终极答案。

        属性:
        ----------
        predictions : list
            分类后将存储模型预测结果。
        accuracy : float 或 None
            评估后将存储模型的准确率。
        confusion_matrix : array 或 None
            评估后将存储混淆矩阵。
        classification_report : str 或 None
            评估后将存储分类报告。
        model : object 或 None
            训练后将存储KNN模型。
        """
        self.dataset = dataset # dataset{ key: label, value: list of all paths to each images}
        self.k = k
        self.test_size = size # 测试集比例
        self.scale= scale # 图像缩放大小
        self.random_state = random_state # 随机种子
        self.predictions = []
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None
        self.model = None
        
    def fit(self):
        """
        训练KNN分类器模型。
        
        该方法处理数据集中的图像，将其缩放为指定大小，转换为扁平化数组，
        然后使用这些处理后的图像训练KNN分类器。
        同时划分训练集和测试集，并在测试集上进行初步预测。
        
        返回:
            self: 返回实例本身，以支持方法链式调用
        """
        images = []
        labels = []
        paths = []
        for label, image_paths in self.dataset.items():
            for image_path in image_paths:
                img = Image.open(image_path).convert('L')  # 'L'模式表示灰度图
                img = img.resize((self.scale, self.scale))
                arr = np.array(img).flatten()
                images.append(arr)
                labels.append(label)
                paths.append(image_path)
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
            np.array(images), np.array(labels), np.array(paths),
            test_size=self.test_size, random_state=self.random_state, stratify=self.labels
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.paths_train, self.paths_test = p_train, p_test

        return self
    
    def train(self):
        """
        训练KNN分类器模型。
        主打的掩耳盗铃。
        
        该方法使用训练数据拟合KNN模型。
        """
        if self.model is not None:
            raise ValueError("模型已存在")
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)
        return self
    
    def reset(self):
        """
        重置KNN模型，清除当前模型和相关数据。
        """
        self.model = None
        self.predictions = []
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None
        return self
    
    def predict(self, new_images):
        """
        预测新图像的类别
        注意：该方法仅在模型训练后可用。
        
        参数:
            new_images: 图像路径列表或单个图像路径
        
        返回:
            预测的类别标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")
            
        # 单个图像路径转换为列表
        if isinstance(new_images, str):
            new_images = [new_images]
            
        processed_images = []
        for img_path in new_images:
            img = Image.open(img_path).convert('L')
            img = img.resize((self.scale, self.scale))
            img_array = np.array(img).flatten()
            processed_images.append(img_array)
            
        # 使用训练好的模型进行预测
        return self.model.predict(np.array(processed_images))
    
    def run(self):
        """
        运行KNN分类器，训练模型并在测试集上进行预测。
        返回:
            result: 测试集的预测结果与真实标签的字典
        """
        
        # 训练模型
        self.fit().train()
        
        # 在测试集上进行预测并评估
        self.predictions = self.model.predict(self.X_test)
        
        # 计算准确率
        self.accuracy = accuracy_score(self.y_test, self.predictions)
        
        # 计算混淆矩阵
        self.confusion_matrix = confusion_matrix(self.y_test, self.predictions)
        
        # 生成分类报告
        self.classification_report = classification_report(self.y_test, self.predictions)
        
        print(f"准确度: {self.accuracy}")
        print(f"混淆矩阵:\n{self.confusion_matrix}")
        print(f"分类报告:\n{self.classification_report}")
        
        return self.paths_test, self.predictions