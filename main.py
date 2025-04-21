import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

import knn # 同目录

dataset_dir = "data_set"

if __name__ == "__main__":
    
    dataset = {}

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if os.path.isdir(label_path):
            print(f"label \"{label}\" detected, importing images")
            image_paths = []
            for filename in os.listdir(label_path):
                if not filename.endswith('.bmp'):
                    print(f"Unexpected file format detected: {filename}")
                file_path = os.path.join(label_path, filename)
                image_paths.append(file_path)

            
            if image_paths:
                dataset[label] = image_paths
            else:
                print(f"No \".bmp\" files found in {label}")
    
    if dataset:

        knn_model = knn.knn(dataset,3)
        paths_test, predictions = knn_model.run()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        
        print("预测结果：")
        for i, path, prediction in zip(range(len(paths_test)), paths_test, predictions):
            print(f"样本{i}: {path} -> {prediction}, {'正确' if prediction in path else '错误'}")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(knn_model.confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()