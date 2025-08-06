import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import time
import shutil

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class GMM:
    def __init__(self, Data, K, weights=None, means=None, covars=None):
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)
        col = np.shape(self.Data)[1]
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                self.means.append(mean)
        if covars is not None:
            self.covars = covars
        else:
            self.covars = []
            for i in range(self.K):
                cov = np.eye(col) * 0.1  # 使用单位矩阵初始化
                self.covars.append(cov)

    def log_gaussian(self, x, mean, cov):
        """使用对数空间的高斯分布计算，避免数值溢出"""
        dim = len(mean)
        cov_reg = cov + np.eye(dim) * 1e-6  # 更强的正则化

        # 使用Cholesky分解提高数值稳定性
        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            cov_reg += np.eye(dim) * 1e-3
            L = np.linalg.cholesky(cov_reg)

        # 计算Cholesky分解后的对数概率
        log_det = 2 * np.sum(np.log(np.diagonal(L)))
        diff = x - mean
        solve = np.linalg.solve(L, diff)
        log_prob = -0.5 * (dim * np.log(2 * np.pi) + log_det + np.dot(solve, solve))
        return log_prob

    def GMM_EM(self):
        loglikelihood = 0
        oldloglikelihood = 1
        n_samples, dim = np.shape(self.Data)
        gammas = np.zeros((n_samples, self.K))

        iteration = 0
        max_iter = 50  # 减少最大迭代次数

        while np.abs(loglikelihood - oldloglikelihood) > 1e-6 and iteration < max_iter:
            iteration += 1
            oldloglikelihood = loglikelihood

            # E-step: 使用对数概率计算
            for n in range(n_samples):
                log_probs = []
                for k in range(self.K):
                    log_prob = np.log(self.weights[k] + 1e-300) + self.log_gaussian(self.Data[n], self.means[k],
                                                                                    self.covars[k])
                    log_probs.append(log_prob)

                # 减去最大值避免指数下溢
                max_log = np.max(log_probs)
                log_probs = np.array(log_probs) - max_log
                probs = np.exp(log_probs)

                # 归一化
                total = np.sum(probs)
                if total > 0:
                    gammas[n] = probs / total
                else:
                    gammas[n] = np.ones(self.K) / self.K

            # M-step: 更新参数
            for k in range(self.K):
                nk = np.sum(gammas[:, k])
                self.weights[k] = nk / n_samples

                # 更新均值
                self.means[k] = np.sum(gammas[:, k, np.newaxis] * self.Data, axis=0) / (nk + 1e-10)

                # 更新协方差矩阵
                diff = self.Data - self.means[k]
                weighted_diff = gammas[:, k, np.newaxis] * diff
                cov = weighted_diff.T @ diff / (nk + 1e-10)

                # 确保协方差矩阵对称
                self.covars[k] = 0.5 * (cov + cov.T) + np.eye(dim) * 1e-6

            # 计算对数似然
            loglikelihood = 0
            for n in range(n_samples):
                log_prob_sum = 0
                for k in range(self.K):
                    log_prob = np.log(self.weights[k] + 1e-300) + self.log_gaussian(self.Data[n], self.means[k],
                                                                                    self.covars[k])
                    log_prob_sum += np.exp(log_prob)
                loglikelihood += np.log(log_prob_sum + 1e-300)

        # 获取预测结果和概率
        self.prediction = np.argmax(gammas, axis=1)
        self.gammas = gammas  # 保存概率值
        return iteration


def load_image_dataset(folder_path, img_size=(32, 32), max_samples=7500):
    images = []
    labels = []
    image_paths = []  # 存储原始图像路径
    label_dict = {}
    current_label = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if len(images) >= max_samples:
                    break

                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(img_size)
                    img_array = np.array(img).flatten()
                    images.append(img_array)
                    image_paths.append(img_path)  # 保存原始路径

                    folder_name = os.path.basename(root)
                    if folder_name not in label_dict:
                        label_dict[folder_name] = current_label
                        current_label += 1
                    labels.append(label_dict[folder_name])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    if not images:
        return None, None, None

    data = np.array(images)
    labels = np.array(labels) if len(labels) > 0 else None
    return data, labels, image_paths  # 返回图像路径


def save_best_samples(image_paths, gmm_model, output_folder, top_n=10):
    """保存每个聚类中概率最高的top_n张图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取聚类概率
    gammas = gmm_model.gammas
    predictions = gmm_model.prediction

    # 为每个聚类创建文件夹
    unique_clusters = np.unique(predictions)
    for cluster_id in unique_clusters:
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}_best')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

    print(f"正在提取每个聚类的最佳{top_n}张样本...")

    # 为每个聚类选择最佳样本
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的所有样本
        cluster_indices = np.where(predictions == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        # 获取这些样本属于当前聚类的概率
        cluster_probs = gammas[cluster_indices, cluster_id]

        # 按概率降序排序
        sorted_indices = np.argsort(cluster_probs)[::-1]

        # 选择前top_n个样本
        top_indices = sorted_indices[:min(top_n, len(sorted_indices))]
        top_image_paths = [image_paths[cluster_indices[i]] for i in top_indices]

        # 复制这些图像到聚类文件夹
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}_best')
        for i, img_path in enumerate(top_image_paths):
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(cluster_folder, f"top_{i + 1}_{img_name}")
            shutil.copy2(img_path, dest_path)

        print(f"聚类 {cluster_id}: 保存了 {len(top_image_paths)} 张最佳图片")

    print(f"所有聚类的最佳样本已保存到: {output_folder}")


def visualize_clusters(data, clusters, title="Clustering Results"):
    """可视化聚类结果"""
    if len(data) < 3:
        print("Not enough data points for visualization.")
        return

    try:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)

        # 设置图例字体为Times New Roman，增大字体大小并加粗
        cbar = plt.colorbar(scatter, label='Cluster Label')
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family('Times New Roman')
            l.set_size(29)
            l.set_weight('bold')  # 加粗
        cbar.set_label('Cluster Label', fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗

        # 设置标题和坐标轴标签字体并加粗
        plt.title(title, fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗
        plt.xlabel('PCA Component 1', fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗
        plt.ylabel('PCA Component 2', fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗

        # 设置坐标轴刻度字体并加粗
        plt.xticks(fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗
        plt.yticks(fontfamily='Times New Roman', fontsize=29, weight='bold')  # 加粗

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {str(e)}")


def show_cluster_samples(data, predictions, img_size=(32, 32), n_samples=5):
    """显示每个聚类的样本图像"""
    unique_clusters = np.unique(predictions)
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        print("No clusters to display.")
        return

    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(15, 3 * n_clusters))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_indices = np.where(predictions == cluster_id)[0]

        if len(cluster_indices) > n_samples:
            cluster_indices = np.random.choice(cluster_indices, n_samples, replace=False)

        for j, idx in enumerate(cluster_indices):
            if j < n_samples:  # 确保不超出列数
                img = data[idx].reshape(img_size)

                # 处理单行情况
                if n_clusters == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                ax.imshow(img, cmap='gray')
                ax.axis('off')

                if j == 0:
                    ax.set_title(f'Cluster {cluster_id}\n({len(cluster_indices)} images)')

    plt.tight_layout()
    plt.show()


def run_main():
    # 设置图像路径和参数
    img_folder = r'D:\研究生\实验数据处理完毕（分割完毕） - 副本\data(2024.12.19)\cell_photos\stage5'  # 修改为你的图像文件夹路径
    output_folder = r'D:\研究生\聚类最佳样本'  # 聚类结果保存路径
    img_size = (32, 32)
    n_clusters = 4

    # 加载图像数据集
    print("Loading image data...")
    data, true_labels, image_paths = load_image_dataset(img_folder, img_size=img_size)

    if data is None or len(data) == 0:
        print("No images loaded. Please check the path.")
        return

    print(f"Loaded {len(data)} images, feature dimension: {data.shape[1]}")

    # 数据预处理 - 使用标准化代替归一化
    print("Preprocessing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 使用GMM聚类
    print("Starting GMM clustering...")
    start_time = time.time()
    gmm = GMM(data_scaled, K=n_clusters)
    iterations = gmm.GMM_EM()
    y_pred = gmm.prediction

    print(f"Clustering completed! Time: {time.time() - start_time:.2f}s, Iterations: {iterations}")

    # 评估聚类结果
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        try:
            ari = adjusted_rand_score(true_labels, y_pred)
            print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        except:
            print("Could not compute ARI")

    # 计算轮廓系数
    if len(np.unique(y_pred)) > 1 and len(data) > 1:
        try:
            sample_size = min(600, len(data))
            indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data_scaled[indices]
            sample_labels = y_pred[indices]
            sil_score = silhouette_score(sample_data, sample_labels)
            print(f"Silhouette Score (on {sample_size} samples): {sil_score:.4f}")
        except Exception as e:
            print(f"Error computing silhouette score: {str(e)}")
    else:
        print("Cannot compute silhouette score - only one cluster detected.")

    # 可视化聚类结果
    print("Visualizing clustering results...")
    visualize_clusters(data_scaled, y_pred, title=f"GMM Clustering (K={n_clusters})")

    # 显示每个聚类的样本图像
    print("Showing sample images from each cluster...")
    show_cluster_samples(data, y_pred, img_size=img_size)

    # 保存聚类的最佳样本到文件夹
    print("Saving best samples from each cluster...")
    save_best_samples(image_paths, gmm, output_folder, top_n=10)


if __name__ == '__main__':
    run_main()