import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def k_compress(img, filepath, k):
    """
    kmeans压缩的主方法：
    由于图像数据量较大，传统的kmeans方法会很慢，故选择mini batch kmeans方法。
    将图像的(R,G,B)像素组合聚类后，使用每个聚类中心的像素数据代替整个聚类的像素数据。
    假设k为16，那一张3*8比特表示的图像就被压缩为4比特表示，压缩率还是可以的。
    """
    img_copy = img.copy()
    data = np.array(img_copy).astype(np.uint8)
    data = data / 255.
    data = data.reshape((data.shape[0] * data.shape[1], 3))
    kmeans = MiniBatchKMeans(n_clusters=k, init_size=3*k, batch_size=2000)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    data_recolored = new_colors.reshape((img_copy.size[1], img_copy.size[0], 3))
    data_recolored = data_recolored * 255.
    data_recolored = np.clip(data_recolored, 0, 255)
    image = Image.fromarray(data_recolored.astype('uint8')).convert('RGB')
    image.save(filepath)
