import numpy as np
import pywt
from PIL import Image


def w_compress(img, filepath):
    """
    运用小波变换进行图像压缩的主方法:
    将图像按RGB切片后，每个切片进行哈尔小波的取Approximation成分，再除以最大值乘以255后重建，
    将重建好的切片在第三个维度堆叠后得到完整重建的图像numpy数据，再将numpy对象转化为Image对象；
    小波变换会把图像size缩小为原始大小的一半，故最后需要resize回原始大小。
    """
    img_copy = img.copy()
    data = np.array(img_copy).astype(np.uint8)
    data = np.dstack((extract_then_rebuild(data[:, :, 0]), extract_then_rebuild(data[:, :, 1]),
                      extract_then_rebuild(data[:, :, 2])))
    image = Image.fromarray(data.astype('uint8')).convert('RGB')
    image = image.resize((img.size[0], img.size[1]))
    image.save(filepath)


def extract_then_rebuild(arr):
    """
    将数据进行二维小波变换后，取出Approximation成分，再和最大值作用后重建
    """
    coeffs = pywt.dwt2(arr, 'haar')
    cA = np.array(coeffs[0])
    cAMax = np.amax(cA)
    cA = (cA / cAMax) * 255.0
    return cA
