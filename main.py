import argparse
import numpy as np
import pandas as pd
import time
from methods.jpg_compress import j_compress
from methods.kmeans_compress import k_compress
from methods.wave_compress import w_compress
from pathlib import Path
from PIL import Image
from utils import calc_psnr


def compress(method, img, filepath, rawpath, k=0):
    """
    计算压缩方法的单次运行时间，压缩比率及psnr值
    """
    t = time.time()
    if k > 0:
        method(img, filepath, k)
    else:
        method(img, filepath)
    t = time.time() - t
    compressed_data = np.array(Image.open(filepath)).astype(np.uint8)
    psnr = calc_psnr(compressed_data, np.array(img.copy()).astype(np.uint8))
    return t, filepath.stat().st_size / rawpath.stat().st_size * 100., psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='raw', help='the high quality images\' directory')
    parser.add_argument('--output_dir', default='compressed', help='the directory to store compressed images')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if Path.is_dir(input_dir) and Path.is_dir(output_dir):
        j_dict, k_dict, w_dict = {'JPEG': {}, 'PNG': {}}, {'JPEG': {}, 'PNG': {}}, {'JPEG': {}, 'PNG': {}}

        # kmeans方法的k值从2的4次方取到2的8次方，比较各个k值情况下运行速度及压缩比率
        k_choose = [16, 32, 64, 128, 256]

        # 我发现都压缩到JPEG格式时，小波变换、kmeans和原始jpeg压缩方法压缩比例接近，没有体现出多做这步的优势，
        # 于是增加压缩到PNG格式。可以看到kmeans确实增加了压缩比率。
        suffix_choose = ['JPEG', 'PNG']
        for p in input_dir.iterdir():
            img = Image.open(p)
            stem = p.stem
            for suffix in suffix_choose:
                suffix_dict = j_dict.get(suffix)
                suffix_dict[stem] = compress(j_compress, img, output_dir / suffix / 'J' / (stem + '.' + suffix), p)
                j_dict[suffix] = suffix_dict
                for k in k_choose:
                    suffix_dict = k_dict.get(suffix)
                    temp = suffix_dict.get(k, {})
                    temp[stem] = compress(k_compress, img, output_dir / suffix / 'K' / (str(k) + '-' + stem + '.' + suffix), p, k)
                    suffix_dict[k] = temp
                    k_dict[suffix] = suffix_dict
                suffix_dict = w_dict.get(suffix)
                suffix_dict[stem] = compress(w_compress, img, output_dir / suffix / 'W' / (stem + '.' + suffix), p)
                w_dict[suffix] = suffix_dict
        pd.DataFrame.from_dict(j_dict).to_excel('j_result.xlsx')
        pd.DataFrame.from_dict(k_dict).to_excel('k_result.xlsx')
        pd.DataFrame.from_dict(w_dict).to_excel('w_result.xlsx')

