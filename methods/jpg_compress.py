def j_compress(img, filepath):
    """
    JPEG压缩的主方法：
    直接调用pillow库的save方法。
    https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.save
    """
    img_copy = img.copy()
    img_copy.convert('RGB').save(filepath)
