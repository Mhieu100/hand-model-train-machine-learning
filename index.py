from PIL import Image
img = Image.open(r"D:\Downloads\hieu\data\14552.png")
angle = -35
rotate_img= img.rotate(angle, expand = True)
rotate_img.show()