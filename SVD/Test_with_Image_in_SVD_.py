import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# class that plot the image
class Plotting:
    def __init__(self, imag):
        self.i = imag

    # method that plots an image, passing by parameter the title
    def _plotar(self, title):
        plt.figure(figsize= (12, 8))
        plt.imshow(self.i)
        plt.title('Title: {}'.format(title))
        plt.show()
    
    # method that plots an image, passing by parameter the color and the title 
    def _plotar_with_color(self, color, title):
        plt.figure(figsize= (12, 8))
        plt.imshow(self.i, cmap= color)
        plt.title('Title: {}'.format(title))
        plt.show()

# Calss that run the conversion
class Conversion:
    def __init__(self, imagen):
        self.imag = imagen
    
    # method that open image
    def _open_img(self): 
        self.img = Image.open(self.imag)
        return self.img
    
    # method that convert the imagen for white and black
    def _convert_black_white(self):
        self.img_gray = self.img.convert('LA')
        return self.img_gray

    # method that last convert the imagen for numpy array
    def _no_convert_to_array_numpy(self):
        self.imag_array = np.array(list(self.img_gray.getdata(band= 0)), float)
        self.imag_array.shape = (self.img_gray.size[1], self.img_gray.size[0], self.img_gray.size[2])
    
    # method that apply technique SVD in image 
    def _applying_SVD_in_img(self):
        self.U, self.D, self.V = np.linalg.svd(self.imag_array)
        return self.U, self.D, self.V

    # methof that reconstif the image
    def _reconsting_img(self):
        reconsting = np.matrix(self.U[:,:1]) * np.diag(self.D[:1]) * np.matrix(self.V[:1,:])
        return reconsting
    
    # method that _reconsting image in diferent value
    def _reconsting_img_diferent_value(self):
        for i in [ 5, 10, 15, 20, 30, 50, 60, 70, 80, 90]:
            reconsting = np.matrix(self.U[:, :i]) * np.diag(self.D[:i]) * np.matrix(self.V[:i,:])
            Plotting(reconsting)._plotar_with_color('gray', i)

con = Conversion('Neural_network/sunflower.jpg')
s = con._open_img()
# Plotting._plotar(s, 1)
con._convert_black_white()
con._no_convert_to_array_numpy()

# con._applying_SVD_in_img()
# con._reconsting_img()
# con._reconsting_img_diferent_value()