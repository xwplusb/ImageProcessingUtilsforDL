import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def ShowImageInRowsAndColumns(images:np.array, rows=1, columns=1):
    
    if images.ndim != 4:
        raise Exception("image expected in [N, C, H, W], but found ndim not equal 4")

    if rows * columns != images.shape[0]:
        raise Exception("show image in rows and columns requires to set the row and column manuelly")

    fig = plt.figure()
    image_index_iter = 0

    for i in range(rows):
        for j in range(columns):
            a =fig.add_subplot(rows, columns, image_index_iter+1)
            a.imshow(images[image_index_iter])
            plt.axis('off')
            image_index_iter +=1
    

def UpdateImageInRowsAndColumns(images:np.array, nrows=1, ncols=1, ):
    
    fig, axes = plt.subplots(nrows, ncols)

    axes_image = []

    init_images = images[0]
    init_images_index = 0
    for axe_row in axes:
        for axe in axe_row:
            axe.axis('off')
            image_axe = axe.imshow(init_images[init_images_index])
            axes_image.append(image_axe)
            init_images_index += 1

    
    def update(frame):
        update_images_index = 0
        for img_axe in axes_image:
            img_axe.set_data(frame[update_images_index])
            update_images_index += 1
        return axes_image

    anim = animation.FuncAnimation(fig, update, frames=images, interval=200, repeat=True)

    writer = animation.FFMpegWriter(
        fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save("movie.gif", writer=writer)


a = np.random.randint(low=0, high=255, size=(12, 12,64,64,3))

UpdateImageInRowsAndColumns(a, 3, 4)
plt.show()
