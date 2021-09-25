"""
 get_img_name = file_name.split('_')[0]+"_"+file_name.split('_')[1]
    print(get_img_name)

    get_img_name_folder = "img/"+get_img_name+".png"
    get_img = cv2.imread(get_img_name_folder)
    get_img = cv2.cvtColor(get_img, cv2.COLOR_BGR2GRAY)
    get_img = np.array(get_img)

    img2 = np.zeros((get_img.shape[0],get_img.shape[1],3))
    img2[:,:,0] = get_img
    img2[:,:,1] = get_img
    img2[:,:,2] = get_img
    img2 = np.array(img2,np.int32)

    heat_map = np.load(file_name)
    heat_map = cv2.resize(heat_map, (get_img.shape[0],get_img.shape[1]))

    img3 = np.zeros((get_img.shape[0],get_img.shape[1],3))
    img3[:,:,0] = heat_map
    img3[:,:,1] = heat_map
    img3[:,:,2] = heat_map

    img3 = np.array(img3*255.0,np.int32)

    plt.imshow(img3,cmap='hot')
    plt.show()

    print("Img2 = Max = ",img2.max()," Min = ",img2.min())
    print("heatmap = Max = ",heat_map.max()," Min = ",heat_map.min())
    print(img2.shape)
    plt.imshow(img2,cmap='gray')
    plt.show()

    
    merged_img = img3*0.5 + img2*0.5
    merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img2,cmap='hot')
    plt.show()
"""


import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

all_files = glob.glob("*.npy")
#print(all_files)


for file_name in all_files:
    try:
        get_img_name = file_name.split('_')[0]+"_"+file_name.split('_')[1]
        print(get_img_name)

        get_img_name_folder = "img/"+get_img_name+".png"
        get_img = cv2.imread(get_img_name_folder)
        get_img = cv2.cvtColor(get_img, cv2.COLOR_BGR2GRAY)
        get_img = np.array(get_img)

        heat_map = np.load(file_name)
        heat_map = cv2.resize(heat_map, (get_img.shape[0],get_img.shape[1]))
        cmap = plt.cm.hot
        norm = plt.Normalize(vmin=heat_map.min(), vmax=heat_map.max())
        heat_map = cmap(norm(heat_map))

        
        # img3 = np.array(img3*255.0,np.int32)

        # plt.imshow(heat_map,cmap='hot')
        # plt.show()

        # plt.imshow(get_img,cmap='gray')
        # plt.show()

        plt.imsave(str(file_name.split('.')[0]+".png"),heat_map,dpi=300)

        print("Img = Max = ",get_img.max()," Min = ",get_img.min())
        print("heatmap = Max = ",heat_map.max()," Min = ",heat_map.min())
        print(get_img.shape)
    except:
        print("Passed = > ", file_name)

    
    # merged_img = get_img*0.5 + heat_map*0.5
    # merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(merged_img,cmap='hot')
    # plt.show()

    #break

