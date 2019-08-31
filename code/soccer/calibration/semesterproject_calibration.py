import numpy as np
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import utils.transform as transform_util
import cv2
from skimage.morphology import label, remove_small_objects, skeletonize
from scipy import ndimage

###########################
##  get_image_gradients  ##
###########################
# takes and image as inut and returns image gradient magnitude and direction
def get_image_gradients(img):
    kernel_size = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    sobelx = cv2.Sobel(blur_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(blur_gray,cv2.CV_64F,0,1,ksize=5)
    G = np.sqrt(np.square(sobelx)+np.square(sobely))
    Gdir = np.arctan(np.divide(sobely,sobelx))

    return G, Gdir


###########################
##    get_field_green    ##
###########################
# takes an image as input and returns a homogeneous mask for the largest green area
def get_field_green(img):

    h, w = img.shape[:2]

    # extract green pixels with condition from Cuevas et. al., 2015: g = G/(R+G+B)
    field_green_condition = np.logical_and((img[:,:,1]/(img[:,:,0]+img[:,:,1]+img[:,:,2]))<0.65,(img[:,:,1]/(img[:,:,0]+img[:,:,1]+img[:,:,2]))>0.35)
    # remove small regions
    s = int(0.01*h)
    field_green = cv2.erode(field_green_condition*1.0, np.ones((s,s), dtype=np.uint8))
    # dilate to connect field parts
    field_green = cv2.dilate(field_green*1.0, np.ones((int(1.25*s), int(1.25*s)), dtype=np.uint8))
    # connected component analysis
    field_label = label(field_green)
    # keep only largest componenent
    largestCC = field_label == np.argmax(np.bincount(field_label.flat))
    # remove players
    field_close = cv2.morphologyEx(largestCC*1.0, cv2.MORPH_CLOSE, np.ones((5*s, 5*s), dtype=np.uint8))
    field_close = cv2.erode(field_close, np.ones((int(2*s), int(2*s)), dtype=np.uint8))
    #field_close = cv2.dilate(field_close, np.ones((int(s), int(s)), dtype=np.uint8))
    segmented_field = cv2.dilate(field_close, np.ones((int(s), int(s)), dtype=np.uint8))

    return segmented_field


###########################
##   get_field_markings  ##
###########################
# returns field lines from gradinet magnitude image
def get_field_markings(G_mag, field_green, mask):
    G_thresh = (G_mag*field_green>2.5)*1.0
    mask_dil = cv2.dilate(mask, np.ones((10,10), dtype=np.uint8))
    G_diff = ((G_thresh - mask_dil)>0)*1.0
    G_dil = cv2.dilate(G_diff, np.ones((2,2), dtype=np.uint8))
    G_skel = skeletonize(G_dil)*1.0
    G_dil2 = cv2.dilate(G_skel, np.ones((2,2), dtype=np.uint8))
    g_label = label(G_dil2)
    G_line = remove_small_objects(g_label,20)
    G_line = cv2.dilate(G_line*1.0, np.ones((10,10), dtype=np.uint8))
    G_line_skel = skeletonize(G_line!=0)*1.0

    return G_line_skel


###########################
##  validate_white_line  ##
###########################
# checks whether pixels from a segmented area are likely to belong to a white line
def validate_white_line(img, G_mag, G_direction, edges, field_green):

    h, w = img.shape[:2]

    # create mask from extracted edges
    edges_mask = cv2.dilate(edges, np.ones((10, 10), dtype=np.uint8))
    line_white = np.logical_and(G_mag>2.5,edges_mask)

    # get image data for mask area
    line_white = field_green * line_white
    dir_line_mask = line_white*G_direction
    validated_pixels = np.logical_and(dir_line_mask!=0, dir_line_mask > -10, dir_line_mask <10)
    # get coords of candidate pixels
    y, x = np.where(validated_pixels)
    # size of surrounding to check for each pixel
    s = 20
    l =  np.int(s/2)
    # create check mask
    check_mask = np.zeros([s,s])
    check_mask[0:1,:] = 1
    check_mask[s-1:s] = 1

    for jj in range(len(x)):
        direction = G_direction[y[jj],x[jj]]
        check_mask_i = ndimage.rotate(check_mask, direction*180/np.pi, reshape=False)>0.5

        # check that area around candidate pixel is still within image boundaries
        if (x[jj]+l > 0 and x[jj]-l > 0) and (x[jj]+l < w and x[jj]-l < w) and (y[jj]+l > 0 and y[jj]-l > 0) and (y[jj]+l < h and y[jj]-l < h):
            check_img = img[y[jj]-l:y[jj]+l, x[jj]-l:x[jj]+l,:]
            check_img_mask = np.transpose([check_mask_i]*3,[1,2,0])*check_img
            # get masked region around candidate pixel
            aa = img[y[jj]-l:y[jj]+l,x[jj]-l:x[jj]+l,:]
            bb = aa*np.transpose([check_mask_i,check_mask_i,check_mask_i],[1,2,0])
            # chromaticity criterion for green
            cc = bb[:,:,1]/(bb[:,:,0]+bb[:,:,1]+bb[:,:,2])
            dd = np.logical_and(cc>0.35, cc<0.65)
            # count how many pixels meet condition
            ee = np.sum(dd)
            check = np.logical_and(ee>2, img[y[jj],x[jj],:]> [0.2, 0.3, 0.2])
            # set pixel to 0 if condition is not met
            if not check.all():
                validated_pixels[y[jj], x[jj]] = 0
    return validated_pixels


###########################
##    plot_smooth_dof    ##
###########################
# Plot and save the rotations, translations and focal length.
# path should be set to the folder containing the 'images' folder
def plot_smooth_dof(path, filtered = 0):

    path_base = path.replace("/calib", "")

    savepath = join(path_base, 'result')
    if not exists(savepath):
        os.mkdir(savepath)

    if filtered: suffix = 'smooth'
    else: suffix = ''

    files_cal = os.listdir(path)
    files_cal.sort()
    numfiles = len(files_cal)

    Rx, Ry, Rz, fx = [],[],[],[]
    T = np.empty([0,3])
    for i in range(numfiles):
        # load estimations from 'calib' folder
        calib_val = np.load(join(path, files_cal[i])).item()
        A, R, t = calib_val['A'], calib_val['R'], calib_val['T']
        rx, ry, rz = transform_util.get_angle_from_rotation(R)
        fx = np.append(fx, A[1,1])

        Rx = np.append(Rx, rx)
        Ry = np.append(Ry, ry)
        Rz = np.append(Rz, rz)
        T = np.append(T, t.T, axis=0)

    # plot rotations
    fig, ax = plt.subplots()
    ind = np.arange(0,numfiles,1)
    ax1 = plt.subplot(1,3,1)
    ax1.plot(ind, Rx)
    ax1.set_xlabel('Frame')
    ax1.set_title('Rotation x')
    ax2 = plt.subplot(1,3,2)
    ax2.plot(ind, Ry)
    ax2.set_xlabel('Frame')
    ax2.set_title('Rotation y')
    ax3 = plt.subplot(1,3,3)
    ax3.plot(ind, Rz)
    ax3.set_xlabel('Frame')
    ax3.set_title('Rotation z')
    plt.savefig(savepath + '/R' + suffix + '.png')

    # plot translations
    fig, ax = plt.subplots()
    ax4 = plt.subplot(1,3,1)
    ax4.plot(ind, T[:,0])
    ax4.set_xlabel('Frame')
    ax4.set_title('Translation x')
    ax5 = plt.subplot(1,3,2)
    ax5.plot(ind, T[:,1])
    ax5.set_xlabel('Frame')
    ax5.set_title('Translation y')
    ax6 = plt.subplot(1,3,3)
    ax6.plot(ind, T[:,2])
    ax6.set_xlabel('Frame')
    ax6.set_title('Translation z')
    plt.savefig(savepath + '/T' + suffix + '.png')

    # plot focal length
    if not filtered:
        fig, ax = plt.subplots()
        ax7 = plt.subplot(1,1,1)
        ax7.plot(ind, fx)
        ax7.set_xlabel('Frame')
        ax7.set_title('Focal Length')
        plt.savefig(savepath + '/F' + suffix + '.png')

    #plt.show()


###########################
##   filter_parameters   ##
###########################
# Filter the estimation extrinsic parameters
# path: path to 'calib' folder
# filter_type: 'mean' or 'median', default is 'mean'
# kernel_size: number of values used for filtering, default 5
def filter_parameters(path, filter_type='mean', kernel_size = 5):

    if filter_type == 'mean':
        mean_filter = 1
    elif filter_type == 'median':
        mean_filter = 0
    else:
        print('Using mean filter.')
        mean_filter = 1

    # plot unfiltered values
    plot_smooth_dof(path, filtered=0)

    # load estimations from 'calib' folder
    files = os.listdir(path)
    files.sort()
    numfiles = len(files)

    A_tmp = []
    R_tmp = []
    T_tmp = []
    for i in range(numfiles):
        ld = np.load(join(path, files[i])).item()
        A_, R_, T_ = ld['A'], ld['R'], ld['T']
        A_tmp.append(A_)
        R_tmp.append(R_)
        T_tmp.append(T_)

    t0 = np.array(T_tmp)
    A_tmp = np.array(A_tmp)
    n = len(t0)
    smooth_t = t0

    # convert rotation matrix to rotation angles
    Rx0, Ry0, Rz0 = [], [], []
    for j in range(numfiles):
        Rangle = np.array(transform_util.get_angle_from_rotation(R_tmp[j]))
        Rx0 = np.append(Rx0, Rangle[0])
        Ry0 = np.append(Ry0, Rangle[1])
        Rz0 = np.append(Rz0, Rangle[2])

    smooth_rx = Rx0
    smooth_ry = Ry0
    smooth_rz = Rz0

    kernel_size_i = int(kernel_size/2) # only for even kernel size
    for k in range(1,n-1):
        if k<kernel_size_i:
            div = 2*k+1
            kernel_size = k
        elif k>n-1-kernel_size_i:
            div = 2*(n-1-k)+1
            kernel_size = n-k-1
        else:
            div = 2*kernel_size_i+1
            kernel_size = kernel_size_i

        # mean filtering
        if mean_filter:
            x_m = np.sum(t0[k-kernel_size:k+kernel_size+1,0])/div
            y_m = np.sum(t0[k-kernel_size:k+kernel_size+1,1])/div
            z_m = np.sum(t0[k-kernel_size:k+kernel_size+1,2])/div

            rx_m = np.sum(Rx0[k-kernel_size:k+kernel_size+1])/div
            ry_m = np.sum(Ry0[k-kernel_size:k+kernel_size+1])/div
            rz_m = np.sum(Rz0[k-kernel_size:k+kernel_size+1])/div

        # median filtering
        else:
            x_m = np.median(t0[k-kernel_size:k+kernel_size+1,0])
            y_m = np.median(t0[k-kernel_size:k+kernel_size+1,1])
            z_m = np.median(t0[k-kernel_size:k+kernel_size+1,2])

            rx_m = np.median(Rx0[k-kernel_size:k+kernel_size+1])
            ry_m = np.median(Ry0[k-kernel_size:k+kernel_size+1])
            rz_m = np.median(Rz0[k-kernel_size:k+kernel_size+1])

        smooth_t[k,0] = x_m
        smooth_t[k,1] = y_m
        smooth_t[k,2] = z_m

        smooth_rx[k] = rx_m
        smooth_ry[k] = ry_m
        smooth_rz[k] = rz_m

    # save filtered values to calib files
    for l in range(n):
        smooth_R = transform_util.Rz(smooth_rz[l]).dot(transform_util.Ry(smooth_ry[l])).dot(transform_util.Rx(smooth_rx[l]))
        np.save(join(path, files[l]), {'A': A_tmp[l,:], 'R': smooth_R, 'T': smooth_t[l,:]})

    # plot filtered parameters
    plot_smooth_dof(path, filtered=1)


###########################
##   parameter_history   ##
###########################
# Check that estimated values make sense
# def parameter_history(path, A, R, T):
#
#     def larger(b,a):
#         if a > b:
#             return b
#         else:
#             return a
#
#     files = os.listdir(path)
#     files.sort()
#     numfiles = len(files)
#     A_tmp = []
#     R_tmp = []
#     T_tmp = []
#     l = 5
#     for i in range(numfiles-l,numfiles):
#         ld = np.load(join(path, files[i])).item()
#         A_, R_, T_ = ld['A'], ld['R'], ld['T']
#         A_tmp.append(A_)
#         R_tmp.append(R_)
#         T_tmp.append(T_)
#
#     for j in range(numfiles):
#         Rangle = np.array(transform_util.get_angle_from_rotation(R_tmp[j]))
#         Rx0 = np.append(Rx0, Rangle[0])
#         Ry0 = np.append(Ry0, Rangle[1])
#         Rz0 = np.append(Rz0, Rangle[2])
#
#     A_m = sum(A_tmp)/l
#
#     t0 = np.array(T_tmp)
#
#     x_m = np.sum(t0[:,0])/l
#     y_m = np.sum(t0[:,1])/l
#     z_m = np.sum(t0[:,2])/l
#
#     rx_m = np.sum(Rx0)/l
#     ry_m = np.sum(Ry0)/l
#     rz_m = np.sum(Rz0)/l
#
#     Rangle_i = np.array(transform_util.get_angle_from_rotation(R))
#     rx_i = Rangle_i[0]
#     ry_i = Rangle_i[1]
#     rz_i = Rangle_i[2]
#
#     x_i = T[0]
#     y_i = T[1]
#     z_i = T[2]
#
#     s = 5
#     rx_1 = larger(abs(rx_m-rx_i)*s, rx_i)
#     ry_1 = larger(abs(ry_m-ry_i)*s, ry_i)
#     rz_1 = larger(abs(rz_m-rz_i)*s, rz_i)
#
#     R1 = transform_util.Rz(rz_1[l]).dot(transform_util.Ry(ry_1[l])).dot(transform_util.Rx(rx_1[l]))
#
#     T1 = []*3
#     T1[0] = larger(abs(x_m-x_i)*s, x_i)
#     T1[1] = larger(abs(y_m-y_i)*s, y_i)
#     T1[2] = larger(abs(z_m-z_i)*s, z_i)
#
#     A1 = larger(abs(A_m-A)*s, A)
#
#     return A1, R1, T1


#########################
#########################
## Other Stuff

### Segementation tests

# G_thresh = (G>2.5)*1.0
# mask_dil = cv2.dilate(mask, np.ones((10,10), dtype=np.uint8))
# G_diff = ((G_thresh - mask_dil)>0)*1.0
# G_dil = cv2.dilate(G_diff, np.ones((2,2), dtype=np.uint8))
# G_skel = skeletonize(G_dil)*1.0
# G_dil2 = cv2.dilate(G_skel, np.ones((2,2), dtype=np.uint8))
#
# dist_transf = cv2.distanceTransform((1 - G_dil2).astype(np.uint8), cv2.DIST_L2, 0)
#
# ed_dil1 = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8))
# ed_dil2 = cv2.dilate(edges, np.ones((10, 10), dtype=np.uint8))
# ed_dil3 = cv2.dilate(edges, np.ones((15, 15), dtype=np.uint8))
#
# conv = convolve2d(ed_dil2, np.ones([50,50]), mode="same")
# conv_large = (conv>1300)*1.0
# conv_dil = cv2.dilate(conv_large, np.ones((200, 200), dtype=np.uint8))
# dist_transf = dist_transf + (dist_transf*conv_dil)*5
# ed_dil11 = cv2.dilate(ed_dil1, np.ones((10, 10), dtype=np.uint8))
# ed_dil11 = cv2.erode(ed_dil11, np.ones((10, 10), dtype=np.uint8))
# # dist_transf = dist_transf-ed_dil2*30
#
# dist_transf = dist_transf - 5*dir_line_mask

# def _callback_verify_dof_result(result_):
#     path = '../data/suiden/'
#     files_cal = os.listdir(os.path.join(path, 'calib'))
#     files_cal.sort()
#     numfiles = len(files_cal)
#
#     Rx, Ry, Rz, fx = [],[],[],[]
#     T = np.empty([0,3])
#     if numfiles > 15:
#         for i in range(numfiles-10, numfiles):
#             calib_val = np.load(os.path.join(path, 'calib', files_cal[i])).item()
#             A, R, t = calib_val['A'], calib_val['R'], calib_val['T']
#             rx, ry, rz = transf_utils.get_angle_from_rotation(R)
#             fx = np.append(fx, A[1,1])
#
#             Rx = np.append(Rx, rx)
#             Ry = np.append(Ry, ry)
#             Rz = np.append(Rz, rz)
#             T = np.append(T, t.T, axis=0)
#
#         Rxm = np.mean(Rx)
#         Rxv = np.var(Rx)
#         Rym = np.mean(Ry)
#         Ryv = np.var(Ry)
#         Rzm = np.mean(Rz)
#         Rzv = np.var(Rz)
#
#         Txm = np.mean(T[0])
#         Txv = np.var(T[0])
#         Tym = np.mean(T[1])
#         Tyv = np.var(T[1])
#         Tzm = np.mean(T[2])
#         Tzv = np.var(T[2])
#
#         valRx = abs(result_[0]) < (Rxm + 2*Rxv) and abs(result_[0]) > (Rxm - 2*Rxv)
#         valRy = abs(result_[1]) < (Rym + 2*Ryv) and abs(result_[1]) > (Rym - 2*Ryv)
#         valRz = abs(result_[2]) < (Rzm + 2*Rzv) and abs(result_[2]) > (Rzm - 2*Rzv)
#         valTx = abs(result_[0]) < (Txm + 2*Txv) and abs(result_[0]) > (Txm - 2*Txv)
#         valTy = abs(result_[1]) < (Tym + 2*Tyv) and abs(result_[1]) > (Tym - 2*Tyv)
#         valTz = abs(result_[2]) < (Tzm + 2*Tzv) and abs(result_[2]) > (Tzm - 2*Tzv)
#
#         res = valRx and valRy and valRz and valTx and valTy and valTz
#     else:
#         res = True
#
#     return res
