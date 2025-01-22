import matplotlib.pyplot as plt
import pandas as pd

 
def plot_PR():
    pr_csv_dict = {
        'yolov5': '../runs/train/yolov5_kmean_6402/PR_curve.csv',
        '+c2f': '../runs/train/yolov5_NWD_c2f_v8_6402/PR_curve.csv',
        'bifpn+P2': '../runs/train/yolov5_c2f_bifpn_p2_640/PR_curve.csv',
        'decoupled+odconv': '../runs/train/yolov5_c2f_odconv_bifpn_p2_Decoupled_6402/PR_curve.csv',
        '+cbam+seam (YOLO-SOMI)': '../runs/train/yolov5_c2fEcaCbam_ODConv_bifpn_p2_seam_Decoupled_6404/PR_curve.csv',
    }

     
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()   
     
    fig.savefig("../runs/test/pr.png", dpi=250)
    plt.show()

 
def plot_F1():
    f1_csv_dict = {
        'yolov5_c2fcbam_640': r'../runs/train/yolov5_c2fcbam_640/F1_curve.csv',
        'yolov5_odconv_640': r'../runs/train/yolov5_odconv_640/F1_curve.csv',
        'yolov5_c2f_p2_640': r'../runs/train/yolov5_c2f_p2_640/F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

     
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()   
     
    fig.savefig("../runs/test/F1.png", dpi=250)
    plt.show()

if __name__ == '__main__':
    plot_PR()    
     
