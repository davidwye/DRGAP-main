import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
# 定义颜色列表
colors = ["#F9E4D4", "#F5BF6F", "#C8E2E3"]
def get_data():
    datas = dict()
    for ModelName in ['llava1.5_opt4','qwen_opt3','instruct_blipv2_opt1']:
        tag = ModelName.split('_')[-1]
        ModelName = ModelName.split(f'_{tag}')[0]
        Methods = ['baseline', 'manul', 'fairPlus']
        if 'llava' in ModelName:
            Methods = ['baseline', 'manul', 'fair']
        json_data = []
        for PromptMethod in Methods:
            with open(f'results/benchmark_scores/benchmark_results_CAPTIONING_{PromptMethod}_{tag}_{ModelName}.json',"rb") as f:
                d = json.load(f)
            json_data.append(d)
        datas[ModelName]=json_data
    print(datas)
    return datas

def draw_overall(datas, imgsavepath, legendFlag=False):

    # labels_ra = ['Overall Accuracy', 'RA_avg Single', 'RA_avg Two', 'RA_avg Same Gender', 'RA_avg Diff Gender']
    # labels_gap = ['Gender Gap Single', 'Gender Gap Two', 'Gender Gap Same Gender', 'Gender Gap Diff Gender']

    labels_ra = ['Single', 'Two', 'Same', 'Diff']
    labels_gap = labels_ra
    # labels_gap = ['Single', 'Two', 'SameGender', 'DiffGender']
    models = ['Qwen2-VL', 'InstructBLIP', 'Llava-1.5']  # 调整模型顺序

    plt.rcParams['figure.dpi'] = 300
    fig, axs = plt.subplots(2,3, figsize=(8, 4))

    for data_idx in range(len(datas)):


        ax1 = axs[0][data_idx]
        ax2 = axs[1][data_idx]
        ax1.set_ylim(-1,1)

        # def yaxis_formatter(x, pos):
        #     return f"{x}%"
        # from matplotlib.ticker import FuncFormatter
        # formatter = FuncFormatter(yaxis_formatter)
        # ax1.yaxis.set_major_formatter(formatter)
        # ax2.yaxis.set_major_formatter(formatter)

        ModelName = list(datas.keys())[data_idx]
        json_data = datas[ModelName]
        labels = ['original', 'DR.GAP$_{manual}$', 'DR.GAP']
        # print(json_data)
        # 准备数据
        ra_values = [[] for _ in range(len(json_data))]
        gap_values = [[] for _ in range(len(json_data))]
        for i, data in enumerate(json_data):
            ra_values[i].extend([
                # data['resolution_bias']['all_images']['overall_accuracy'],
                data['resolution_bias']['single_person_images']['RA_avg'],
                data['resolution_bias']['two_person_images']['RA_avg'],
                data['resolution_bias']['two_person_images_same_gender']['RA_avg'],
                data['resolution_bias']['two_person_images_diff_gender']['RA_avg']
            ])
            gap_values[i].extend([
                data['resolution_bias']['single_person_images']['gender_gap'],
                data['resolution_bias']['two_person_images']['gender_gap'],
                data['resolution_bias']['two_person_images_same_gender']['gender_gap'],
                data['resolution_bias']['two_person_images_diff_gender']['gender_gap']
            ])

        # 设置条形图的位置
        x_ra = np.arange(len(labels_ra))
        x_gap = np.arange(len(labels_gap))
        width = 0.2  # 条形图的宽度

        # 绘制RA_avg条形图并在顶部添加数值
        for i in range(len(ra_values)):
            bars = ax1.bar(x_ra + i * width, ra_values[i], width, color=colors[i], label=labels[i])
            # for bar in bars:
            #     height = bar.get_height()
            #     ax1.annotate(f'{height:.2f}',
            #                  xy=(bar.get_x() + bar.get_width() / 2, height),
            #                  xytext=(0, 1),  # 3 points vertical offset
            #                  textcoords="offset points",
            #                  ha='center', va='bottom',
            #                  fontsize=4)

        # 绘制gender_gap条形图并在顶部添加数值
        for i in range(len(gap_values)):
            gap_values[i] = [abs(x) for x in gap_values[i]]
            bars = ax2.bar(x_gap + i * width, gap_values[i], width, color=colors[i], label=labels[i])
            # for bar in bars:
            #     height = bar.get_height()
                # ax2.annotate(f'{height:.2f}',
                #              xy=(bar.get_x() + bar.get_width() / 2, height),
                #              xytext=(0, 1),  # 3 points vertical offset
                #              textcoords="offset points",
                #              ha='center', va='bottom',
                #              fontsize=4)

        # 添加标签和标题

        if data_idx == 0:
            ax1.set_ylabel('Resolution Acc ↑', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Resolution Bias ↓', fontsize=10, fontweight='bold')
        if data_idx != 0:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])

        ax1.set_xticks(x_ra + width * (len(json_data) - 1) / 2)
        ax1.set_xticklabels([])
        # ax1.set_xticklabels(labels_ra)
        ax1.set_ylim(0,1.05)
        ax1.set_title(models[data_idx], fontsize=12, fontweight='bold')
        # ax1.legend()
        ax2.set_xticks(x_gap + width * (len(json_data) - 1) / 2)
        # ax2.set_xticklabels(labels_gap, rotation=15)
        ax2.set_xticklabels(labels_gap)
        # ax2.legend()
        ax2.set_ylim(0,1.05)

        if legendFlag and data_idx == 1:
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.52), fontsize=10)

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # 调整底部间距以容纳图例

    plt.savefig(imgsavepath)

def draw(datas):
    draw_overall(datas, 'resimgs/visogender_vlms_overall_1.jpg',False)
    draw_overall(datas, 'resimgs/visogender_vlms_overall_2.jpg', True)
    h = 1050
    img_crop('resimgs/visogender_vlms_overall_1.jpg', 'resimgs/visogender_vlms_overall_1.jpg', 0,0,2400, h)
    img_crop('resimgs/visogender_vlms_overall_2.jpg', 'resimgs/visogender_vlms_overall_2.jpg',0, h,2400, 1200)
    img_concat('resimgs/visogender_vlms_overall_1.jpg','resimgs/visogender_vlms_overall_2.jpg',
               'resimgs/visogender_vlms_overall.jpg')
def img_crop(inpath, outpath, x, y, w, h):
    image = Image.open(inpath)
    rect = (x,y,w,h)
    crop_image = image.crop(rect)
    crop_image.save(outpath)

def img_concat(imgpath1, imgpath2, outputpath, axis=0):

    img = Image.open(imgpath1)
    im1 = np.array(img)
    img = Image.open(imgpath2)
    im2 = np.array(img)

    im = np.concatenate((im1, im2), axis=axis)  # 纵向拼接

    img = Image.fromarray(im)
    img.save(outputpath)

if __name__ == '__main__':
    # get_data()
    draw(get_data())