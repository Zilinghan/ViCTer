import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def post_processing(data, data_add, n_people, max_new_coming):
    res = {}
    for i in range(n_people):
        res[i] = set()
        
    for i in range(n_people):
        for ts in data_add[i]:
            res[i].add(ts)

    for person in data:
        person_obj = data[person]
        for idx, face in enumerate(person_obj):
#             if idx == 0:
#                 continue
            face_obj = person_obj[face]
            for t in face_obj['frame_idx']:
                if not face_obj['face_class'] is None:
                    if face_obj['face_class'] != -1:
                        res[face_obj['face_class']].add(t)
            if face_obj['new_coming_counter'] < max_new_coming and face_obj['new_coming_counter']>0:
                for t in face_obj['new_coming_frames']:
                    if not face_obj['face_class'] is None:
                        if face_obj['face_class'] != -1:
                            res[face_obj['face_class']].add(t)
    for i in range(n_people):
        res[i] = list(res[i])
        res[i].sort()
    return res            

def draw_figure(ax, recording_list, y, total_frame):
    no_color = 'white'
    yes_color = 'black'
    for i in range(total_frame):
        if i in recording_list:
            ax.barh(y=y, width=1, left=i, height=0.5, label='True', color=yes_color)
        else:
            ax.barh(y=y, width=1, left=i, height=0.5, label='False', color=no_color)
    return

def draw_figure_2(ax, recording_list, y, total_frame):
    no_color = 'white'
    yes_color = 'black'
    show_flag = False
    has_draw = True
    draw_begin = 0
    last_item = None
    for i in recording_list:
        if not show_flag:
            ax.barh(y=y, width=i-draw_begin, left=draw_begin, height=0.5, label='False', color=no_color)
            show_flag = True
            draw_begin = i
            last_item = i
        else:
            if i != last_item + 1:
                ax.barh(y=y, width=last_item+1-draw_begin, left=draw_begin, height=0.5, label='True', color=yes_color)
                ax.barh(y=y, width=i-(last_item)+1, left=last_item+1, height=0.5, label='False', color=no_color)
                show_flag = True
                draw_begin = i
                last_item = i
            else:
                last_item = i
        ax.barh(y=y, width=last_item+1-draw_begin, left=draw_begin, height=0.5, label='True', color=yes_color)
    return

def draw_screentimes(data, total_frame, output_dir):
    output_dir = Path(output_dir)
    # create the canvas
    fig, ax = plt.subplots(figsize=(10,len(data)))
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_xlim(0, total_frame)
    
    for i in data:
        draw_figure_2(ax, data[i], i, total_frame)

    plt.savefig(str(output_dir/'output.jpg'))

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap.get(7)/cap.get(5)
    return -1

def get_csv_files(filePath):
    all_files = os.listdir(filePath)
    csv_files = []
    for file in all_files:
        if file[-3:] == 'csv':
            csv_files.append(file)
    csv_files.sort(key = lambda x:int(x[:-4]))
    return csv_files

def printable_to_seconds(t):
    hh, mm, ss, ms = [int(i) for i in t.split(':')]
    return hh*3600+mm*60+ss+(ms/(100.0))

def read_labels(csv_path, total_sec):
    fig, ax = plt.subplots(figsize=(10,2))
    no_color, yes_color = 'white', 'black'
    df = pd.read_csv(csv_path)
    row, col = df.shape
    for i in range(row):
        start, end = df.iloc[i, 0], df.iloc[i, 1]
        start_t = printable_to_seconds(start)
        end_t = printable_to_seconds(end)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_xlim(0, total_sec)
        ax.barh(y=0, width=end_t-start_t, left=start_t, height=0.5, label='True', color=yes_color)

def event_plot_setup(video_path, label_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps, total_frames = cap.get(5), cap.get(7)
    else:
        print("Not opened")
        return 
    cap.release()
    # Read the labels
    csv_files = get_csv_files(label_path)
    # Plot configurations
    fig, ax = plt.subplots(figsize=(18, 0.3*len(csv_files)*3))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(0, total_frames)
    colors = ['deepskyblue', 'violet', 'limegreen', 'tomato', 'gold', 'pink']
    # Plot the ground truth (labels)
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(label_path, file))
        row, col = df.shape
        for i in range(row):
            start, end = df.iloc[i, 0], df.iloc[i, 1]
            start_t = printable_to_seconds(start)
            end_t = printable_to_seconds(end)
            ax.barh(y=idx*3, width=(end_t-start_t)*fps, left=start_t*fps, height=0.8, label='True', color=colors[idx], alpha=0.45)
    return fig, ax

def event_plot(ax, screentime, video_path, label_path, idx=1):
    def _format_convert(times, stride=1):
        ret = []
        start = times[0]
        end = times[0]
        for curr in times[1:]:
            if curr <= end + stride:
                end = curr
            else:
                ret.append((start, end+stride))
                start = curr
                end = curr
        ret.append((start, end+stride))
        return ret
    def _draw_figure(ax, recording_list, y, color, idx):
        for start, end in recording_list:
            ax.barh(y=y, width=end-start, left=start, height=0.8, color=color, alpha=0.65+(idx-1)*0.2)
        return 
    def _compute_iou(l1, l2, total_frames, fps):
        t1 = [False for _ in range(int(total_frames)+10)]
        t2 = [False for _ in range(int(total_frames)+10)]
        for start, end in l1:
            for i in range(int(start), int(end+1)):
                t1[i] = True
        row, col = l2.shape
        for i in range(row):
            start, end = l2.iloc[i, 0], l2.iloc[i, 1]
            start = printable_to_seconds(start) * fps
            end = printable_to_seconds(end) * fps
            for i in range(int(start), int(end+1)):
                t2[i] = True
        intersection = [t1[i] and t2[i] for i in range(int(total_frames))]
        union = [t1[i] or t2[i] for i in range(int(total_frames))]
        return sum(intersection)/sum(union), sum(intersection), sum(union)
        

    colors = ['deepskyblue', 'violet', 'limegreen', 'tomato', 'gold', 'pink']
    for i in screentime:
        _draw_figure(ax, _format_convert(screentime[i]), i*3+idx, colors[i], idx)
    # Computer the IOU

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps, total_frames = cap.get(5), cap.get(7)
    else:
        print("Not opened")
        return 
    cap.release()
    # Read the labels
    csv_files = get_csv_files(label_path)
    ious = []
    intersection_count = 0
    union_count = 0
    for i, file in zip(screentime, csv_files):
        l1 = _format_convert(screentime[i])
        l2 = pd.read_csv(os.path.join(label_path, file))
        iou, intersection, union = _compute_iou(l1, l2, total_frames, fps)
        ious.append(iou)
        intersection_count += intersection
        union_count += union
    print(f"Average IoU is {intersection_count/union_count}")
    print("IoUs are: ", ious)
