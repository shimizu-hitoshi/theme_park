#!/usr/bin/env python

# -*- coding: euc-jp -*-
#coding: euc-jp 

import os, re, sys
import csv
import copy
# import    urllib2
import    math
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import    numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import    random
# import    json
import    argparse
from PIL import ImageEnhance
import glob
import    cv2
# sys.path.append('../navi_a2c')
sys.path.append('../navi_curriculum')
from edges import Edge
from collections import defaultdict

def lon2x(lon, z):
    return int((lon/180.0+1.0)*math.pow(2,z)/2.0)

def lat2y(lat,z):
    return int((-math.log(math.tan((45.0+lat/2.0)*math.pi/180))+math.pi)*math.pow(2,z)/(2*math.pi))

def lon2px(lon, z):
    R = 256.0 * math.pow(2, z) /math.pi /2
    return R * math.pi*(lon+180)/ 180.0
def lat2py(lat, z):
    R = 256.0 * math.pow(2, z) /math.pi /2
    return -R * math.log(math.tan((45.0+lat/2.0)*math.pi/180.0))

##ヒュベニ    http://blogs.yahoo.co.jp/qga03052/33991636.html
def deg2rad(deg):
    return( deg * (2 * math.pi) / 360 )

def Hubeny(lat1, lon1, lat2, lon2) :
    a =6378137.000
    b =6356752.314140
    e =math.sqrt( (a**2 - b**2) / a**2 )
    e2 =e**2
    mnum =a * (1 - e2)

    my =deg2rad((lat1+lat2) /2.0)
    dy =deg2rad( lat1-lat2)
    dx =deg2rad( lon1-lon2)
    sin =math.sin(my)
    w =math.sqrt(1.0-e2 * sin *sin)
    m =mnum /(w *w *w)
    n =a/w
    dym =dy*m
    dxncos=dx*n*math.cos(my)
    return( math.sqrt( dym**2 + dxncos**2) )

def    analyselog(file, img_bk, edges):
    img.paste(img_bk,(0,0))
    # fp = open("%s/%s" %(filedir,  file[0]))
    fp = open(file)
    draw    = ImageDraw.Draw(img)
    for line in fp:
        dat = re.split("\s+", line.strip("\r\n"))
        dat[0]    = int(dat[0])
        fr    = int(dat[1])
        to    = int(dat[2])
        dist    = float(dat[3])
        # print(dat)
        px=-1
        py=-1
        for i in range(1, len(edges.CURVE[fr][to])):
            p1    = edges.CURVE[fr][to][i-1] # LAT, LNG
            p2    = edges.CURVE[fr][to][i]
            delta    = Hubeny(p1[0], p1[1],p2[0],p2[1])
            if(delta == 0):
                px = p1[1]
                py = p1[0]
                break
            if(delta >= dist):
                rate    = dist / delta
                px    = p2[1] * rate + p1[1] * (1-rate)
                py    = p2[0] * rate + p1[0] * (1-rate)
                break
            dist -= delta
        if(px<0):
            px    = edges.CURVE[fr][to][-1][1]
            py    = edges.CURVE[fr][to][-1][0]
        # print("curve",px,py)
        px = lon2px(px,z) - min_x *256 # ｘｙ座標に変換
        py = lat2py(py,z) + math.pow(2,z)/2*256 - min_y * 256

        color    =int(255*( float(dat[4])/1.8))
        #color    =int( min(255,color))
        color    = colors[min(color, 255)]*255
        # color = (0,0,0)

        rx    = dat[0] %5 - 2
        ry    = dat[0] % 25 / 5 - 2
            
        # if(img.size[0]>2048):
        #     rx*=2
        #     ry*=2
        # print((px+rx-2, py+ry-2),(px+rx+2, py+ry+2))
        # print(int(color[0]),int(color[1]),int(color[2]))
        #draw.point((px+rx, py+ry), (int(color[0]),int(color[1]),int(color[2])))
        draw.rectangle(((px+rx-2, py+ry-2),(px+rx+2, py+ry+2)), (int(color[0]),int(color[1]),int(color[2])))
    return img

def add_shelter(img, goal_remain, navi, edges):
    # img.paste(img_bk,(0,0))
    draw    = ImageDraw.Draw(img)
    for nodeid,goalid in sorted( edges.observed_goal.items() ):
        p = edges.POINT[nodeid] # 避難所の座標
        px = p[1]
        py = p[0]

        # # 満杯になるにつれて赤くなっていく．満杯で黒
        # if goal_remain.goal_remain[goalid]==0:
        #     ratio = 0
        # else:
        #     ratio = 1. - goal_remain.goal_remain[goalid] / goal_remain.goal_capa[goalid]

        # # 満杯になるにつれて黒くなっていく
        # ratio = goal_remain.goal_remain[goalid] / goal_remain.goal_capa[goalid]

        # r = 5 # 避難所の半径
        # # print(px, py, min_x, min_y, z)
        px = lon2px(px,z) - min_x *256 # ｘｙ座標に変換
        py = lat2py(py,z) + math.pow(2,z)/2*256 - min_y * 256
        # draw.ellipse(((px-r, py-r),(px+r, py+r)), fill=(int( 255 * ratio ), 0, 0)) # red

        if nodeid not in navi:
            # 青字で残容量を表示
            text    = "%d" % goal_remain.goal_remain[goalid]
            sz    = font.getsize(text)
            draw.text((px-sz[0]/2, py-sz[1]/2), text ,  font = font, fill=(0,0,255))
        else:
            # 緑字で誘導先を表示
            text    = "%d" % navi[nodeid]
            sz    = font.getsize(text)
            draw.text((px-sz[0]/2, py-sz[1]/2), text ,  font = font, fill=(0,255,0))

    return img

def    pil2np(pil):
    #http://tatabox.hatenablog.com/entry/2013/07/21/231751
    bgr = np.asarray(pil)
    rgb = bgr[:,:,::-1].copy()
    return rgb
    #rgb = bgr[:,:,[2,1,0,3]].copy()

def read_minxy(fn):
    with open(fn) as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            min_x = int(line[0])
            min_y = int(line[1])
    return min_x, min_y

def read_stationlog(fn):
    # formatは
    # goalID \t 駅滞在人数 \t 駅到着人数累計 \t 電車乗車人数 \t 方面数␣方面:人数␣・・・
    ret = {}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            goalid = int(line[0])
            cum_arrive = int(line[2])
            ret[goalid] = cum_arrive
    return ret

def read_events(fn):
    # print(fn)
    # history_events.txt
    ret = defaultdict(dict)
    with open(fn) as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            t = int(line[0])
            command = line[1]
            if "clear" in command:
                ret[t] = defaultdict(int)
            else: # signageコマンド
                nodeid1 = int(line[2]) # from
                nodeid2 = int(line[4]) # to
                ret[t][nodeid1] = nodeid2
    # print(sorted(ret))
    # sys.exit()
    return ret

# delで誘導を削除していたときの関数
# def read_events(fn):
#     # print(fn)
#     # history_events.txt
#     ret = defaultdict(dict)
#     with open(fn) as f:
#         for line in f:
#             line = line.strip()
#             line = line.split(" ")
#             t = int(line[0])
#             command = line[1]
#             if "del" in command:
#                 nodeid = int( line[2] )
#                 ret[t][nodeid] = -1
#             else:
#                 nodeid1 = int(line[2]) # from
#                 nodeid2 = int(line[4]) # to
#                 ret[t][nodeid1] = nodeid2
#     # print(sorted(ret))
#     # sys.exit()
#     return ret

class State():
    def __init__(self, goal_capa):
        self.goal_capa = goal_capa
        self.goal_remain = goal_capa
    def update(self, stationlog, observed_goal):
        cur_goal = np.zeros(self.goal_remain.shape)
        for nodeid,goalid in sorted( observed_goal.items() ):
            cur_goal[goalid] = self.goal_capa[goalid] - stationlog[nodeid]
        self.goal_remain = cur_goal

if __name__=="__main__":
    parser  = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument('-d', '--file_dir', 
        dest ="filedir", 
        required=False,
        # default    = "results/umeda",
        default    = "results/kawaramachi",
        # default    = filedir,
        )
    parser.add_argument('-l', '--log_step', 
            dest ="step", 
            required=False,
            default = 1,
    )
    parser.add_argument('-u', '--user_dir', 
        dest ="userdir", 
        required=False,
        # default = "data/umeda",
        default = "data/kawaramachi",
        # default = filedir,
        )

    options = parser.parse_args()
    filedir    = options.filedir
    step    = int( options.step )
    userdir = options.userdir
    # fnCurve = "%s/curve.txt"%userdir
    # fnPoint = "%s/points.txt"%userdir

    # まだここを手作業で入力する必要がある

    # umeda
    z = 16
    lng        = 135.498340
    lat        = 34.705580

    # kawaramachi
    z = 16
    lng        = 135.768723
    lat        = 35.003780

    minxyfn = "%s/minxy.txt"%userdir
    min_x, min_y = read_minxy(minxyfn)
    # min_x = lon2x(lng,z)-1
    # min_y = lat2y(lat,z)-1

    imgfn = "%s/image_goal.png"%userdir
    img = Image.open(imgfn)
    # base    = Image.open("%s/%s" % (filedir, files[0]))
    base    = img.convert("RGB")
    width    = base.size[0]
    height    = base.size[1]
    print(imgfn, base.size)
    fps = 30
    fps = 5
    fourcc = cv2.VideoWriter_fourcc( 'D','I', 'V','X' )
    avifn = "%s/mov.avi"%filedir
    avi    = cv2.VideoWriter(avifn, fourcc, fps, (width, height), 1);

    # fontfile    = "nicomoji-plus_v0.9.ttf"
    fontfile = "nicomoji-plus_1.11.ttf"
    fontsize    = 14
    font        = ImageFont.truetype(fontfile, 24)
    draw         = ImageDraw.Draw(base)

    img_bk    = img.copy()

    # CURVE = read_curve(fnCurve)
    # POINT = read_point(fnPoint)
    # print(CURVE)
    # print(POINT)
    colors    = cm.jet_r(np.arange(0, 256))
    edges = Edge(userdir)
    print(edges.observed_goal)
    print(edges.goal_capa)
    goal_remain = State( edges.goal_capa ) # 最初は残容量＝容量
    # sys.exit()
    # files    =[[f , int(re.search("\d+", f).group(0))]for f in  os.listdir(filedir) if(re.search("log\d+.txt", f))]
    # files    = sorted(files, key=lambda x:x[1])
    files = sorted(glob.glob("%s/log*.txt"%filedir))
    # events = read_events("%s/history_events.txt"%filedir)
    events = read_events("%s/event.txt"%filedir)
    print(events)
    # sys.exit()
    T = 1000
    navi = {}
    for file in files:
        if "station" in file:
            stationlog = read_stationlog(file)
            goal_remain.update(stationlog, edges.observed_goal)
            # print(goal_remain.goal_remain)
            # sys.exit()
            continue
        t = int(re.search("\d+", file.split("/")[-1]).group(0))
        if len( events ) > 0:
            if t > min( events.keys() ):
                k, navi = sorted( events.items() )[0]
                del events[k]
                # print(navi)
        # t = int(re.search("\d+", file).group(0))
        # if t > T:
        #     continue
        print(file, t, navi)
        # sys.exit()
        tmp_img = analyselog(file, img_bk, edges)
        tmp_img = add_shelter(tmp_img, goal_remain, navi, edges)
        imgfn = re.sub(".txt",".png",file)
        # tmp_img.save("%s/image%06d.png" % (filedir, t))
        # tmp_img.save(imgfn)
        tmp_img = tmp_img.convert("RGB")
        base.paste(tmp_img,( 0, 0))
        # print base.size
        # print img.size
        txt     = "%02d:%02d:%02d" % (t/3600, t/60 % 60 , t % 60)
        print(t, txt, goal_remain.goal_remain)
        draw.text((0, 0),txt,  font = font, fill=(255, 255, 255 ))
        #print img
        fo = pil2np(base)
        avi.write(fo);
        # print(img.size)
    avi.release()

