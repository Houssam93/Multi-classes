nom_data="IMDB"


print("...Start Imports")

from sklearn.manifold import TSNE
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.pyplot as plt



if nom_data=="BBC_Sport" :
    list_cat=['atheltics','cricket','football','rugby','tennis']
    print("...Loading embedding")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print('...Embedding loaded')
    link_athletics="../Data/athletics" # business class=-2
    link_cricket="../Data/cricket" #entertainment class=-1
    link_football="../Data/football" # politics class=0
    link_rugby="../Data/rugby" # sport class=1
    link_tennis="../Data/tennis" # tech class=2
    text_to_embed=[]
    classes=[]


    path = link_athletics
    files = os.listdir(path)
    for i in files:
        f=open(link_athletics+"/"+i,'r', encoding='unicode_escape')
        t=f.read()
        text_to_embed.append(t)
        f.close()
        classes.append(-2)
        
    path = link_cricket
    files = os.listdir(path)
    for i in files:
        f=open(link_cricket+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append(-1)
        
    path = link_football
    files = os.listdir(path)
    for i in files:
        f=open(link_football+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append(0)
        
    path = link_rugby
    files = os.listdir(path)
    for i in files:
        f=open(link_rugby+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append(1)
        
    path = link_tennis
    files = os.listdir(path)
    for i in files:
        f=open(link_tennis+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append(2)

    print("...Start Embedding")

    text_embedded=embed(text_to_embed)

if nom_data=="BBC" :
    list_cat=['Business','Ent','POL','Sport','TECH']

    print("...Loading embedding")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print('...Embedding loaded')
    link_business="../Data/business" # business class=-2
    link_entertainment="../Data/entertainment" #entertainment class=-1
    link_politics="../Data/politics" # politics class=0
    link_sport="../Data/sport" # sport class=1
    link_tech="../Data/tech" # tech class=2
    text_to_embed=[]
    classes=[]

     
    path = link_business
    files = os.listdir(path)
    for i in files:
        f=open(link_business+"/"+i,'r', encoding='unicode_escape')
        t=f.read()
        text_to_embed.append(t)
        f.close()
        classes.append([1,0,0,0,0])
        
    path = link_entertainment
    files = os.listdir(path)
    for i in files:
        f=open(link_entertainment+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append([0,1,0,0,0])
        
    path = link_politics
    files = os.listdir(path)
    for i in files:
        f=open(link_politics+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append([0,0,1,0,0])
        
    path = link_sport
    files = os.listdir(path)
    for i in files:
        f=open(link_sport+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append([0,0,0,1,0])
        
    path = link_tech
    files = os.listdir(path)
    for i in files:
        f=open(link_tech+"/"+i,'r', encoding='unicode_escape')
        text_to_embed.append(f.read())
        f.close()
        classes.append([0,0,0,0,1])

    print("...Start Embedding")

    text_embedded=embed(text_to_embed)



if nom_data=="Tweets_5C" :
    file_=open("../Data/Tweets_5C.txt","r")
    file=file_.readlines()
    file_.close()


    ID=[]
    author=[]
    classes=[]
    Tweets=[]



    for i in range(len(file)):
            liste=file[i].split("\t")
            ID.append(liste[0])
            author.append(liste[1])
            classes.append(int(liste[2]))
            Tweets.append(liste[3])


    count_neg__=0
    count_neg=0
    count_neu=0
    count_pos=0
    count_pos__=0
    for i in classes:
        if i==-2:
            count_neg__+=1
        elif i==-1:
            count_neg+=1
        elif i==0:
            count_neu+=1
        elif i==1:
            count_pos+=1
        else:
            count_pos__+=1
    #print("Total :",count_neg__+count_neg+count_neu+count_pos+count_pos__,"\n")
    #print("Very Negative:", count_neg__,"\n")
    #print("Negative:", count_neg,"\n")
    #print("Neutral:", count_neu,"\n")
    #print("Positive:", count_pos,"\n")
    #print("Very Positive:", count_pos__,"\n")
    #print(len(classes))



    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print('embedding chargé')



    print('on embed les tweetsV2')
    text_embedded=list(embed(Tweets))
    print('Tweets Embeddé')
    def transfo_y(y_train):
        y_train_multi=[]
        for i in range(len(y_train)):
            if y_train[i]==-2:
                y_train_multi.append([1,0,0,0,0])
            elif y_train[i]==-1:
                y_train_multi.append([0,1,0,0,0])
            elif y_train[i]==0:
                y_train_multi.append([0,0,1,0,0])
            elif y_train[i]==1:
                y_train_multi.append([0,0,0,1,0])
            else:
                y_train_multi.append([0,0,0,0,1])
        return y_train_multi


    list_cat=['Very_Neg','Neg','Neutral','Pos','Very_Pos']
    classes=transfo_y(classes)



if nom_data=="IMDB" :
    only_phrase=True

    
    def transfo_y(y_train):
        y_train_multi=[]
        for i in range(len(y_train)):
            if y_train[i]==0:
                y_train_multi.append([1,0,0,0,0])
            elif y_train[i]==1:
                y_train_multi.append([0,1,0,0,0])
            elif y_train[i]==2:
                y_train_multi.append([0,0,1,0,0])
            elif y_train[i]==3:
                y_train_multi.append([0,0,0,1,0])
            else:
                y_train_multi.append([0,0,0,0,1])
        return y_train_multi

    # openning the file

    df=pd.read_csv("../Data/IMDB/train.tsv",sep="\t")

    print("...Start Embedding")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("...Embedding loaded")
    text_embedded=[]
    classes=[]
    if only_phrase:
        i=1
        for row in df.itertuples():
            if row.SentenceId==i:
                text_embedded.append(row.Phrase)
                classes.append(row.Sentiment)
                i+=1
    else:
        text_embedded=df["Phrase"]
        classes=df["Sentiment"]
    text_embedded=list(embed(text_embedded))
    classes=transfo_y(classes)
    print("...Embedding ended")

    print("len classes ",len(classes))
    print("len text ", len(text_embedded))



    list_cat=['negative','somewhat negative','neutral','somewhat positive','positive']



if nom_data=="linkedin" :

    def transfo_y(y_train):
        y_train_multi=[]
        for i in range(len(y_train)):
            if y_train[i]==0:
                y_train_multi.append([1,0,0,0])
            elif y_train[i]==1:
                y_train_multi.append([0,1,0,0])
            elif y_train[i]==2:
                y_train_multi.append([0,0,1,0])
            elif y_train[i]==3:
                y_train_multi.append([0,0,0,1])
        return y_train_multi


    print("...Start Embedding")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("...Embedding loaded")


    config = pd.read_csv("../Data/Linkedin/train.txt",sep='\t')
    turn1=[]
    turn2=[]
    turn3=[]
    classes=[]
    for row in config.itertuples():
        turn1.append(row.turn1)
        turn2.append(row.turn2)
        turn3.append(row.turn3)
        if row.label=='angry':
            classes.append(0)
        elif row.label=='happy':
            classes.append(1)
        elif row.label=='sad':
            classes.append(2)
        else:
            classes.append(3)
    classes=transfo_y(classes)


    turn1=list(embed(turn1))
    turn2=list(embed(turn2))
    turn3=list(embed(turn3))


    text_embedded=[]
    for i in range(len(turn1)):
        m=[]
        for j in range(len(turn1[i])):
            m.append(float(turn1[i][j]))
        for j in range(len(turn2[i])):
            m.append(float(turn2[i][j]))
        for j in range(len(turn3[i])):
            m.append(float(turn3[i][j]))
        text_embedded.append(m)

    list_cat=['angry','happy','sad','others']
















print("...Text embedded succesfully")


print("...Spliting the data")

print("...Data is ready!")

def plot_TSNE_5C(nom_data,text_embedded,classes,list_cat):
    tsne = TSNE()
    X=text_embedded
    y=list(classes)
    colors=['r','g','b','c','m']
    print('on fait tsne')
    X_2d=tsne.fit_transform(X)
    print('on fait les courbes')
    X_2d0=[]
    X_2d1=[]
    X_2d2=[]
    X_2d3=[]
    X_2d4=[]
    for i in range(len(y)):
            if y[i]==-2 or y[i]==[1,0,0,0,0] :
                X_2d0.append(X_2d[i])
            elif y[i]==-1 or y[i]==[0,1,0,0,0]:
                X_2d1.append(X_2d[i])
            elif y[i]==0 or y[i]==[0,0,1,0,0]:
                X_2d2.append(X_2d[i])
            elif y[i]==1 or y[i]==[0,0,0,1,0]:
                X_2d3.append(X_2d[i])
            else:
                X_2d4.append(X_2d[i])
            
                
    X_2d0=np.array(X_2d0)
    X_2d1=np.array(X_2d1)
    X_2d2=np.array(X_2d2)
    X_2d3=np.array(X_2d3)
    X_2d4=np.array(X_2d4)

    plt.figure()
        
    i=0
    plt.scatter(X_2d0[:,0], X_2d0[:,1],c=colors[i],label=list_cat[i])
    i=1
    plt.scatter(X_2d1[:,0], X_2d1[:,1],c=colors[i],label=list_cat[i])
    i=2
    plt.scatter(X_2d2[:,0], X_2d2[:,1],c=colors[i],label=list_cat[i])
    i=3
    plt.scatter(X_2d3[:,0], X_2d3[:,1],c=colors[i],label=list_cat[i])
    i=4
    plt.scatter(X_2d4[:,0], X_2d4[:,1],c=colors[i],label=list_cat[i])

    plt.legend()
    plt.title("Data viz 2D "+nom_data)
    plt.show()
    plt.savefig("../images/"+"Data_viz_2D "+nom_data+".png")

    tsne = TSNE(n_components=3)
    X=text_embedded
    y=list(classes)
    X_2d=tsne.fit_transform(X)
    X_2d0=[]
    X_2d1=[]
    X_2d2=[]
    X_2d3=[]
    X_2d4=[]
    for i in range(len(y)):
            if y[i]==-2 or y[i]==[1,0,0,0,0] :
                X_2d0.append(X_2d[i])
            elif y[i]==-1 or y[i]==[0,1,0,0,0]:
                X_2d1.append(X_2d[i])
            elif y[i]==0 or y[i]==[0,0,1,0,0]:
                X_2d2.append(X_2d[i])
            elif y[i]==1 or y[i]==[0,0,0,1,0]:
                X_2d3.append(X_2d[i])
            else:
                X_2d4.append(X_2d[i])
            
                
    X_2d0=np.array(X_2d0)
    X_2d1=np.array(X_2d1)
    X_2d2=np.array(X_2d2)
    X_2d3=np.array(X_2d3)
    X_2d4=np.array(X_2d4)

    fig=plt.figure()
    colors=['r','g','b','c','m']
    ax=fig.add_subplot(111,projection='3d')
    i=0
    ax.scatter(X_2d0[:,0], X_2d0[:,1],X_2d0[:,2],c=colors[i],label=list_cat[i])
    i=1
    ax.scatter(X_2d1[:,0], X_2d1[:,1],X_2d1[:,2],c=colors[i],label=list_cat[i])
    i=2
    ax.scatter(X_2d2[:,0], X_2d2[:,1],X_2d2[:,2],c=colors[i],label=list_cat[i])
    i=3
    ax.scatter(X_2d3[:,0], X_2d3[:,1],X_2d3[:,2],c=colors[i],label=list_cat[i])
    i=4
    ax.scatter(X_2d4[:,0], X_2d4[:,1],X_2d4[:,2],c=colors[i],label=list_cat[i])
    plt.legend()
    plt.title("Data viz 3D "+nom_data)
    plt.show()
    plt.savefig("../images/"+"Data_viz_3D "+nom_data+".png")


plot_TSNE_5C(nom_data,text_embedded,classes,list_cat)