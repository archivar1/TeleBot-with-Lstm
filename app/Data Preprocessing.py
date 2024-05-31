import json
import math
import librosa.feature
import os
import librosa


dataset_path = "Data/genres_original"
jsonpath = "data_json"

sample_rate = 22050
samples_per_track = sample_rate * 30
#Подготовка данных для обучения
def preprocess(dataset_path,json_path,num_mfcc=13,n_fft=2048,hop_length=512,num_segment=10):
    data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
    }


    samples_per_segment = int(samples_per_track / num_segment)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):

        if dirpath != dataset_path:

            #Добавление всех меток
            label = str(dirpath).split('\\')[-1]
            data["mapping"].append(label)

            #print("\nInside ",label)


            for f in filenames:
                file_path = dataset_path +"/" + str(label) + "/" + str(f)
                y, sr = librosa.load(file_path, sr = sample_rate)

                #Деление всех аудиофайлов на сегментов из 5 частей
                for n in range(num_segment):
                    start = samples_per_segment * n
                    finish = start + samples_per_segment
                    #print(start,finish)
                    mfcc = librosa.feature.mfcc(y=y[start:finish],sr =sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)
                    mfcc = mfcc.T


                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        #print("Назваание трека ", file_path, n+1)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)


if __name__ == "__main__":
    preprocess(dataset_path,jsonpath,num_segment=10)
