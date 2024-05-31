
import librosa.feature
from keras._tf_keras.keras import saving

import librosa




def predict_genre(song_path):
    # Параметры получаемые из аудиосигнала(могут измениться в зависимости от настройки модели)
    num_mfcc = 13
    n_fft = 2048
    hop_length = 512
    sample_rate = 22050
    samples_per_track = sample_rate * 30
    num_segment = 10
    model = saving.load_model("model_RNN_LSTM1.keras")
    model.summary()

    classes = ["blues","classical","country","disco","hiphop",
                "jazz","metal","pop","reggae","rock"]

    class_predictions = []

    samples_per_segment = int(samples_per_track / num_segment)



    #Загрузка аудиофайла
    x, sr = librosa.load(song_path, sr = sample_rate)
    song_length = int(librosa.get_duration(filename=song_path))

    prediction_per_part = []
    parts =0
    flag = 0
    if song_length > 30:
        #print("Песня длиннее 30 секунд")
        samples_per_track_30 = sample_rate * song_length
        parts = int(song_length/30)
        samples_per_segment_30 = int(samples_per_track_30 / (parts))
        flag = 1
        #print("Песня разделена на "+str(parts)+" частей")
    elif song_length == 30:
        parts = 1
        flag = 0
    elif song_length <= 30:
        return  "Ошибка, песня длится менее 30 секунд"

    for i in range(0,parts):
        if flag == 1:
            #print("Часть песни ",i+1)
            start30 = samples_per_segment_30 * i
            finish30 = start30 + samples_per_segment_30
            y = x[start30:finish30]
            #print(len(y))
        elif flag == 0:

            #print("Длительность песни 30 секунд")
            start30 = samples_per_segment
            finish30 = start30 + samples_per_segment
            y = x[start30:finish30]

        for n in range(num_segment):
            start = samples_per_segment * n
            finish = start + samples_per_segment
            #print(len(y[start:finish]))
            mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)
            mfcc = mfcc.T
            #print(mfcc.shape)
            mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            #print(mfcc.shape)
            array = model.predict(mfcc)*100
            array = array.tolist()

            #Определить максимально подходящий жанр
        class_predictions.append(array[0].index(max(array[0])))
        genre_counts = {genre: class_predictions.count(i) for i, genre in enumerate(classes)}
        total_predict = len(class_predictions)
        genre_percent = {genre: (count/total_predict) * 100 for genre, count in genre_counts.items()}
        prediction_per_part.append(genre_percent)
        class_predictions =[]
    occurence_dict = {}
    aggregated_percentages = {}
    for part_predictions in prediction_per_part:
        for genre, percentage in part_predictions.items():
            if genre in aggregated_percentages:
                aggregated_percentages[genre] += percentage
            else:
                aggregated_percentages[genre] = percentage

    # Привести к 100 процентам
    total_percentage = sum(aggregated_percentages.values())
    for genre, percentage in aggregated_percentages.items():
        aggregated_percentages[genre] = (percentage / total_percentage) * 100
    res = {}
    newline = '\n'
    #Вероятность в процентах для каждого жанра
    for genre, percentage in aggregated_percentages.items():
       res[genre] = percentage
    return f'{newline.join(f"Жанр:{key}: Процент:{round(value)}" for key, value in res.items())}\nВ итоге: \n{max(aggregated_percentages,key=aggregated_percentages.get)}'


