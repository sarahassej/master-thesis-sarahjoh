import pickle
import numpy as np
import torch
import os
import natsort


# Find minimum length of numpy feature-vectors (axis=1), all vectors have 173 in first dimension
def get_min_length(rootdir):
    min_length = 100000
    for subdir, dirs, files in os.walk(rootdir):
        if "x_data" in subdir:
            for file in files:
                filepath = subdir + os.sep + file
                data_length = np.load(filepath).shape[1]
                if data_length < min_length:
                    min_length = data_length
    return min_length


def remove_surprise_disgust(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if int(file[7]) > 6:
                os.remove(os.path.join(subfolder, file))
                # print(os.path.join(subfolder, file))


# For all feature-vectors, slice to (173, min_length)
def chop_files(rootdir, length):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(".npy") and "x_data" in filepath:
                x = np.load(filepath)
                x = x[:, :length]
                np.save(filepath, x)


# Function to create a tensor of all feature-vectors in a folder
def create_tensor(folder):
    files = natsort.natsorted([file for file in os.listdir(folder) if file.endswith(".npy")])
    features = [torch.from_numpy(np.load(os.path.join(folder, file))) for file in files]
    tensor = torch.stack(features).float()
    print(tensor.shape)
    for f in files:
        # print(os.path.join(folder, f))
        os.remove(os.path.join(folder, f))
    torch.save(tensor, os.path.join(folder, "x.pt"))


# Function calling create_tensor for all folders containing feature-vectors to be combined to tensor
def create_tensors(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        if "x_data" in subdir:
            create_tensor(subdir)


def create_actor_index_dict(folder):
    actors = np.load(folder + 'actors.npy')
    actor_index_dict = {}
    f = open(folder + 'actor_index_dict', "wb")
    for i in range(len(actors)):
        actor = actors[i]
        if actor not in actor_index_dict:
            actor_index_dict[actor] = [i]
        else:
            actor_index_dict[actor].append(i)
    pickle.dump(actor_index_dict, f)
    f.close()


if __name__ == "__main__":
    # Do before extracting features
    # dataset_speech = "dataset/Audio_Speech_Actors_01-24"
    # dataset_song = "dataset/Audio_Song_Actors_01-24"
    # remove_surprise_disgust(dataset_speech)
    # remove_surprise_disgust(dataset_song)

    # Do after extracting features
    # chop_length = get_min_length('processed_data')
    # chop_files('processed_data', chop_length)

    # create_tensors('1D_processed_data/v2')
    # create_actor_index_dict('1D_processed_data/v2/speech/')
    create_actor_index_dict('1D_processed_data/v2/song/')
